import librosa
import numpy as np
import scipy.signal as ss
from ssspy.bss.fdica import NaturalGradFDICA
from ssspy.bss.ica import NaturalGradICA


def fdica(audio, n_fft=4096, hop_length=2048, n_iter=150):

    def contrast_fn(y):
        return 2 * np.abs(y)

    def score_fn(y):
        denom = np.maximum(np.abs(y), 1e-10)
        return y / denom

    fdica = NaturalGradFDICA(
        step_size=1e-1,
        contrast_fn=contrast_fn,
        score_fn=score_fn,
        is_holonomic=True,
    )
    _, _, spectrogram_mix = ss.stft(
        audio, window="hann", nperseg=n_fft, noverlap=n_fft - hop_length)

    spectrogram_est = fdica(spectrogram_mix, n_iter)
    _, waveform_est = ss.istft(
        spectrogram_est, window="hann", nperseg=n_fft, noverlap=n_fft - hop_length)

    return waveform_est


def ica(audio):

    def contrast_fn(x):
        return np.log(1 + np.exp(x))

    def score_fn(x):
        return 1 / (1 + np.exp(-x))

    ica = NaturalGradICA(
        contrast_fn=contrast_fn, score_fn=score_fn, is_holonomic=True
    )

    waveform_est = ica(audio, n_iter=500)
    return waveform_est


def main(audio_name, n_iter=1):
    audio, sr = librosa.load(audio_name, sr=None, mono=False)

    print(audio.shape)
    # audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    # print(audio.shape)

    # sr = 16000

    onset_strengths = librosa.onset.onset_strength(y=audio[0], sr=sr)

    biggest_strength_idx = np.argmax(onset_strengths)

    onset_time = librosa.frames_to_time(
        biggest_strength_idx, sr=sr, hop_length=512)

    cut_audio = audio[:, :int(onset_time * sr)]

    mean_am = np.mean(np.abs(cut_audio), axis=1)
    biggest_channels = np.argsort(mean_am)[::-1][:4]

    audio = audio[biggest_channels]

    print(audio.shape)
    for _ in range(n_iter):

        audio = fdica(audio)
        audio = ica(audio)

    return audio, sr
