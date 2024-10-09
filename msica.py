import numpy as np
import scipy.signal as ss
from ssspy.bss.fdica import NaturalGradFDICA
from ssspy.bss.ica import NaturalGradICA


def fdica(audio, n_fft=1024, hop_length=512, n_iter=300):

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


def msica(audio, n_iter=1):

    audio = fdica(audio)
    audio = ica(audio)

    return audio
