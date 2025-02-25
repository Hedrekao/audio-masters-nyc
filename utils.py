from sys import argv
import numpy as np
import librosa
import soundfile as sf


def save_audio(audio, name, sr):
    sf.write(f'{ name }.wav', audio, sr, )


def spectral_subtraction(noisy, alpha):

    noisy_spectrum = np.abs(librosa.stft(noisy))
    noisy_phase = np.angle(librosa.stft(noisy))

    noise = np.mean(noisy_spectrum, axis=1, keepdims=True)

    clean_spectrum = np.maximum(noisy_spectrum - alpha * noise, 0)

    clean_spectrum = np.exp(1j * noisy_phase) * clean_spectrum
    clean_audio = librosa.istft(clean_spectrum)

    return clean_audio


def cut_audio_fn(audio, sr):
    onset_strengths = librosa.onset.onset_strength(y=audio[0], sr=sr)
    biggest_strength_idx = np.argmax(onset_strengths)
    onset_time = librosa.frames_to_time(
        biggest_strength_idx, sr=sr, hop_length=512)
    cut_audio = audio[:, :int(onset_time * sr)]

    return cut_audio


def resample_audio(audio, sr, new_sr):
    return librosa.resample(audio, orig_sr=sr, target_sr=new_sr)


if __name__ == '__main__':
    file_name = argv[1]

    audio, sr = librosa.load(f"{ file_name }.wav", sr=None)

    for i in range(2, len(argv), 2):
        type = argv[i]
        if i + 1 < len(argv):
            param = argv[i + 1]
        if type == 'resample':
            new_sr = int(argv[3])
            audio = resample_audio(audio, sr, new_sr)
            file_name = f'{ file_name }_{ new_sr }'
            sr = new_sr
        elif type == 'spectral':
            alpha = float(argv[3])
            audio = spectral_subtraction(audio, alpha)
            file_name = f'{ file_name }_spectral'
        elif type == 'cut':
            cut_audio = cut_audio_fn(audio, sr)
            file_name = f'{ file_name }_cut'
        elif type == 'extract_chan':
            audio, sr = librosa.load(f"{ file_name }.wav", sr=None, mono=False)

            cut_audio_fn(audio, sr)

            mean_am = np.mean(np.abs(cut_audio), axis=1)
            biggest_channels = np.argsort(mean_am)[::-1][:4]

            audio = audio[biggest_channels].T
            file_name = f'{ file_name }_4chan'

    save_audio(audio, file_name, sr)


def get_best_channels_ampl(audios, clean_audio):

    print(clean_audio.shape[1])

    mean_ampl = []

    # concatenated audio channels

    for chan in audios:
        print(chan.shape)
        try:
            short_chan = chan[clean_audio.shape[1]]
            mean_ampl.append(np.mean(np.abs(short_chan)))
        except BaseException:
            pass

    # get top 1 channels with highest ampl
    best_channel_idx = np.argmax(mean_ampl)
    return audios[best_channel_idx]
