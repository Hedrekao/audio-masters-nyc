import numpy as np
from ssspy.transform import whiten
from ssspy.bss.ica import GradICA, NaturalGradICA, FastICA

def get_best_channels(full_audio, sr):
    clean_audio = full_audio[:, :3*sr]

    clean_audio.shape

    mean_am = np.mean(np.abs( clean_audio ), axis=1)

    sorted_mean_amp = sorted(mean_am, reverse=True)

    sorted_mean_amp = sorted_mean_amp[:4]
    highest_amp_indieces = [i for i,val in enumerate(mean_am) if val in sorted_mean_amp]
    highest_amp_indieces

    four_channels = np.array([full_audio[i] for i in highest_amp_indieces])

    return four_channels

def runGradICA(four_channels):
        def contrast_fn(x):
            return np.log(1 + np.exp(x))

        def score_fn(x):
            return 1 / (1 + np.exp(-x))

        ica = GradICA(
            contrast_fn=contrast_fn, score_fn=score_fn, is_holonomic=True
        )

        waveform_mix_whitened = whiten(four_channels)
        return ica(waveform_mix_whitened, n_iter=500)

def runNaturalGradientDescentICA(four_channels):
    def contrast_fn(x):
        return np.log(1 + np.exp(x))

    def score_fn(x):
        return 1 / (1 + np.exp(-x))

    ica = NaturalGradICA(
        contrast_fn=contrast_fn, score_fn=score_fn, is_holonomic=True
    )

    return ica(four_channels, n_iter=500)

def runFastICA(four_channels):
    def contrast_fn(x):
        return np.log(1 + np.exp(x))

    def score_fn(x):
        return 1 / (1 + np.exp(-x))

    def d_score_fn(x):
        sigma = 1 / (1 + np.exp(-x))
        return sigma * (1 - sigma)

    ica = FastICA(
        contrast_fn=contrast_fn,
        score_fn=score_fn,
        d_score_fn=d_score_fn,
    )

    return ica(four_channels, n_iter=10)
