import numpy as np
from ssspy.transform import whiten
from ssspy.bss.ica import GradICA, NaturalGradICA, FastICA
from ssspy.bss.fdica import GradFDICA, NaturalGradFDICA, AuxFDICA, GradLaplaceFDICA, NaturalGradLaplaceFDICA, AuxLaplaceFDICA
from ssspy.bss.iva import GradIVA, NaturalGradIVA, FastIVA, FasterIVA, AuxIVA, GradLaplaceIVA, NaturalGradGaussIVA, NaturalGradLaplaceIVA, AuxGaussIVA, AuxLaplaceIVA, GradGaussIVA
from ssspy.transform import whiten
from ssspy.algorithm import projection_back
import scipy.signal as ss
from utils import GaussILRMA, TILRMA, GGDILRMA

def get_best_channels(full_audio, sr):
    clean_audio = full_audio[:, :3*sr]

    clean_audio.shape

    mean_am = np.mean(np.abs( clean_audio ), axis=1)

    sorted_mean_amp = sorted(mean_am, reverse=True)

    sorted_mean_amp = sorted_mean_amp[:16]
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
        return ica(waveform_mix_whitened, n_iter=100)

def runNaturalGradientDescentICA(four_channels):
    def contrast_fn(x):
        return np.log(1 + np.exp(x))

    def score_fn(x):
        return 1 / (1 + np.exp(-x))

    ica = NaturalGradICA(
        contrast_fn=contrast_fn, score_fn=score_fn, is_holonomic=True
    )

    return ica(four_channels, n_iter=100)

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

def runGradFDICA(four_channels):
    n_fft, hop_length = 4096, 2048

    def contrast_fn(y):
        return 2 * np.abs(y)

    def score_fn(y):
        denom = np.maximum(np.abs(y), 1e-10)
        return y / denom

    fdica = GradFDICA(
        step_size=1e-1,
        contrast_fn=contrast_fn,
        score_fn=score_fn,
        is_holonomic=True,
        scale_restoration=False
    )

    _, _, spectrogram_mix = ss.stft(four_channels, window="hann", nperseg=n_fft, noverlap=n_fft-hop_length)
    spectrogram_mix_whitened = whiten(spectrogram_mix)
    spectrogram_est = fdica(spectrogram_mix_whitened, n_iter=100)
    spectrogram_est = projection_back(spectrogram_est, reference=spectrogram_mix)

    _, waveform_est = ss.istft(spectrogram_est, window="hann", nperseg=n_fft, noverlap=n_fft-hop_length)

    return waveform_est

def runNaturalGradFDICA(four_channels):
    n_fft, hop_length = 4096, 2048

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

    _, _, spectrogram_mix = ss.stft(four_channels, window="hann", nperseg=n_fft, noverlap=n_fft-hop_length)
    spectrogram_est = fdica(spectrogram_mix, n_iter=100)
    _, waveform_est = ss.istft(spectrogram_est, window="hann", nperseg=n_fft, noverlap=n_fft-hop_length)

    return waveform_est

def runAuxFDICA(four_channels, ip):
    n_fft, hop_length = 4096, 2048

    def contrast_fn(y):
        return 2 * np.abs(y)

    def d_contrast_fn(y):
        return 2 * np.ones_like(y)

    fdica = AuxFDICA(
        spatial_algorithm=ip,  # "IP1, IP2, ISS1, ISS2, IPA".
        contrast_fn=contrast_fn,
        d_contrast_fn=d_contrast_fn,
        scale_restoration=False
    )

    _, _, spectrogram_mix = ss.stft(four_channels, window="hann", nperseg=n_fft, noverlap=n_fft-hop_length)

    spectrogram_mix_whitened = whiten(spectrogram_mix)
    spectrogram_est = fdica(spectrogram_mix_whitened, n_iter=20)
    spectrogram_est = projection_back(spectrogram_est, reference=spectrogram_mix)

    _, waveform_est = ss.istft(spectrogram_est, window="hann", nperseg=n_fft, noverlap=n_fft-hop_length)

    return waveform_est

def runGradBasedLaplaceFDICA(four_channels):
    n_fft, hop_length = 4096, 2048

    fdica = GradLaplaceFDICA(
        step_size=1e-1,
        is_holonomic=True,
        scale_restoration=False
    )

    _, _, spectrogram_mix = ss.stft(four_channels, window="hann", nperseg=n_fft, noverlap=n_fft-hop_length)
    spectrogram_mix_whitened = whiten(spectrogram_mix)
    spectrogram_est = fdica(spectrogram_mix_whitened, n_iter=100)
    spectrogram_est = projection_back(spectrogram_est, reference=spectrogram_mix)

    _, waveform_est = ss.istft(spectrogram_est, window="hann", nperseg=n_fft, noverlap=n_fft-hop_length)

    return waveform_est

def runNatGradBasedLaplaceFDICA(four_channels):
    n_fft, hop_length = 4096, 2048

    fdica = NaturalGradLaplaceFDICA(
        step_size=1e-1,
        is_holonomic=True,
    )

    _, _, spectrogram_mix = ss.stft(four_channels, window="hann", nperseg=n_fft, noverlap=n_fft-hop_length)
    spectrogram_mix_whitened = whiten(spectrogram_mix)
    spectrogram_est = fdica(spectrogram_mix_whitened, n_iter=100)
    spectrogram_est = projection_back(spectrogram_est, reference=spectrogram_mix)

    _, waveform_est = ss.istft(spectrogram_est, window="hann", nperseg=n_fft, noverlap=n_fft-hop_length)

    return waveform_est

def runAuxLaplaceFDICA(four_channels, ip):
    n_fft, hop_length = 4096, 2048

    fdica = AuxLaplaceFDICA(
        spatial_algorithm=ip,  # IP1/IP2.
        scale_restoration=False
    )

    _, _, spectrogram_mix = ss.stft(four_channels, window="hann", nperseg=n_fft, noverlap=n_fft-hop_length)
    spectrogram_mix_whitened = whiten(spectrogram_mix)
    spectrogram_est = fdica(spectrogram_mix_whitened, n_iter=100)
    spectrogram_est = projection_back(spectrogram_est, reference=spectrogram_mix)

    _, waveform_est = ss.istft(spectrogram_est, window="hann", nperseg=n_fft, noverlap=n_fft-hop_length)

    return waveform_est

def runGradIVA(four_channels):
    n_fft, hop_length = 4096, 2048

    def contrast_fn(y):
        return 2 * np.linalg.norm(y, axis=1)

    def score_fn(y):
        norm = np.linalg.norm(y, axis=1, keepdims=True)
        norm = np.maximum(norm, 1e-10)
        return y / norm

    iva = GradIVA(
        step_size=1e+2,
        contrast_fn=contrast_fn,
        score_fn=score_fn,
        is_holonomic=True,
        scale_restoration=False
    )

    _, _, spectrogram_mix = ss.stft(four_channels, window="hann", nperseg=n_fft, noverlap=n_fft-hop_length)
    spectrogram_mix_whitened = whiten(spectrogram_mix)
    spectrogram_est = iva(spectrogram_mix_whitened, n_iter=100)
    spectrogram_est = projection_back(spectrogram_est, reference=spectrogram_mix)

    _, waveform_est = ss.istft(spectrogram_est, window="hann", nperseg=n_fft, noverlap=n_fft-hop_length)
    return waveform_est

def runNatGradIVA(four_channels):
    n_fft, hop_length = 4096, 2048

    def contrast_fn(y):
        return 2 * np.linalg.norm(y, axis=1)

    def score_fn(y):
        norm = np.linalg.norm(y, axis=1, keepdims=True)
        norm = np.maximum(norm, 1e-10)
        return y / norm

    iva = NaturalGradIVA(
        step_size=1e-1,
        contrast_fn=contrast_fn,
        score_fn=score_fn,
        is_holonomic=True
    )

    _, _, spectrogram_mix = ss.stft(four_channels, window="hann", nperseg=n_fft, noverlap=n_fft-hop_length)
    spectrogram_est = iva(spectrogram_mix, n_iter=100)
    _, waveform_est = ss.istft(spectrogram_est, window="hann", nperseg=n_fft, noverlap=n_fft-hop_length)

    return waveform_est

def runFastIVA(four_channels):
    n_fft, hop_length = 4096, 2048

    def contrast_fn(y):
        return 2 * np.linalg.norm(y, axis=1)

    def d_contrast_fn(y):
        return 2 * np.ones_like(y)

    def dd_contrast_fn(y):
        return 2 * np.zeros_like(y)

    iva = FastIVA(
        contrast_fn=contrast_fn,
        d_contrast_fn=d_contrast_fn,
        dd_contrast_fn=dd_contrast_fn
    )

    _, _, spectrogram_mix = ss.stft(four_channels, window="hann", nperseg=n_fft, noverlap=n_fft-hop_length)
    spectrogram_est = iva(spectrogram_mix, n_iter=100)
    _, waveform_est = ss.istft(spectrogram_est, window="hann", nperseg=n_fft, noverlap=n_fft-hop_length)

    return waveform_est

def runFasterIVA(four_channels):
    n_fft, hop_length = 4096, 2048

    def contrast_fn(y):
        return 2 * np.linalg.norm(y, axis=1)

    def d_contrast_fn(y):
        return 2 * np.ones_like(y)

    iva = FasterIVA(
        contrast_fn=contrast_fn,
        d_contrast_fn=d_contrast_fn
    )

    _, _, spectrogram_mix = ss.stft(four_channels, window="hann", nperseg=n_fft, noverlap=n_fft-hop_length)
    spectrogram_est = iva(spectrogram_mix, n_iter=100)
    _, waveform_est = ss.istft(spectrogram_est, window="hann", nperseg=n_fft, noverlap=n_fft-hop_length)

    return waveform_est

def runAuxIVA(four_channels, ip):
    n_fft, hop_length = 4096, 2048

    def contrast_fn(y):
        return 2 * np.linalg.norm(y, axis=1)

    def d_contrast_fn(y):
        return 2 * np.ones_like(y)

    iva = AuxIVA(
        spatial_algorithm=ip, # IP1/IP2/ISS1/ISS2/IPA/
        contrast_fn=contrast_fn,
        d_contrast_fn=d_contrast_fn
    )

    _, _, spectrogram_mix = ss.stft(four_channels, window="hann", nperseg=n_fft, noverlap=n_fft-hop_length)
    spectrogram_est = iva(spectrogram_mix, n_iter=100)
    _, waveform_est = ss.istft(spectrogram_est, window="hann", nperseg=n_fft, noverlap=n_fft-hop_length)

    return waveform_est

def runGradBasedLaplaceIVA(four_channels):
    n_fft, hop_length = 4096, 2048

    iva = GradLaplaceIVA(
        step_size=1e+2,
        is_holonomic=True,
        scale_restoration=False
    )

    _, _, spectrogram_mix = ss.stft(four_channels, window="hann", nperseg=n_fft, noverlap=n_fft-hop_length)
    spectrogram_est = iva(spectrogram_mix, n_iter=100)
    _, waveform_est = ss.istft(spectrogram_est, window="hann", nperseg=n_fft, noverlap=n_fft-hop_length)

    return waveform_est

def runNatGradBasedLaplaceIVA(four_channels):
    n_fft, hop_length = 4096, 2048

    iva = NaturalGradLaplaceIVA(
        step_size=1e-1,
        is_holonomic=True,
    )

    _, _, spectrogram_mix = ss.stft(four_channels, window="hann", nperseg=n_fft, noverlap=n_fft-hop_length)
    spectrogram_est = iva(spectrogram_mix, n_iter=100)
    _, waveform_est = ss.istft(spectrogram_est, window="hann", nperseg=n_fft, noverlap=n_fft-hop_length)

    return waveform_est

def runAuxLaplaceIVA(four_channels, ip):
    n_fft, hop_length = 4096, 2048

    iva = AuxLaplaceIVA(
        spatial_algorithm=ip, # IP1/IP2/ISS1/ISS2/IPA/
    )

    _, _, spectrogram_mix = ss.stft(four_channels, window="hann", nperseg=n_fft, noverlap=n_fft-hop_length)
    spectrogram_est = iva(spectrogram_mix, n_iter=100)
    _, waveform_est = ss.istft(spectrogram_est, window="hann", nperseg=n_fft, noverlap=n_fft-hop_length)

    return waveform_est

def runGradBasedGauss(four_channels):
    n_fft, hop_length = 4096, 2048

    iva = GradGaussIVA(
        step_size=1e-1,
        is_holonomic=True,
        scale_restoration=False
    )

    _, _, spectrogram_mix = ss.stft(four_channels, window="hann", nperseg=n_fft, noverlap=n_fft-hop_length)
    spectrogram_est = iva(spectrogram_mix, n_iter=100)
    _, waveform_est = ss.istft(spectrogram_est, window="hann", nperseg=n_fft, noverlap=n_fft-hop_length)

    return waveform_est

def runNatGradBasedGauss(four_channels):
    n_fft, hop_length = 4096, 2048

    iva = NaturalGradGaussIVA(
        step_size=1e-1,
        is_holonomic=True,
        scale_restoration=False
    )

    _, _, spectrogram_mix = ss.stft(four_channels, window="hann", nperseg=n_fft, noverlap=n_fft-hop_length)
    spectrogram_est = iva(spectrogram_mix, n_iter=100)
    _, waveform_est = ss.istft(spectrogram_est, window="hann", nperseg=n_fft, noverlap=n_fft-hop_length)

    return waveform_est

def runAuxBasedGauss(four_channels, ip):
    n_fft, hop_length = 4096, 2048

    iva = AuxGaussIVA(
        spatial_algorithm=ip, # IP1/IP2/ISS1/ISS2/IPA/
    )

    _, _, spectrogram_mix = ss.stft(four_channels, window="hann", nperseg=n_fft, noverlap=n_fft-hop_length)
    spectrogram_est = iva(spectrogram_mix, n_iter=100)
    _, waveform_est = ss.istft(spectrogram_est, window="hann", nperseg=n_fft, noverlap=n_fft-hop_length)

    return waveform_est

def runGaussILRMA(four_channels, spatial, source):
    n_fft, hop_length = 4096, 2048

    ilrma = GaussILRMA(
        n_basis=8,
        spatial_algorithm="IP1",  # IP1/IP2/ISS1/ISS2/IPA.
        source_algorithm="MM", # MM/ME,
        domain=2,
        partitioning=True,  # w/ partitioning function
        rng=np.random.default_rng(42),
    )

    _, _, spectrogram_mix = ss.stft(four_channels, window="hann", nperseg=n_fft, noverlap=n_fft-hop_length)
    spectrogram_est = ilrma(spectrogram_mix, n_iter=100)
    _, waveform_est = ss.istft(spectrogram_est, window="hann", nperseg=n_fft, noverlap=n_fft-hop_length)

    return waveform_est

def run_t_ILRMA(four_channels, spatial, source):
    n_fft, hop_length = 4096, 2048

    ilrma = TILRMA(
        n_basis=8,
        dof=1000,
        spatial_algorithm=spatial,  # You can set "IP" instead of "IP1".
        source_algorithm=source,
        domain=2,
        partitioning=True,  # w/ partitioning function
        rng=np.random.default_rng(42),
    )

    _, _, spectrogram_mix = ss.stft(four_channels, window="hann", nperseg=n_fft, noverlap=n_fft-hop_length)
    spectrogram_est = ilrma(spectrogram_mix, n_iter=100)
    _, waveform_est = ss.istft(spectrogram_est, window="hann", nperseg=n_fft, noverlap=n_fft-hop_length)

    return waveform_est

def runGGDILRMA(four_channels, spatial):
    n_fft, hop_length = 4096, 2048

    ilrma = GGDILRMA(
        n_basis=8,
        beta=1.95,
        spatial_algorithm=spatial,  # IP1/IP2/ISS1/ISS2/IPA.
        domain=2,
        partitioning=True,  # w/ partitioning function
        normalization=False,
        rng=np.random.default_rng(42),
    )

    _, _, spectrogram_mix = ss.stft(four_channels, window="hann", nperseg=n_fft, noverlap=n_fft-hop_length)
    spectrogram_est = ilrma(spectrogram_mix, n_iter=100)
    _, waveform_est = ss.istft(spectrogram_est, window="hann", nperseg=n_fft, noverlap=n_fft-hop_length)

    return waveform_est

