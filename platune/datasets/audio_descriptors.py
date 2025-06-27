import gin
import torch
import librosa
import numpy as np
from scipy.interpolate import interp1d
import pyloudnorm as pyln

from .timbral_models import *


DISCRETE_RESAMPLING = ['f0', 'integrated_loudness']
CONTINUOUS_RESAMPLING = ['booming', 'brightness', 'depth', 'hardness', 'roughness', 'sharpness', 'warmth', 'rms', 'centroid', 'bandwidth', 'rolloff', 'flatness', 'zcr', 'loudness1s']

def compute_timbral(y: np.ndarray,
                    sr: int,
                    descriptors: list = [None],
                    mean: bool = False,
                    resampler=None) -> dict:
    """
    Compute all descriptors inside the timbral models (audio common) library

    Parameters
    ----------
    x : np.ndarray
        Input audio signal (samples)
    sr : int
        Input sample rate
    mean : bool, optional
        [TODO] : Compute the mean of descriptors

    Returns
    -------
    dict
        Dictionnary containing all features.

    """
    # Features to compute
    features_dict = {
        "booming": timbral_booming,
        "brightness": timbral_brightness,
        "depth": timbral_depth,
        "hardness": timbral_hardness,
        "roughness": timbral_roughness,
        "sharpness": timbral_sharpness,
        "warmth": timbral_warmth
    }
    # Results dict
    features = {}
    if max([(feature in descriptors)
            for feature in list(features_dict.keys())]):
        if sr < 44100:
            if resampler is None:
                y = librosa.core.resample(y, orig_sr=sr, target_sr=44100)
                sr = 44100
            else:
                # upsample file to avoid errors
                y = resampler(torch.tensor(y)).numpy()
                sr = 44100
        for name, func in features_dict.items():
            if name in descriptors:
                features[name] = func(y, fs=sr, mean=mean)
    return features


def compute_librosa(y: np.ndarray,
                    sr: int,
                    descriptors: list = [None],
                    mean: bool = False,
                    resampler=None) -> dict:
    """
    Compute all descriptors inside the Librosa library

    Parameters
    ----------
    x : np.ndarray
        Input audio signal (samples)
    sr : int
        Input sample rate
    mean : bool, optional
        [TODO] : Compute the mean of descriptors

    Returns
    -------
    dict
        Dictionnary containing all features.

    """
    # Features to compute
    features_dict = {
        "rolloff": librosa.feature.spectral_rolloff,
        "bandwidth": librosa.feature.spectral_bandwidth,
        "centroid": librosa.feature.spectral_centroid
    }
    # Results dict
    features = {}
    # Temporal features
    if "rms" in descriptors:
        features["rms"] = librosa.feature.rms(y=y)
    if "zcr" in descriptors:
        features["zcr"] = librosa.feature.zero_crossing_rate(y, center=False)
    if "f0" in descriptors:
        features["f0"] = librosa.yin(y, fmin=50, fmax=5000, sr=sr)[np.newaxis, :]
    if "flatness" in descriptors:
        features["flatness"] = librosa.feature.spectral_flatness(y=y, n_fft=2048, hop_length=512, center=False)
    # Spectral features
    # S, phase = librosa.magphase(librosa.stft(y=y))
    # Compute all descriptors

    for name, func in features_dict.items():
        if name in descriptors:
            features[name] = func(y=y, sr=sr, n_fft=2048, hop_length=512, center=False)
            # features[name] = func(S=S)
    return features


def compute_framewise_integrated_loudness(y, sr, meter, window_duration : float = 1.):
    window_size = int(sr * window_duration)  
    hop_size = window_size // 4

    short_term_loudness = []
    for i in range(0, len(y) - window_size + 1, hop_size):
        window = y[i : i + window_size]
        loudness = meter.integrated_loudness(window)  
        short_term_loudness.append(loudness)

    short_term_loudness = np.array(short_term_loudness)
    return short_term_loudness


def compute_pyloudnorm(y: np.ndarray,
                    sr: int,
                    descriptors: list = [None],
                    mean: bool = False,
                    resampler=None) -> dict:
    # Results dict
    features = {}

    # Create a meter (EBU R128 standard)
    meter = pyln.Meter(sr)
    if 'integrated_loudness' in descriptors:
        features['integrated_loudness'] = meter.integrated_loudness(y)
    if 'loudness1s' in descriptors:
        features['loudness1s'] = compute_framewise_integrated_loudness(y, sr, meter, window_duration=1.)
    return features
    

def compute_all_old(x: np.ndarray,
                sr: int,
                descriptors: list = [None],
                mean: bool = False,
                resample=None,
                resampler=None) -> dict:
    """
    Compute all descriptors inside a given dictionnary of function. This
    high-level launch computations and merge dictionnaries.
    Finally allows to resample all descriptor series to a common length.

    Parameters
    ----------
    x : np.ndarray
        Input audio signal (samples)
    sr : int
        Input sample rate
    mean : bool, optional
        [TODO] : Compute the mean of descriptors
    resample : bool, optional
        Resample all series to the maximum length found. The default is True.

    Returns
    -------
    dict
        Dictionnary containing all features.

    """
    # List of feature sub-libraries to use
    librairies = {"librosa": compute_librosa, "timbral": compute_timbral}
    # Final features
    final_features = {}
    for n, func in librairies.items():
        # Process all functions
        cur_dict = func(x, sr, descriptors, mean, resampler)
        # Merge dictionnaries
        final_features.update(cur_dict)
    # Resample all series to max length
    if resample:
        # m_len = max([x.shape[0] for x in final_features.values()])
        for n, v in final_features.items():
            if len(v.shape) > 1:
                v = v[0]
            final_features[n] = librosa.core.resample(v,
                                                      orig_sr=v.shape[0],
                                                      target_sr=resample)
            final_features[n].shape
    return final_features


@gin.configurable
def compute_all(x: np.ndarray, sr: int, descriptors: list[str], z_length: int = None) -> dict:
    """
    Compute all descriptors inside a given dictionnary of function. This
    high-level launch computations and merge dictionnaries.
    Finally aligns the computed features to latent time length by upsampling the array
    using either : 
    - linear interpolation (for continuous) 
    - nearest-neighbors resampling (for discrete)

    Parameters
    ----------
    x : np.ndarray
        Input audio signal (samples)
    sr : int
        Input sample rate


    Returns
    -------
    dict
        Dictionnary containing all features.

    """
    # List of feature sub-libraries to use
    librairies = {"librosa": compute_librosa, "timbral": compute_timbral, "pyloudnorm": compute_pyloudnorm}
    # Final features
    final_features = {}
    for n, func in librairies.items():
        # Process all functions
        cur_dict = func(x, sr, descriptors)
        # Merge dictionnaries
        final_features.update(cur_dict)
    
    # align features to latent time length
    if z_length is not None:
        
        for n, v in final_features.items():
            if len(v.shape) > 1:
                v = v[0]

            if isinstance(v, float):
                final_features[n] = np.full((z_length,), v)
                assert final_features[n].shape[0] == z_length
                continue

            z_indices = np.linspace(0, 1, z_length)
            current_indices = np.linspace(0, 1, v.shape[0])

            if n in CONTINUOUS_RESAMPLING:
                interp_type = 'linear'
            elif n in DISCRETE_RESAMPLING: 
                interp_type = 'nearest'
            else:
                print(f'Warning - interpolation type not specified for descriptor {n}, applying default linear interpolation')
                interp_type = 'linear'

            interp_func = interp1d(current_indices, v, kind=interp_type)

            final_features[n] = interp_func(z_indices)

            # # nearest-neighbors resampling
            # select_indices = np.linspace(0, v.shape[0] - 1, z_length).round().astype(int)
            # final_features[n] = v[select_indices]

            assert final_features[n].shape[0] == z_length
    
    return final_features
