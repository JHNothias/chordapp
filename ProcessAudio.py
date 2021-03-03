import HStats as hs
import numpy as np
import math as m
from librosa.feature import chroma_stft
from librosa import cqt, perceptual_weighting, cqt_frequencies, note_to_hz
import numba as nb


# gives the audio duration
giveDuration = lambda y, sr: len(y)/sr
# smooths a signal by convolving it.
smoothout = lambda data, smw, mode='valid': np.convolve(data, np.ones(smw)/smw, mode=mode)

getChromagram = lambda y, sr: chroma_stft(y, sr)


getPerceptuallyAdjustedChromagram = lambda y, sr: chroma_stft(S = perceptual_weighting(cqt(y, n_bins=84, sr = sr, fmin = (f_min := note_to_hz('C2'))), cqt_frequencies(84, fmin = f_min), kind="A"), sr = sr)

@nb.jit
def quickprocessChromagram(Chro, treshold=0):
    # print('treshold', treshold)
    C = hs.sortToCif(Chro).T
    if treshold == 0:
        for line in C:
            line = line/max(line)
    else:
        # print(C.shape)
        for line in C:
            line = line/max(line)
            line = (line >= treshold)*line
            # if np.random.choice([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])*np.random.choice([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]):
            #     print(line)
    distributions = np.apply_along_axis(hs.toCircD, 1, C)
    centroids = np.apply_along_axis(hs.resVect, 1, distributions)
    return distributions, centroids

#@nb.njit
def processChromagram(chro, treshold, csmw, mode = 'valid'):
    ChroSmd = np.array([smoothout(chro[i], csmw, mode) for i in range(12)])
    C = hs.sortToCif(ChroSmd).T

    for line in C:
        line = line/max(line)
        line = (line > treshold)*line

    distributions = np.apply_along_axis(hs.toCircD, 1, C)
    centroids = np.apply_along_axis(hs.resVect, 1, distributions)
    return distributions, centroids


# not lambda expressions because they mey benefit from multiprocessing.

def computeHCenters(centroids):
    return np.apply_along_axis(hs.hcenter, 0, centroids).T

def computeVariance(centroids):
    return np.apply_along_axis(hs.variance, 0, centroids).T

def computeH(d, c):
    return np.fromiter((hs.harmoniousness(d[i], c[i]) for i in range(len(c))), float).T

def computeCoh(d, c):
    return np.fromiter((hs.coharmoniousness(d[i], c[i]) for i in range(len(c))), float).T
