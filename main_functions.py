import sys
print(sys.version)
from numbers import Number
import plotly.express as px
import plotly.graph_objects as go
from typing import Any, Callable, Literal
import numpy as np
#import librosa
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
#from pytimbre.audio_files.wavefile import WaveFile
#from pytimbre.spectral.spectra import SpectrumByFFT
#from pytimbre.waveform import Waveform
from flucoma import fluid
#import flucoma
#from pathlib import Path
from IPython.display import Audio, display
#import ot.sliced as ots
from itertools import groupby
from operator import itemgetter
from typing import Iterable
from functools import reduce
import random
from sklearn.manifold import MDS, Isomap, SpectralEmbedding
import taichi as ti
import networkx as nx
#import scipy.optimize as opt
#from scipy.stats import special_ortho_group

def slice_wfm(flucowfm:fluid.FluidSingleOutput, method:Literal['onset', 'even'] = 'onset', sr:int=44100, silence_treatment:Literal['keep', 'glue', 'discard'] = 'discard', silence_threshold=.05, metric=5, slice_threshold=.18, minslicelength=10, slicelength:int = 44100, n_interpolated_slices=1):
  n_interpolated_slices = n_interpolated_slices + 1
  signal = np.asarray(flucowfm)
  peak = np.max(np.abs(signal))
  signal = signal / peak if peak > 0 else signal
  if method == 'onset':
    slicepoints = np.asarray([0] + list(fluid.onsetslice(flucowfm, metric=metric, threshold=slice_threshold, fftsettings=[1024, -1, -1], minslicelength=minslicelength)))
  
  elif method == 'even':
    slicepoints = np.asarray(range(0, signal.shape[1], slicelength), np.int32)
  
  if n_interpolated_slices > 1:
    slicepoints_new = []
    for s, t in zip(slicepoints[:-1], slicepoints[1:]):
      for n in range(n_interpolated_slices):
         slicepoints_new.append(s + n * (t-s)//n_interpolated_slices)
    slicepoints = slicepoints_new
  
  deltas = np.fromiter((slicepoints[i+1] - slicepoints[i] if i+1 < len(slicepoints) else ((signal.shape[1])) - slicepoints[i]-1 for i in range(len(slicepoints))), dtype=int)
  
  init_slices = [(int(slicepoints[i]), int(slicepoints[i] + sum(deltas[i:min(i+n_interpolated_slices, len(deltas))]))) for i in range(len(slicepoints))] # skips over some slicepoints based on n_intercal

  silenceidxs = [i for i in range(len(init_slices)) if np.max(signal[:, init_slices[i][0] : init_slices[i][1]]) <= silence_threshold]
  
  match silence_treatment:
    case 'keep' :
        return np.asarray(init_slices), np.asarray(silenceidxs)
    case 'glue' :
      groups = []
      for _, g in groupby(enumerate(silenceidxs), lambda x: x[0] - x[1]):
          group = list(map(itemgetter(1), g))
          groups.append(group)
 
      glued = []
      used = set(idx for group in groups for idx in group)
      silenceidxsbis = []
      i = 0
      while i < len(init_slices):
          if i not in used:
              glued.append(init_slices[i])
              i += 1
          else:
              # Find the group that starts at i
              for group in groups:
                  if group[0] == i:
                      glued.append((init_slices[group[0]][0], init_slices[group[-1]][1]))
                      silenceidxsbis.append(len(glued)-1)
                      i = group[-1] + 1
                      break
      print(f'glued {len(silenceidxs)} silences for a total of {sum([(init_slices[i][1] - init_slices[i][0]) for i in silenceidxs])/sr:.2f}s')

      
      return np.asarray(glued), np.asarray(silenceidxsbis)
    case 'discard':
      res = [s for i, s in enumerate(init_slices) if i not in silenceidxs]
      print(f'removed {len(silenceidxs)} silences for a total of {sum([(init_slices[i][1] - init_slices[i][0]) for i in silenceidxs])/sr:.2f}s')
      return np.asarray(res), np.asarray(silenceidxs)

def showslice(aslice, flucowfm=None, spect=None, freq = None, hop_length=512, sr=44100, trace_overlays : (None | Iterable[tuple[np.ndarray, np.ndarray, Any]])=None, mode:Literal['plotly', 'pyplot']='plotly',):
  if spect is not None:
    if mode == 'plotly':
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Heatmap(
            z = spect[:,aslice[0]//hop_length: aslice[1]//hop_length],
            x = np.linspace(aslice[0]/sr, aslice[1]/sr, aslice[1]//hop_length - aslice[0]//hop_length),
            colorscale='plasma'))
        if trace_overlays is not None:
          for t in trace_overlays:
            x = t[1] - np.min(t[1])
            x = x / np.max(x)
            x = x * (aslice[1]/sr - aslice[0]/sr)
            x = x + aslice[0]/sr
            y = t[0]
            fig.add_trace(go.Scatter(
              mode='lines',
              x=x,
              y=y,
              line=dict(color=t[2],width=2)), secondary_y=True)
        fig.show()
    if mode == 'pyplot':
        plt.imshow(spect[:,aslice[0]//hop_length: aslice[1]//hop_length], origin='lower', aspect=1/2.41)
        plt.xticks(np.linspace(0, aslice[1]//hop_length - aslice[0]//hop_length, 1+2*(aslice[1]-aslice[0])//sr), labels = list(map(lambda n: f"{n:.1f}s", list(np.linspace(aslice[0]/sr, aslice[1]/sr, 1+2*(aslice[1] - aslice[0])//sr)))))
        plt.show()
    #px.imshow(spect[:,aslice[0]//hop_length: aslice[1]//hop_length], origin='lower', aspect=(1, 1.8)).show()
  if flucowfm is not None:
    display(Audio(np.asarray(flucowfm)[:,aslice[0]: aslice[1]], rate=sr))

def showslice_descriptors_overlay(lo, up, slices, flucowfm=None, spect=None, hop_length=512, sr=44100, n_intercal=1, descs:(None | Iterable[np.ndarray])=None, colors : (None | Iterable[Any])=None):
  if descs is not None:
    desc_slice = (min(lo, len(slices)-1), min(up + n_intercal + 1, len(slices)-1))
    desc_x = np.arange(min(lo, len(slices)-1), min(up + n_intercal + 1, len(slices)-1))
    if colors is None:
      colors = ["#%06x" % random.randint(0, 0xFFFFFF) for i in range(len(descs))] # type: ignore
    trace_overlays= [(d[desc_slice[0] : desc_slice[1]], desc_x, c) for d, c in zip(descs, colors)]
  else:
    trace_overlays = None
  showslice((slices[lo][0], slices[up][1]), flucowfm=flucowfm, spect=spect, hop_length=hop_length, sr=sr, trace_overlays=trace_overlays)
  
@ti.func
def _ti_plomp(f1, f2):
    fmin = min(f1, f2)
    fmax = max(f1, f2)
    s = 0.24 / (0.021 * fmin + 19.0)
    return ti.exp(-3.5 * s * (fmax - fmin)) - ti.exp(-5.75 * s * (fmax - fmin))

@ti.kernel
def _ti_roughness(spectra : ti.types.ndarray(), frequencies : ti.types.ndarray(), res : ti.types.ndarray()): # type: ignore
    N = frequencies.shape[0]
    for x, y in ti.ndrange(N, N):
        if x <= y:
            f1 = frequencies[x]
            f2 = frequencies[y]
            for i in res:
                a1 = spectra[i, x]
                a2 = spectra[i, y]
                p = _ti_plomp(f1, f2)
                res[i] += a1 * a2 * p

def roughness_self(spectra: (Iterable[np.ndarray] | np.ndarray), frequencies:np.ndarray):
    if isinstance(spectra, Iterable):
        spectra = np.stack(spectra) # type: ignore
    res = ti.ndarray(ti.f32, len(spectra)) # type: ignore
    _ti_roughness(np.asarray(spectra, np.float32), np.asarray(frequencies, np.float32), res)
    return res.to_numpy()

@ti.kernel
def _ti_spectrogram_concordance_matrix(spectrogram_T : ti.types.ndarray(), segments : ti.types.ndarray(), res:ti.types.ndarray()): # type: ignore
    S = spectrogram_T.shape[1]
    for i, j in res:
        if i <= j:
            si, ei = segments[i, 0], segments[i, 1]
            sj, ej = segments[j, 0], segments[j, 1]
            minlen = min(ei-si, ej-sj)
            for x, y in ti.ndrange(minlen, S):
                res[i, j] += (spectrogram_T[x+si, y] * spectrogram_T[x+sj, y])
            asum = 0.
            bsum = 0.
            for x, y in ti.ndrange(minlen, S):
                asum += spectrogram_T[x+si, y]**2
                bsum += spectrogram_T[x+sj, y]**2
            res[i, j] = 1 - res[i, j]/(ti.sqrt(asum) * ti.sqrt(bsum)) 
            res[j, i] = res[i, j]

def concordance_matrix_spectrogram(spectrogram:np.ndarray, slices:(list[tuple[int, int]] | np.ndarray), hop_length=512):
    spectrogram = spectrogram.T
    slices = np.asarray(slices, np.uint32)//hop_length
    ti_spectrogram = ti.ndarray(dtype=ti.f32, shape=spectrogram.shape)
    ti_segments = ti.ndarray(dtype=ti.u32, shape=slices.shape)
    res = ti.ndarray(dtype=ti.f32, shape=(len(slices), len(slices)))

    ti_spectrogram.from_numpy(spectrogram)
    ti_segments.from_numpy(slices)
    res.fill(0.)
    _ti_spectrogram_concordance_matrix(spectrogram, slices, res)
    return res.to_numpy()

def symmetrize(a, mode : Literal['u', 'l'] = 'u'):
    if mode == 'u':
        a = np.triu(a) 
    if mode == 'l':
        a = np.tril(a) 
    return a + a.T - np.diag(a.diagonal())

def wasserstein_matrix(spectrogram:np.ndarray, slices:(list[tuple[int, int]] | np.ndarray), hop_length=512, p=2):
    slices = np.asarray(slices)//hop_length
    spectrogram = np.asarray(spectrogram.T)
    dm = np.zeros((len(slices), len(slices)))
    for i in range(len(slices)):
        if i == len(slices) or i%(len(slices)//20) == 0:
            print(f"{100*i/len(slices):.0f}%")
        for j in range(len(slices)):
            if i <= j:
                si, ei = slices[i, 0], slices[i, 1]
                sj, ej = slices[j, 0], slices[j, 1]
                mindist = min((ei-si), (ej-sj))
                dm[i, j] = ots.sliced_wasserstein_distance(X_s=spectrogram[si : si + mindist], X_t=spectrogram[sj : sj + mindist], p=p)/(mindist * spectrogram.shape[1])
    return symmetrize(dm)

def simple_preprocess(anarray : np.ndarray, rescale = True, topositive = True, useabs = True):
    res = anarray - np.min(anarray.flatten()) if topositive else anarray
    if useabs:
        res = res / np.nanmax(np.abs(res.flatten())) if rescale else res
    else:
        res = res / np.nanmax(res.flatten()) if rescale else res
    return res

def normalize_varmean(anarray):
    m = np.mean(anarray)
    s = np.std(anarray)
    return (anarray - m) / s

def n_least(a:np.ndarray, b:np.ndarray, n:int) -> np.ndarray:
    """ `a` is some array, `b` is a boolean array of the same shape.
        For each row, mark as True a cell if it is among the (up to)
        n cells that have the least values in `a`, among those that are marked as True in `b`."""
    assert a.shape == b.shape, "a and b must have the same shape."
    
    indices = np.arange(a.shape[1])
    res = np.zeros(a.shape, dtype = bool)
    @np.vectorize
    def do_the_job(rowi:int):
        sorted_idxs = np.argsort(a[rowi])
        sorted_bool = b[rowi][sorted_idxs]
        filtrd_idxs = sorted_idxs[sorted_bool]
        idxs_nbest = filtrd_idxs[:n]
        res[rowi][idxs_nbest] = True
    do_the_job(indices)
    return res

def n_neighborhoods(dm, n_neighbors=5):
    """
    Given a distance matrix dm, it returns a boolean matrix with the same dimensions such that
    each row i highlights a ball of radius r around i with:
    $$r = \\mathrm{sup} \\{r \\in ℝ | Card(\\{j | dm[i,j] < r\\}) < n_neighbors\\}$$
    This provides a way of generating balls in a discrete metric space that have a limited cardinality.
    """
    balls = np.zeros(dm.shape, dtype=bool)
    for i in range(len(dm)):
        sorted_distances = np.sort(dm[i])
        r = sorted_distances[n_neighbors - 1]
        balls[i] = dm[i] <= r
    return balls

def local_minima(dm: np.ndarray, n_neighbors=5):
    """
    Given a distance matrix dm, it returns a boolean matrix with the same dimensions such that
    each row i highlights the points j such that given a small radius r:
    ∀k ∈ 𝔹_r(j), dm[i, j] <= dm[i, k].
    The radius r is determined using the n_neighborhoods function.
    """
    N = len(dm)
    balls = n_neighborhoods(dm, n_neighbors=n_neighbors)
    local_minimas = np.zeros((N, N), dtype=bool)

    for i in range(N):  # Iterate over each point as the reference point
        for j in range(N):  # Body checks if point j is a local minimum relative to i
            is_min = True
            for k in np.arange(N)[balls[j]]:  # Iterate over all points in the neighborhood of j
                if dm[i, j] > dm[i, k]:  # If any point r has a shorter distance to i than j does
                    is_min = False  # j is not a local minimum
                    break  # Exit the loop early
            local_minimas[i, j] = is_min  # Record the result

    return local_minimas

def crossingGraph(dm:np.ndarray, n_neighbors:int = 3, n_connected_neighbors:int = 3, n_best_crossings:int = 3, homogenize:bool = True):
    """
    Returns a crossing graph G over the metric given by dm. A crossing graph oover a metric space is a graph highlighting the most
    prominent features of the metric space, known as crossings or local minima with respect some point, while also preserving
    locality. Concretely, given two radii r and s, every point i will be linked to all points j suh that:
        - dm[i, j] < s, or
        - ∀k ∈ 𝔹_r(j), dm[i, j] <= dm[i, k].
    The radii r and s are determined using the n_neighborhoods function:
        - n_neighbors is used in determinig r
        - n_connected_neighbors is used in determinig s.
    The argument n_best_crossings serves to limit the number of crossings present in the graph. I.e, if n_best_crossings == 3,
    given any point i, then among all points j such that ∀k ∈ 𝔹_r(j), dm[i, j] <= dm[i, k], only the 3 points j such that
    dm[i, j] is smalles will be kept.

    The argument `homogenize` toggles a post-processing step on the graph G making it such that:
    for any distance d, if there exists points i, j such that (i, j) ∈ Edges(G) and dm[i, j] == d, then for all points k, l
    such that dm[k, l] == d, we have (k, l) ∈ Edges(G).
    
    This graph then serves to compute a new distance matrix that will keep these most prominent features while discarding
    the features that are less important to the overall shape of the metric space. This method gives excellent results,
    has a fast execution time compared to other methods, lessens the need for "magic values" that may vary with the metric space
    (for instance radii that have to be guessed), while also guaranteeing a fairly sparse graph.
    """
    conballs = n_neighborhoods(dm, n_neighbors=n_connected_neighbors)
    crossings = local_minima(dm, n_neighbors=n_neighbors)
    nleast = n_least(dm, crossings, n_best_crossings)
    res = np.logical_or(conballs, nleast)
    adj = np.logical_or(res, res.T)
    if homogenize:
        weights = set(list((dm*adj).flatten()))
        adj = np.logical_or(reduce(np.logical_or, (dm == w for w in weights if w > 0), np.zeros(shape=dm.shape)), adj)

    G = nx.Graph()
    G.add_nodes_from(range(len(dm)))
    grid = np.stack(np.meshgrid(np.arange(dm.shape[0]), np.arange(dm.shape[1])), axis=2)
    for i, j in grid[adj]:
        G.add_edge(i, j, weight=dm[i, j])
    return G

def localGraph(G:nx.Graph, v:int, iterations:int = 2, _ResGraph:(nx.Graph | None) = None):
    if _ResGraph is None:
        ResGraph = nx.Graph()
    else:
        ResGraph = _ResGraph
    if iterations == 0:
        return _ResGraph
        
    for i in G.neighbors(v):
        if i != v:
            ResGraph.add_edge(v,i)
        localGraph(G, i, iterations = iterations-1, _ResGraph=ResGraph)
    return ResGraph

def pathDistances(G: nx.Graph) -> np.ndarray:
    """
    Computes a distance matrix from a weighted graph representing the minimal path length between any two points on the graph.
    """
    N = len(G.nodes())
    ddict = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
    distance_matrix = np.full((N, N), np.inf)
    indices = np.indices((N, N))
    @np.vectorize
    def convert(i, j):
        try:
            distance_matrix[i, j] = ddict[i][j]
        except:
            pass
    convert(indices[0], indices[1])
    return distance_matrix

def xyzMDS(dm:np.ndarray, dimension=3) -> np.ndarray:
    """
    Embeds a metric space given its distance matrix `dm` into `dimensions` number of dimensions.
    returns an iterable of coordinates.
    """
    mds = MDS(n_components=dimension, n_jobs=-1, dissimilarity='precomputed',)
    return mds.fit_transform(dm)

def xyzIsomap(dm:np.ndarray, dimension=3) -> np.ndarray:
    iso = Isomap(n_components=dimension, n_jobs=-1, n_neighbors=80)
    return iso.fit_transform(dm)

def xyzSpectral(dm:np.ndarray, dimension=3) -> np.ndarray:
    spectral = SpectralEmbedding(n_components=dimension, affinity='precomputed')
    return spectral.fit_transform(1/(dm+1e-6))

def feature_to_color(feature, slices):
    """ 1D feature arrays only !!"""
    hop = slices[-1][1] // feature.shape[0]
    res = np.asarray([np.average(feature[s[0]//hop : s[1]//hop]) for s in slices], dtype=np.float32)
    var = np.asarray([np.var(feature[s[0]//hop : s[1]//hop]) for s in slices], dtype=np.float32)
    dif = np.asarray([np.average(np.diff(feature[s[0]//hop : s[1]//hop])) for s in slices], dtype=np.float32)
    np.nan_to_num(res, copy=False)
    np.nan_to_num(var, copy=False)
    np.nan_to_num(dif, copy=False)
    res = res - np.min(res)
    var = var - np.min(var)
    dif = dif - np.min(dif)
    res = res/np.max(np.abs(res)) if np.any(res != 0) else res
    var = var/np.max(np.abs(var)) if np.any(var != 0) else var
    dif = dif/np.max(np.abs(dif)) if np.any(dif != 0) else dif
    return res, var, dif

def compute_features(flucowfm, slices, hop_length, stft_spect, stft_freq, mergeLR = True):
    features = {
        'shape' : lambda i : fluid.spectralshape(i, fftsettings=[1024, hop_length, -1]),
        'loudness' : lambda i : fluid.loudness(i, hopsize=hop_length),    
        'novelty' : lambda i : fluid.noveltyfeature(i, fftsettings=[1024, hop_length, -1]),
        'pitch' : lambda i : fluid.pitch(i, fftsettings=[1024, hop_length, -1]),
        'roughness' : lambda _: np.stack([roughness_self(stft_spect[0].T, frequencies=stft_freq), roughness_self(stft_spect[1].T, frequencies=stft_freq)], axis=0),
        'time' : lambda _ : np.arange(len(slices)) / len(slices)

    }

    feature_names = {
        'shape' : ['centroid', 'spread', 'skew', 'kurtosis', 'rolloff', 'flatness', 'crest'],
        'pitch' : ['pitch measured', 'pitch confidence'],
        'novelty' : ['novelty'],
        'roughness' : ['roughness'],
        'loudness' : ['loudness', 'true peak'],
        'time' : ['time']
    }
    
    names = ['centroid', 'spread', 'skew', 'kurtosis', 'rolloff', 'flatness', 'crest', 'pitch measured', 'pitch confidence', 'novelty', 'roughness', 'loudness', 'true peak', 'time']
    
    features_memo = dict()
    for k in features:
        print(k)
        f = simple_preprocess(np.asarray(features[k](flucowfm)))
        print(f.shape)
        N = len(feature_names[k])
        if len(f.shape) > 1:
            for i in range(f.shape[0]):
                if len(f.shape) != 2:
                    f = f.reshape((1, -1))
                LoR = '-l' if i//N == 0 else '-r'
                name = feature_names[k][i%N]+LoR
                features_memo[name] = f[i, :].flatten()
        else:
            name = feature_names[k][0]
            features_memo[name] = f
    if mergeLR:
        for name in names:
            if name + '-l' in features_memo:    
                features_memo[name] = (features_memo[name + '-l'] + features_memo[name + '-r'])/2
                del features_memo[name + '-l']
                del features_memo[name + '-r']
    return features_memo

def slice_features(features_dict, slices, squash = False, squashfactor = 10):
    if squash:
        def squashf(data:np.ndarray): # type: ignore
            data = data - np.min(data)
            data = data / np.max(data) if np.max(data) > 0 else data
            data = 2 * (data - np.average(data))
            data = np.arcsinh(squashfactor * data) / np.arcsinh(squashfactor)
            return (1 + data) / 2
    else:
        def squashf(x):  return x
    
    colors_memo = dict()
    for name in features_dict:
        if name != 'time':
            res, var, dif = feature_to_color(features_dict[name], slices) # type: ignore
            res = simple_preprocess(squashf(res))
            var = simple_preprocess(squashf(var))
            dif = simple_preprocess(squashf(dif))
            colors_memo[name] = res
            colors_memo[name + '-var'] = var
            colors_memo[name + '-dif'] = dif
        else :
            colors_memo[name] = features_dict[name]
    return colors_memo

@ti.kernel
def _ti_normal_distribs(dm:ti.types.ndarray(), dvars:ti.types.ndarray(), res:ti.types.ndarray(), falloff:ti.types.float64): # type: ignore
    N = dm.shape[0]
    for i in dvars:
        dvar = 0.0
        for j in range(N):
            dvar += dm[i, j]**2
        dvars[i] = dvar/N
    for i, j in res:
        res[i, j] = 2 * falloff * ti.exp(- dm[i, j]**2 / (2*dvars[i]*(1/falloff)**2))/(dvars[i]*ti.sqrt(2*ti.math.pi))
    for i in range(res.shape[0]):
        cum = 0.
        for j in range(res.shape[1]):
            cum += res[i, j]
        for j in range(res.shape[1]):
            res[i, j] /= cum

def normal_distribs(dm:np.ndarray, falloff:Number = 2): # type: ignore
    res = ti.ndarray(ti.float64, dm.shape)
    dvars = ti.ndarray(ti.float64, dm.shape[0])
    _ti_normal_distribs(dm, dvars, res, falloff)
    return res.to_numpy()

def gaussian_smoothness(dm, f, d, params, falloff):
    """
    dm : distance matrix
    f : int (index) x array (parameters) -> T
    d : T x T -> 0 <= float <= 1
    params : array[array]
    """
    distrib = normal_distribs(dm, falloff=falloff)
    res = ti.ndarray(ti.f64, len(params))
    g = ti.func(f)
    e = ti.func(d)
    @ti.kernel
    def _ti_gaussian_smoothness(distrib:ti.types.ndarray(), params:ti.types.ndarray(), res:ti.types.ndarray()): # type: ignore
        N = params.shape[0]
        M = distrib.shape[0]
        for p in range(N):
            a = 0.
            for i in range(M):
                b = 0.
                for j in range(M):
                    b += distrib[i, j] * e(g(i, params[p]), g(j, params[p]))
                a += (1/M) * b
            res[p] = 1 - a
            
    _ti_gaussian_smoothness(distrib, params, res)
    return res.to_numpy()

def invert_distribution(distrib:np.ndarray):
    invdistrib = np.asarray([np.max(distrib[i, :]) - distrib[i, :] for i in range(len(distrib))])
    invdistrib = np.asarray([invdistrib[i, :]/np.sum(invdistrib[i, :]) for i in range(len(distrib))])
    return invdistrib

def gaussian_localization(dm, f, d, params:(list[Number] | None) = None, falloffs : (Number | list[Number]) = 5, distribs : (None | np.ndarray | list[np.ndarray]) = None): # type: ignore
    """
    dm : distance matrix
    f : int (index) x array (parameters) -> T
    d : T x T -> 0 <= float <= 1
    params : array[array]
    """

    if isinstance(falloffs, Number):
        falloffs = [falloffs]
    if params is None:
        params = np.asarray([0]) # type: ignore
    elif isinstance(params, list):
        params = np.asarray(params) # type: ignore

    
    res = ti.ndarray(ti.f64, len(params)) # type: ignore
    g = ti.func(f)
    e = ti.func(d)

    @ti.kernel
    def _ti_gaussian_localization(distrib:ti.types.ndarray(), invdistrib:ti.types.ndarray(), params:ti.types.ndarray(), res:ti.types.ndarray()): # type: ignore
        N = params.shape[0]
        M = distrib.shape[0]
        for p in range(N):
            a = 0.
            for i in range(M):
                b = 0.
                c = 0.
                for j in range(M):
                    d = e(g(i, params[p]), g(j, params[p]))
                    b += distrib[i, j] * d
                    c += invdistrib[i, j] * d
                a += (1/M) * ti.abs(b-c)
            res[p] = a
    
    if distribs is None :
        distribs = [normal_distribs(dm, falloff=falloff) for falloff in falloffs]
    elif isinstance(distribs, np.ndarray):
        distribs = [distribs]
    invdistribs = [invert_distribution(distrib) for distrib in distribs]

    scores = []
    for distrib, invdistrib in zip(distribs, invdistribs):
        _ti_gaussian_localization(distrib, invdistrib, params, res)
        scores.append(res.to_numpy())

    return np.max(np.vstack(scores), axis=0)



def get_localization_scores(descriptors : dict, dm, distribs : (None | np.ndarray | list[np.ndarray]) = None, falloffs:(Number | list[Number])=[1, 5]): # type: ignore
    if isinstance(falloffs, Number):
        falloffs = [falloffs]

    if distribs is None:
        distribs = [normal_distribs(dm, falloff=falloff) for falloff in falloffs]
    if isinstance(distribs, np.ndarray):
        distribs = [distribs]
    res = dict()
    def d(a, b):
        return ti.abs(a-b)
    for k in descriptors:
        field = ti.field(ti.f32, (descriptors[k].shape[0], 1))
        field.from_numpy(normalize_varmean(descriptors[k]).reshape((descriptors[k].shape[0], 1)))
        def f(i, p):
            return field[i, p]
        res[k] = gaussian_localization(dm, f, d, falloffs=falloffs, distribs=distribs)[0]
    return res

def desc_from_combination(descriptors_dict, desc:np.ndarray):
    colors_vector = np.vstack([descriptors_dict[k] for k in descriptors_dict]).T
    newdesc = np.sum(colors_vector * np.repeat(desc.reshape(1, -1), colors_vector.shape[0], axis=0), axis=1)
    newdesc = newdesc - np.min(newdesc)
    newdesc = newdesc / np.max(newdesc)
    return newdesc

def randomdescs(descriptors_dict:dict, dm:np.ndarray, tries:int=20, distance_function : (Callable | None) = None):
    combs = 2*np.random.random((tries, len(descriptors_dict)))-1
    descs = {i : desc_from_combination(descriptors_dict=descriptors_dict, desc=combs[i, :]) for i in range(len(combs))}
    scores = get_localization_scores(descs, dm, falloffs = 5)
    scores_sorted = sorted(scores.items(), key=lambda item: item[1])
    return scores_sorted, combs

def optimize_descs(descriptors_dict:dict, dm:np.ndarray, n_descriptors:int = 5, distance_function : (Callable | None) = None, tol = .1, heat=.5, exponent=5):
    assert n_descriptors <= len(list(descriptors_dict))
    if distance_function is None :
        def d(a, b):
            return ti.abs(a-b)
    else:
        d = distance_function

    colors_vector = np.vstack([descriptors_dict[k] for k in descriptors_dict]).T
    print("setting initial conditions...")
    x0 = 2*np.random.random(colors_vector.shape[1]) - 1
    scorestmp = get_localization_scores(descriptors_dict, dm=dm)
    scores = np.asarray([scorestmp[k] for k in scorestmp])
    scores = scores / np.max(scores)
    x0 = ((1-heat) + heat * x0) * scores ** exponent + heat * x0
    vector = x0[np.newaxis,:]
    orthogonal_vertices = np.linalg.svd(vector)[-1]
    rot = special_ortho_group.rvs(colors_vector.shape[1])
    x0s = (rot @ orthogonal_vertices)[:n_descriptors, :]
    x0s = x0s.reshape(x0s.shape[0]*x0s.shape[1])

    weights_shape = (n_descriptors, colors_vector.shape[1])

    def f(weights:np.ndarray):
        weights = weights / np.linalg.norm(weights)
        weights = weights.reshape((1, -1))
        #print(weights.shape, colors_vector.shape, np.repeat(weights, colors_vector.shape[0], axis=0).shape)
        newdesc = np.sum(colors_vector * np.repeat(weights, colors_vector.shape[0], axis=0), axis=1)
        newdesc = newdesc - np.min(newdesc)
        newdesc = newdesc / np.max(newdesc) 
        #print(newdesc.shape)
        _ti_newdesc = ti.field(ti.f32, newdesc.shape)
        _ti_newdesc.from_numpy(newdesc)
        def g(i, p):
            return _ti_newdesc[i]
        return -gaussian_localization(dm, g, d, np.asarray([0]), 5)[0]
    
    print("optimisation started. Be ready to wait.")
    weights_res = []
    for i in range(n_descriptors):
        print(f"optimizing descriptor {i}...")
        optres = opt.minimize(f, x0=np.reshape(x0s, weights_shape)[i], tol = tol, bounds=[(-1, 1)]*colors_vector.shape[1])
        weights_res.append(optres.x/np.linalg.norm(optres.x))
    return weights_res

def network3d(G:nx.Graph, xyz:np.ndarray,
            highlight_nodes:(Iterable[int] | None) = None, highlight_edges:(Iterable[tuple[int, int]] | None) = None,
            nodecolors:(Callable | Iterable | None) = None, edgecolors:(Callable | Iterable | None) = None,
            nodetext:(Callable | Iterable | None) = None, edgetext:(Callable | Iterable | None) = None,
            showEdges = True, showNodes = True, showHNodes = True, showHEdges = True, nodecolorscale = "rainbow", edgecolorscale = "rainbow", index_conversion=lambda i:i, nodesize = 6) -> go.FigureWidget:
    """G nodes must be integers."""
        
    if isinstance(nodecolors, Iterable):
        ncf = lambda i: nodecolors[i]
    else :
        ncf = nodecolors
    if isinstance(nodetext, Iterable):
        ntf = lambda i: nodetext[i]
    else :
        ntf = nodetext
    if isinstance(edgetext, Iterable):
        etf = lambda e: edgetext[e]
    else :
        etf = edgetext
    if isinstance(edgecolors, Iterable):
        ecf = lambda e: edgecolors[e]
    else :
        ecf = edgecolors
        
    if nodecolors is None:
        ncf = lambda i: i
    if nodetext is None:
        ntf = lambda i: f"{i}"
    if edgetext is None:
        etf = lambda e: ""
    if edgecolors is None:
        ecf = lambda e: "grey"

    def edgeconvert(e):
        return (index_conversion(e[0]), index_conversion(e[1]))
        
    scatter = go.Scatter3d(
    x = xyz.T[0],
    y = xyz.T[1],
    z = xyz.T[2],
    mode='markers',
    text = [ntf(i) for i in G.nodes],
    marker=dict(size=nodesize,
                color=[ncf(i) for i in G.nodes],
                colorscale=nodecolorscale,
                colorbar=dict()),
    )

    if highlight_nodes is not None:
        hscatter = go.Scatter3d(
        x = xyz.T[0][list(map(index_conversion, highlight_nodes))],
        y = xyz.T[1][list(map(index_conversion, highlight_nodes))],
        z = xyz.T[2][list(map(index_conversion, highlight_nodes))],
        mode='markers',
        text = [ntf(i) for i in highlight_nodes],
        marker=dict(size=nodesize,
                    color=[ncf(i) for i in highlight_nodes],
                    colorscale=nodecolorscale,
                    colorbar=dict()),
        )


    edge_x = []
    edge_y = []
    edge_z = []
    edge_colors = []
    for e in G.edges():
        x0, y0, z0 = xyz[edgeconvert(e)[0]]
        x1, y1, z1 = xyz[edgeconvert(e)[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])
        edge_colors.extend([ncf(e[0]), ncf(e[1]), ecf(e)])

    edges = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode='lines',
        text = [etf(edgeconvert(e)) for e in G.edges()],
        line=dict(color = edge_colors, width=2, colorscale = edgecolorscale)
    )

    if highlight_edges is not None:
        edge_x = []
        edge_y = []
        edge_z = []
        for e in highlight_edges:
            x0, y0, z0 = xyz[edgeconvert(e)[0]]
            x1, y1, z1 = xyz[edgeconvert(e)[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])

        hedges = go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode='lines',
            text = [etf(edgeconvert(e)) for e in highlight_edges],
            line=dict(color=[ecf(edgeconvert(e)) for e in highlight_edges], width=2, colorscale = edgecolorscale)
        )

    data = []
    if showNodes:
            data.append(scatter)
    if showEdges:
            data.append(edges)
    if showHEdges and highlight_edges is not None:
        data.append(hedges)
    if showHNodes and highlight_nodes is not None:
            data.append(hscatter)
    
    layout = go.Layout(scene=dict(aspectmode="data"))
    fig = go.FigureWidget(data=data, layout=layout)
    return fig

