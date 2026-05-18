import taichi as ti
from typing import Iterable
import numpy as np

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
