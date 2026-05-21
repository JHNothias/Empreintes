from typing import Literal
import numpy as np

def running_total(feature : np.ndarray) -> np.ndarray:
    return np.cumsum(feature)

def novelty_kernel(kernel_radius: int = 3, gaussian_var: float = 1.0, kerneltype: Literal['hh', 'hnh', 'nhh'] = 'hh') -> np.ndarray:
    """
    Build a Gaussian-tapered novelty kernel for SSM-based boundary detection.

    Kernel types (quadrant layout: TL | TR / BL | BR):
        'hh'  : Foote checkerboard — fires at boundaries between two
                homogeneous regions.
                [ +  + | -  - ]
                [ +  + | -  - ]
                [-------------]
                [ -  - | +  + ]
                [ -  - | +  + ]

        'hnh' : fires when leaving a homogeneous region into a heterogeneous
                one. Only TL is positive; everything else is negative.
                [ +  + | -  - ]
                [ +  + | -  - ]
                [-------------]
                [ -  - | +  - ]
                [ -  - | -  + ]

        'nhh' : fires when entering a homogeneous region from a heterogeneous
                one. Only BR is positive; everything else is negative.
                [ +  - | -  - ]
                [ -  + | -  - ]
                [-------------]
                [ -  - | +  + ]
                [ -  - | +  + ]

    Parameters
    ----------
    kernel_size : int
        Side length of the square kernel (enforced even for clean quadrants).
    gaussian_var : float
        Variance σ of the 2-D Gaussian taper.
    kerneltype : {'hh', 'hnh', 'nhh'}

    Returns
    -------
    kernel : np.ndarray, shape (kernel_size, kernel_size)
    """

    # 2-D Gaussian taper, centred across the quadrant boundary
    k = kernel_radius * 2
    ax = np.arange(k) - kernel_radius + 0.5
    xx, yy = np.meshgrid(ax, ax)
    gaussian = np.exp(-(xx**2 + yy**2) / (2 * gaussian_var**2))
    diag_block = np.where(np.eye(kernel_radius, dtype=bool), 1, -1)

    # Build the sign matrix per kernel type
    if kerneltype == 'hh':
        # Standard checkerboard: +1 on TL/BR, -1 on TR/BL
        checker = np.ones((k, k))
        checker[:kernel_radius, kernel_radius:] = -1
        checker[kernel_radius:, :kernel_radius] = -1

    elif kerneltype == 'hnh':
        # All -1 except TL block
        checker = -np.ones((k, k))
        checker[:kernel_radius, :kernel_radius] = 1
        checker[kernel_radius:, kernel_radius:] = diag_block

    elif kerneltype == 'nhh':
        # All -1 except BR block
        checker = -np.ones((k, k))
        checker[:kernel_radius, :kernel_radius] = diag_block
        checker[kernel_radius:, kernel_radius:] = 1
    else:
        raise ValueError(f"Unknown kerneltype '{kerneltype}'. Choose 'hh', 'hnh', or 'nhh'.")

    kernel = gaussian * checker

    # Zero-mean only for 'hh' — the asymmetric kernels are intentionally biased
    if kerneltype == 'hh':
        kernel -= kernel.mean()

    return kernel

def novelty_hh(feature : np.ndarray, kernel_radius:int = 3, gaussian_var: float = 1.0) -> np.ndarray:
    kernel = novelty_kernel(kernel_radius=kernel_radius, gaussian_var=gaussian_var, kerneltype='hh')
    # Apply the symmetric kernel to compute novelty
    novelty = running_total(np.convolve(feature, kernel, mode='valid'))
    return novelty

def novelty_hnh(feature : np.ndarray, kernel_radius:int = 3, gaussian_var: float = 1.0) -> np.ndarray:
    kernel = novelty_kernel(kernel_radius=kernel_radius, gaussian_var=gaussian_var, kerneltype='hnh')
    # Apply the symmetric kernel to compute novelty
    novelty = running_total(np.convolve(feature, kernel, mode='valid'))
    return novelty

def novelty_nhh(feature : np.ndarray, kernel_radius:int = 3, gaussian_var: float = 1.0) -> np.ndarray:
    kernel = novelty_kernel(kernel_radius=kernel_radius, gaussian_var=gaussian_var, kerneltype='nhh')
    # Apply the symmetric kernel to compute novelty
    novelty = running_total(np.convolve(feature, kernel, mode='valid'))
    return novelty

def variance(feature : np.ndarray, kernel_size : int = 200):
    return np.fromiter(map(np.var, [feature[i, np.min(i+kernel_size, len(feature))] for i in range(len(feature))]), dtype=float)

def slice_feature(feature : np.ndarray, slices : list[tuple[int, int]], mode: Literal['avg', 'max', 'min', 'first', 'last'] = 'avg', normalize : bool = True) -> np.ndarray:
    func = {'avg' : np.average,
            'max' : np.max,
            'min' : np.min,
            'first' : lambda arr : arr[0],
            'last' : lambda arr : arr[-1]}[mode]

    hop = slices[-1][1] // feature.shape[0]
    res =  np.fromiter((func(feature[slice[0]//hop : slice[1]//hop]) for slice in slices),
        dtype=float)
    np.nan_to_num(res, copy=False)

    if normalize :
        res = res - np.min(res)
        res = res/np.max(np.abs(res)) if np.any(res != 0) else res
    return res


def average_difference(feature : np.ndarray, slices : list[tuple[int, int]]) -> np.ndarray:
    """
    `feature` : feature values over the duration of the audio.
    `slice` : audio slice [start, end) in samples.
    `hop` : hop size used in computing the feature.
    """
    hop = slices[-1][1] // feature.shape[0]
    res =  np.fromiter((np.average(np.diff(feature[slice[0]//hop : slice[1]//hop])) for slice in slices), dtype=float)
    np.nan_to_num(res, copy=False)
    res = res - np.min(res)
    res = res/np.max(np.abs(res)) if np.any(res != 0) else res
    return res
