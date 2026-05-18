import numpy as np

def slice_regular(wfm : np.ndarray, slicelength_s : float, sr : int, n_interpolated_slices):
    slicelength = slicelength_s * sr
    slicepoints = np.asarray(range(0, wfm.shape[1], int(slicelength)))

    if n_interpolated_slices > 1:
        slicepoints_new = []
        for s, t in zip(slicepoints[:-1], slicepoints[1:]):
            for n in range(n_interpolated_slices):
                slicepoints_new.append(s + n * (t-s)//n_interpolated_slices)
        slicepoints = slicepoints_new

    deltas = np.fromiter((slicepoints[i+1] - slicepoints[i] for i in range(len(slicepoints))), dtype=int)
    return np.asarray([(int(slicepoints[i]), int(slicepoints[i] + sum(deltas[i:min(i+len(slicepoints)-i, len(deltas))]))) for i in range(len(slicepoints))])
