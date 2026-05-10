from LazyTree import LazyTree
from main_functions import *
import librosa

def interface():
    """Returns an object that orchestrates the whole pipeline."""
    interface = LazyTree({
        "init_settings" : {
            "title" : lambda _, i : None,
            "filepath" : lambda _, i : None,
            "hop_length": lambda _, i: 256,
            "silence_treatment": lambda _, i: 'keep',
            "silence_threshold": lambda _, i: 0.05,
            "slice_threshold": lambda _, i: 0.18,
            "n_mels": lambda _, i: 512,
            "sr": lambda _, i: 44100,
            "slicelength_seconds": lambda _, i: 2,
            "n_interpolated_slices": lambda _, i: 1,
            "method" : lambda _, i: "even",
            "slice_distance" : lambda _, i : "concordance",
            "crossing_parameters" : lambda _, i : (3, 3, 3),
            "squash" : lambda _, i : True,
            "save_filepath" : lambda _, i : ''
        },
        "waveform" : lambda _, i: fluid.FluidSingleOutput(interface.localget(i, "init_settings-filepath")),
        "slices" : lambda _, i: interface.localget(i, "_slices_output")[0],
        "silences" : lambda _, i: interface.localget(i, "_slices_output")[1],
        "_slices_output" : lambda _, i : slice_wfm(interface.localget(i, "waveform"),
                                     method=interface.localget(i, "init_settings-method"),
                                     slice_threshold=interface.localget(i, "init_settings-slice_threshold"),
                                     minslicelength=50,
                                     silence_treatment=interface.localget(i, "init_settings-silence_treatment"),
                                     silence_threshold=interface.localget(i, "init_settings-silence_threshold"),
                                     n_interpolated_slices=interface.localget(i, "init_settings-n_interpolated_slices"),
                                     slicelength=interface.localget(i, "init_settings-slicelength_seconds")*interface.localget(i, "init_settings-sr")),
        "spectrograms" : {
            "stft" : lambda _, i : np.abs(librosa.stft(y=np.asarray(interface.localget(i, "-waveform")), hop_length=interface.localget(i, "-init_settings-hop_length"), n_fft=4092)),
            "mel" : lambda _, i : np.abs(librosa.feature.melspectrogram(y=np.asarray(interface.localget(i, "-waveform")),
                                      hop_length=interface.localget(i, "-init_settings-hop_length"),
                                      sr=interface.localget(i, "-init_settings-sr"), power=1,
                                      n_mels = interface.localget(i, "-init_settings-n_mels"), fmin=1, n_fft=4092)),
            "stft_freq" : lambda _, i : librosa.fft_frequencies(n_fft=4096, sr=interface.localget(i, "-init_settings-sr")),
            "mel_freq" : lambda _, i : librosa.mel_frequencies(n_mels=interface.localget(i, "-init_settings-n_mels"), fmin=1, fmax = interface.localget(i, "-init_settings-sr")/2)
        },
        "dm" : {
            "dm" : lambda _, i : simple_preprocess(interface.localget(i, interface.localget(i, "-init_settings-slice_distance"))),
            "concordance" : lambda _, i : concordance_matrix_spectrogram(spectrogram=np.log(1+(interface.localget(i, "-spectrograms-mel")[0] + interface.localget(i, "-spectrograms-mel")[1])/2), slices=interface.localget(i, "-slices"), hop_length=interface.localget(i, "-init_settings-hop_length")),
            "concordance_harmonic" : lambda _, i : concordance_matrix_spectrogram(spectrogram=librosa.decompose.hpss(np.log(1+(interface.localget(i, "-spectrograms-mel")[0] + interface.localget(i, "-spectrograms-mel")[1])/2), kernel_size=15, margin=1)[0], slices=interface.localget(i, "-slices"), hop_length=interface.localget(i, "-init_settings-hop_length")),
            "wasserstein" : lambda _, i : wasserstein_matrix(np.log(1+(interface.localget(i, "-spectrograms-mel")[0] + interface.localget(i, "-spectrograms-mel")[1])/2),
                                                                 interface.localget(i, "-slices"), hop_length=interface.localget(i, "-init_settings-hop_length"), p=2),
            "path" : lambda _, i : pathDistances(interface.localget(i, "-graph"))
        },
        "graph" : lambda _, i : crossingGraph(interface.localget(i, "-dm-dm"), *interface.localget(i, "-init_settings-crossing_parameters")),
        "embedding" : {
            "dm" : lambda _, i : xyzMDS(interface.localget(i, "-dm-dm")),
            "path": lambda _, i : xyzMDS(interface.localget(i, "-dm-path")),
        },
        "features" : {
            "raw" : lambda _, i : compute_features(interface.localget(i, "-waveform"),
                                                 interface.localget(i, "-slices"),
                                                 interface.localget(i, "-init_settings-hop_length"),
                                                 interface.localget(i, "-spectrograms-stft"),
                                                 interface.localget(i, "-spectrograms-stft_freq"), mergeLR = True),
            "sliced" : lambda _, i : slice_features(interface.localget(i, "raw"), interface.localget(i, "-slices"),
                                                 squash = interface.localget(i, "-init_settings-squash"), squashfactor = 10),
            "scores" : lambda _, i : get_localization_scores(interface.localget(i, "sliced"), dm=interface.localget(i, "-dm-path"), falloffs=[1, 5]) # type: ignore
        },
        "descriptors" : lambda _, i: optimize_descs(interface.localget(i, "-features-sliced"), interface.localget(i, "-dm-path"), n_descriptors=5, tol=0.05, heat=.4, exponent = 3),
        "apply_descriptors" : lambda _, i:  [desc_from_combination(interface.localget(i, "-features-sliced"), interface.localget(i, "descriptors")[j]) for j in range(len(interface.localget(i, "descriptors")))],
        "scores" : lambda _, i : get_localization_scores({j : d for j, d in enumerate(interface.localget(i, "apply_descriptors"))}, dm=interface.localget(i, "dm-path"))
    })
    return interface
