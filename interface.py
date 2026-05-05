from LazyTree import LazyTree
from main_functions import *
import librosa

def interface():
    interface = LazyTree({
        "init_settings" : {
            "title" : lambda i : None,
            "filepath" : lambda i : None,
            "hop_length": lambda i: 256,
            "silence_treatment": lambda i: 'keep',
            "silence_threshold": lambda i: 0.05,
            "slice_threshold": lambda i: 0.18,
            "n_mels": lambda i: 512,
            "sr": lambda i: 44100,
            "slicelength_seconds": lambda i: 2,
            "n_interpolated_slices": lambda i: 1,
            "method" : lambda i: "even",
            "slice_distance" : lambda i : "concordance",
            "crossing_parameters" : lambda i : (3, 3, 3),
            "squash" : lambda i : True,
            "save_filepath" : lambda i : ''
        },
        "waveform" : lambda i: fluid.FluidSingleOutput(interface.localget(i, "init_settings-filepath")),
        "slices" : lambda i: interface.localget(i, "_slices_output")[0],
        "silences" : lambda i: interface.localget(i, "_slices_output")[1],
        "_slices_output" : lambda i : slice_wfm(interface.localget(i, "waveform"),
                                     method=interface.localget(i, "init_settings-method"),
                                     slice_threshold=interface.localget(i, "init_settings-slice_threshold"),
                                     minslicelength=50,
                                     silence_treatment=interface.localget(i, "init_settings-silence_treatment"),
                                     silence_threshold=interface.localget(i, "init_settings-silence_threshold"),
                                     n_interpolated_slices=interface.localget(i, "init_settings-n_interpolated_slices"),
                                     slicelength=interface.localget(i, "init_settings-slicelength_seconds")*interface.localget(i, "init_settings-sr")),
        "spectrograms" : {
            "stft" : lambda i : np.abs(librosa.stft(y=np.asarray(interface.localget(i, "-waveform")), hop_length=interface.localget(i, "-init_settings-hop_length"), n_fft=4092)),
            "mel" : lambda i : np.abs(librosa.feature.melspectrogram(y=np.asarray(interface.localget(i, "-waveform")),
                                      hop_length=interface.localget(i, "-init_settings-hop_length"),
                                      sr=interface.localget(i, "-init_settings-sr"), power=1,
                                      n_mels = interface.localget(i, "-init_settings-n_mels"), fmin=1, n_fft=4092)),
            "stft_freq" : lambda i : librosa.fft_frequencies(n_fft=4096, sr=interface.localget(i, "-init_settings-sr")),
            "mel_freq" : lambda i : librosa.mel_frequencies(n_mels=interface.localget(i, "-init_settings-n_mels"), fmin=1, fmax = interface.localget(i, "-init_settings-sr")/2)
        },
        "dm" : {
            "dm" : lambda i : simple_preprocess(interface.localget(i, interface.localget(i, "-init_settings-slice_distance"))),
            "concordance" : lambda i : concordance_matrix_spectrogram(spectrogram=np.log(1+(interface.localget(i, "-spectrograms-mel")[0] + interface.localget(i, "-spectrograms-mel")[1])/2), slices=interface.localget(i, "-slices"), hop_length=interface.localget(i, "-init_settings-hop_length")),
            "concordance_harmonic" : lambda i : concordance_matrix_spectrogram(spectrogram=librosa.decompose.hpss(np.log(1+(interface.localget(i, "-spectrograms-mel")[0] + interface.localget(i, "-spectrograms-mel")[1])/2), kernel_size=15, margin=1)[0], slices=interface.localget(i, "-slices"), hop_length=interface.localget(i, "-init_settings-hop_length")),
            "wasserstein" : lambda i : wasserstein_matrix(np.log(1+(interface.localget(i, "-spectrograms-mel")[0] + interface.localget(i, "-spectrograms-mel")[1])/2),
                                                                 interface.localget(i, "-slices"), hop_length=interface.localget(i, "-init_settings-hop_length"), p=2),
            "path" : lambda i : pathDistances(interface.localget(i, "-graph"))
        },
        "graph" : lambda i : crossingGraph(interface.localget(i, "-dm-dm"), *interface.localget(i, "-init_settings-crossing_parameters")),
        "embedding" : {
            "dm" : lambda i : xyzMDS(interface.localget(i, "-dm-dm")),
            "path": lambda i : xyzMDS(interface.localget(i, "-dm-path")),
        },
        "features" : {
            "raw" : lambda i : compute_features(interface.localget(i, "-waveform"),
                                                 interface.localget(i, "-slices"),
                                                 interface.localget(i, "-init_settings-hop_length"),
                                                 interface.localget(i, "-spectrograms-stft"),
                                                 interface.localget(i, "-spectrograms-stft_freq"), mergeLR = True),
            "sliced" : lambda i : slice_features(interface.localget(i, "raw"), interface.localget(i, "-slices"),
                                                 squash = interface.localget(i, "-init_settings-squash"), squashfactor = 10),
            "scores" : lambda i : get_localization_scores(interface.localget(i, "sliced"), dm=interface.localget(i, "-dm-path"), falloffs=[1, 5]) # type: ignore
        },
        "descriptors" : lambda i: optimize_descs(interface.localget(i, "-features-sliced"), interface.localget(i, "-dm-path"), n_descriptors=5, tol=0.05, heat=.4, exponent = 3),
        "apply_descriptors" : lambda i:  [desc_from_combination(interface.localget(i, "-features-sliced"), interface.localget(i, "descriptors")[j]) for j in range(len(interface.localget(i, "descriptors")))],
        "scores" : lambda i : get_localization_scores({j : d for j, d in enumerate(interface.localget(i, "apply_descriptors"))}, dm=interface.localget(i, "dm-path"))
    })
    return interface