from empreintes.LazyTree import LazyTree #type:ignore
from flucoma import fluid
from empreintes.utils.batch_compute_node import batch_compute_node, batch_item_node #type:ignore
import numpy as np

flucoma_interface = LazyTree({
    'inputs': {
        'flucomawfm': lambda tree, i: None,
        "hopsize": lambda tree, i: None,
    },
    'compute': {
        'shape': batch_compute_node(
            compute_fn= lambda wfm, hopsize, params: np.asarray(fluid.spectralshape(wfm, **params, numchans=1)),
            inputs=['-inputs-flucomawfm', '-inputs-hopsize'],
            settings_location='-settings',
            map_location='-map-shape',
        ),
        'loudness': batch_compute_node(
            compute_fn= lambda wfm, hopsize, params: np.asarray(fluid.loudness(wfm, **params, numchans=1)),
            inputs=['-inputs-flucomawfm', '-inputs-hopsize'],
            settings_location='-settings',
            map_location='-map-loudness',
        ),
        'novelty': batch_compute_node(
            compute_fn= lambda wfm, hopsize, params: np.asarray(fluid.noveltyfeature(wfm, **params, numchans=1)),
            inputs=['-inputs-flucomawfm', '-inputs-hopsize'],
            settings_location='-settings',
            map_location='-map-novelty',
        ),
        'pitch': batch_compute_node(
            compute_fn= lambda wfm, hopsize, params: np.asarray(fluid.pitch(wfm, **params, numchans=1)),
            inputs=['-inputs-flucomawfm', '-inputs-hopsize'],
            settings_location='-settings',
            map_location='-map-pitch',
        ),
    },
    'descriptors': {
        'centroid': lambda tree, i : batch_item_node('-compute-shape', '-settings')(tree, i)[0],
        'spread': lambda tree, i : batch_item_node('-compute-shape', '-settings')(tree, i)[1],
        'skew': lambda tree, i : batch_item_node('-compute-shape', '-settings')(tree, i)[2],
        'kurtosis': lambda tree, i : batch_item_node('-compute-shape', '-settings')(tree, i)[3],
        'rolloff': lambda tree, i : batch_item_node('-compute-shape', '-settings')(tree, i)[4],
        'flatness': lambda tree, i : batch_item_node('-compute-shape', '-settings')(tree, i)[5],
        'crest': lambda tree, i : batch_item_node('-compute-shape', '-settings')(tree, i)[6],
        'pitch measured': lambda tree, i : batch_item_node('-compute-pitch', '-settings')(tree, i)[0],
        'pitch confidence': lambda tree, i : batch_item_node('-compute-pitch', '-settings')(tree, i)[1],
        'novelty': lambda tree, i : batch_item_node('-compute-novelty', '-settings')(tree, i),
        'loudness': lambda tree, i : batch_item_node('-compute-loudness', '-settings')(tree, i)[0],
        'true peak': lambda tree, i : batch_item_node('-compute-loudness', '-settings')(tree, i)[1],
    },
    'map': {
        'shape': lambda tree, i: ["centroid", "spread", "skew", "kurtosis", "rolloff", "flatness", "crest"],
        'loudness': lambda tree, i: ["loudness", "true peak"],
        'novelty': lambda tree, i: ["novelty"],
        'pitch': lambda tree, i: ["pitch measured", "pitch confidence"],
    },
    'settings': {
        'centroid': lambda tree, i:{'fftsettings':[1024, -1, -1]},
        'spread': lambda tree, i:{'fftsettings':[1024, -1, -1]},
        'skew': lambda tree, i:{'fftsettings':[1024, -1, -1]},
        'kurtosis': lambda tree, i:{'fftsettings':[1024, -1, -1]},
        'rolloff': lambda tree, i: {'fftsettings':[1024, -1, -1]},
        'flatness': lambda tree, i:{'fftsettings':[1024, -1, -1]},
        'crest': lambda tree, i:{'fftsettings':[1024, -1, -1]},
        'pitch measured': lambda tree, i:{'algorithm': 2, 'fftsettings': [1024, -1, -1], 'maxfreq' : 10000, 'minfreq' : 20, 'unit': 0},
        'pitch confidence': lambda tree, i:{'algorithm': 2, 'fftsettings': [1024, -1, -1], 'maxfreq' : 10000, 'minfreq' : 20, 'unit': 0},
        'novelty': lambda tree, i:{'padding':0, 'algorithm':0, 'kernelsize':3, 'filtersize':1, 'fftsettings':[1024, -1, -1]},
        'loudness': lambda tree, i: {'hopsize' : 512, 'windowsize':1024},
        'true peak': lambda tree, i: {'hopsize' : 512, 'windowsize':1024},
    }
}, id="", memoize=True)
