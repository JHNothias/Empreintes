from LazyTree import LazyTree
from .roughness import roughness_self
import numpy as np

misc_interface = LazyTree({
    'inputs': {
        'wfm': lambda self, i: None,
        "spectra" : lambda self, i: None,
        "frequencies" : lambda self, i : None,
        "sr" : lambda self, i: None
    },
    'compute': {
        'roughness' : lambda self, i : roughness_self(self.localget(i, "-inputs-spectra"), self.localget(i, "-inputs-frequencies"))
    },
    'descriptors': {
        "roughness" : lambda self, i : self.localget(i, "-compute-roughness"),
        "time" : lambda self, i : np.arange(0, len(self.localget(i, "-inputs-wfm")), dtype = float)/self.localget(i, "-inputs-sr")
    },
    'map': {
    },
    'settings': {
        'roughness' : lambda self, i : None,
        "time" : lambda self, i : None
    }
}, id="", memoize=True)
