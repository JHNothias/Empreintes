# %%
try :
    import flucoma
except ImportError:
    raise ImportError("Aborting because `python-flucoma` could not be imported despite being requested.\nIf you want to use FluCoMa, please install this library with `pip install python-flucoma`.\nAlso make sure that the FluCoMa command line interface is installed and visible on your PATH.")

import shutil
if shutil.which("fluid-ampfeature") is None:
    raise ImportError("Aborting because the FluCoMa command line interface is not installed or visible on your PATH despite being requested.\nIf you want to use FluCoMa, you can install it form here : `https://www.flucoma.org/download/`.")

from .interface import flucoma_interface
