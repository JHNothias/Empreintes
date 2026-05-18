# %%
import numpy as np
from flucoma import fluid
import matplotlib.pyplot as plt

wfm = fluid.FluidSingleOutput("/home/eilen/Audio/samples/Flock.wav")

# %%
params = {'padding':0, 'algorithm':0, 'kernelsize':3, 'filtersize':1, 'fftsettings':[1024, -1, -1]}
res = np.asarray(fluid.noveltyfeature(wfm, **params, numchans=1))
print(res.shape)

# %%

plt.plot(res.T)
plt.show()
# %%

from empreintes.audio_descriptors.FluCoMa import flucoma_interface # type:ignore
from empreintes.LazyTree import LazyTree # type:ignore

# %%

flucoma_interface.set(id = "inputs-flucomawfm", value = wfm)
print(flucoma_interface.d.keys())
# %%

descriptors = flucoma_interface.getsubtree('descriptors')

print(descriptors)
# %%
for d, v in descriptors.items():
    plt.plot(v)
    plt.title(d)
    plt.show()

# %%

print(descriptors['novelty'])
# %%
from pprint import pp
wfm = fluid.FluidSingleOutput("/home/eilen/Audio/samples/Waves and stones.wav")

pp(flucoma_interface.times_changed)

flucoma_interface.set(id = "inputs-flucomawfm", value = wfm)
pp(flucoma_interface.times_changed)
flucoma_interface.set(id = "settings-centroid", value = {'fftsettings':[1024, -1, -1]})
pp(flucoma_interface.times_changed)

descriptors = flucoma_interface.getsubtree('descriptors')
pp(flucoma_interface.times_changed)

for d, v in descriptors.items():
    plt.plot(v)
    plt.title(d)
    plt.show()
