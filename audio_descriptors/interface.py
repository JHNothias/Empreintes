from LazyTree import LazyTree
from Custom.interface import misc_interface
from FluCoMa.interface import flucoma_interface

def interface(settings):
    descriptors_interface = LazyTree({
        "inputs" : {
            "slices" : None,
        },
        "raw" : {
            "misc" : misc_interface.new(),
            "flucoma" : flucoma_interface.new() if any(['flucoma' in descriptors_interface.get('requested_descriptors')])
        },
        "derived" : {
            ...
            for lib in descriptors_interface.spec["raw"]
            for desc in descriptors_interface.spec["raw"][lib]["descriptors"]
        },
        "requested" : {
            "descriptors" : lambda tree, i : [], # 'flucoma-centroid'
            "transforms" : lambda tree, i : []
        },
        "settings" : {
            "transforms" :
                {}
        }

    })
