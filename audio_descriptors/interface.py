from LazyTree import LazyTree
from Custom.interface import misc_interface
from FluCoMa.interface import flucoma_interface

def with_settings(tree : LazyTree, settings_location : str):
    res = tree.new()
    tree.se

descriptors_interface = LazyTree({
    "raw" : {
        "misc" : misc_interface.new(),
        "flucoma" : flucoma_interface.new()
    },
    "derived" : {

    },
    "settings" : {
        "misc"
    }
})
