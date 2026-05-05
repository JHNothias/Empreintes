import interface as Int
import pprint
import taichi as ti
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from main_functions import *
import os
import pandas as pd

# Paramètres, ne toucher à rien d'autre. Le calcul est assez long et génère énormément de données, plusieurs Go. J'arrivais à 60 Go pour 'Les Espaces Acoustiques' de Grisey.
# Mon meilleur conseil est de mettre le dossier où toutes les données sont sauvegardées sur un disque externe.
audio_folder = 'Path/to/Folder'
save_folder = 'Path/to/Folder'
dataset_title = 'Analyse_les_espaces_acoustiques' #Titre de l'ensemble de données
filenames = ["Partiels.wav", "...etc.wav"] #Liste des audio à analyser, que du wav pour flucoma
slicelength_seconds = 2 # rester sur des petites valeurs...
n_interpolated_slices = 4 # rester sur des petites valeurs...
save_spectrogram = False # Prend énormément de place sinon







scores = dict()
audio_folder = audio_folder.removesuffix('/') + '/'
save_folder = save_folder.removesuffix('/') + '/'
titles = [filename.removesuffix('.wav') for filename in filenames]


print("Imports OK, started baking.")
for filename in filenames:
    interface = Int.interface()
    ti.reset()
    ti.init(arch=ti.gpu, default_fp=ti.types.f32, log_level=ti.ERROR)
    print('-'*10 + f"Started the processing of {filename}." + '-'*10)

    path = audio_folder + filename
    savepath = save_folder + dataset_title +f'/{filename.removesuffix('.wav')}_data.pickle'
    
    interface.set('init_settings-title', filename.removesuffix('.wav'))
    interface.get('init_settings-title')
    interface.set('init_settings-filepath', path)
    interface.set('init_settings-save_filepath', savepath)
    interface.set('init_settings-slicelength_seconds', slicelength_seconds)
    interface.set('init_settings-n_interpolated_slices', n_interpolated_slices)
    interface.set('init_settings-slice_distance', 'concordance')

    
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    
    print("starting parameters :")
    pprint.pp(interface.getsubtree("init_settings", recompute=True))
    interface.report = True
    interface.memo = dict()
    interface.get('features-scores', recompute=True)
    if not save_spectrogram :
        print('Deleting spectrogram...')
        interface.setmemo('spectrograms-mel', None)
    print("Saving...")
    interface.save(interface.get('init_settings-save_filepath'))
    
    print('Generating descriptors plot...')
    fig = px.line(np.stack(list(interface.get('features-sliced').values())).T)
    for i, trace in enumerate(fig.data):
        trace.name = list(interface.get('features-sliced').keys())[i] # type: ignore
    
    figsavepath = save_folder + dataset_title + f"/{interface.get('init_settings-title')}/{interface.get('init_settings-title')}_features.html"
    os.makedirs(os.path.dirname(figsavepath), exist_ok=True)
    
    fig.write_html(
        figsavepath,
        full_html=True,
        include_plotlyjs="cdn",
        auto_open=False,
        config={"displayModeBar": False}
    )
    
    print('Generating descriptor networks...')
    for k in interface.get('features-sliced').keys():
        print(f"{k}: {interface.get('features-scores')[k]}")
        fig = network3d(G=interface.get('graph'), xyz=interface.get('embedding-path'), nodecolors=interface.get('features-sliced')[k])
        
        figsavepath = save_folder + dataset_title + f"/{interface.get('init_settings-title')}/{interface.get('init_settings-title')}_network_{k}.html"
        os.makedirs(os.path.dirname(figsavepath), exist_ok=True)
        
        fig.write_html(
        figsavepath,
        full_html=True,
        include_plotlyjs="cdn",
        auto_open=False,
        config={"displayModeBar": False}
        )
    
    scores[interface.get('init_settings-title')] = interface.get('features-scores')
    
    del interface
    print('Next !')


print('Finished. Generating scores table ...')

descriptors = list(scores[list(scores.keys())[0]].keys())
data = [[round(100*s, 1) for s in S.values()] for S in scores.values()]
headers = ['Descripteur'] + titles
values = [descriptors] + [[round(100*s, 1) for s in scores[title].values()] for title in titles]
df = pd.DataFrame(values).T.sort_values(0).T # type: ignore


fig = go.Figure(data=[go.Table(header=dict(values=headers),
                 cells=dict(values=df.values))])
fig.update_layout(
    updatemenus=[
        {
            "buttons": [
                {
                    "method": "restyle",
                    "label": b["l"],
                    "args": [{"cells": {"values": df.T.sort_values(b["c"]).T.values}},[0],],
                }
                for b in [{"l": h, "c": i} for i, h in enumerate(headers)]
            ],
            "direction": "down",
            "y": 1,
        }
    ]
)
figsavepath = save_folder + dataset_title + f"/{dataset_title}_Scores_table.html"
os.makedirs(os.path.dirname(figsavepath), exist_ok=True)

fig.write_html(figsavepath,
        full_html=True,
        include_plotlyjs="cdn",
        auto_open=False,
        config={"displayModeBar": False}
        )
print('Finished !')

    