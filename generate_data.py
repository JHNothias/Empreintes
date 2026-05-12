import interface as Int
import pprint
import taichi as ti
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from main_functions import *
import os
import pandas as pd
from pathlib import Path



def run_analysis_pipeline(audio_folder:str, save_folder:str, dataset_title:str, filenames: list[str],
    slicelength_seconds:float, n_interpolated_slices:int, save_spectrogram:bool, colorscale:str, restart:bool):

    scores = dict()
    audio_folder = audio_folder.removesuffix('/') + '/'
    save_folder = save_folder.removesuffix('/') + '/'
    titles = [filename.removesuffix('.wav') for filename in filenames]


    print("Imports OK, started baking.")
    for filename in filenames:
        title = filename.removesuffix(".wav")
        interface = Int.interface()
        savepath = save_folder + dataset_title +f'/{title}_data.pickle'
        skip_recompute_flag = False
        if restart and title + '_data.pickle' in os.listdir(Path(save_folder + dataset_title)):
            print(f"found {title + '_data.pickle'} in savepath, trying to load it.")
            try :
                interface.load(savepath)
                skip_recompute_flag = True
                print("Loading succeded, skipping computation.")
            except :
                print("Loading failed, recomputing.")

        if not skip_recompute_flag :
            ti.reset()
            ti.init(arch=ti.gpu, default_fp=ti.types.f32, log_level=ti.ERROR)
            print('-'*10 + f"Started the processing of {filename}." + '-'*10)

            path = audio_folder + filename

            interface.set('init_settings-title', title)
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
                interface.setmemo('spectrograms-mel', None, invalidate = False)
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
            fig = network3d(G=interface.get('graph'), xyz=interface.get('embedding-path'), nodecolors=interface.get('features-sliced')[k], nodecolorscale=colorscale, edgecolorscale=colorscale)

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
