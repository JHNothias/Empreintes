import argparse
import pprint
import os
from generate_data import run_analysis_pipeline

def parse_args():
    parser = argparse.ArgumentParser(
        description="Basic cli interface for the Empreintes algorithm.\nFor now, the analysis pipeline is fixed and requires flucoma-python and taichi to be installed and operational."
    )

    parser.add_argument(
        "--audio_folder",
        type=str,
        required=True,
        help="Path to folder containing audio files (.wav only for the moment)."
    )

    parser.add_argument(
        "--save_folder",
        type=str,
        required=True,
        help="Path where output data will be stored. A new folder is created per dataset."
    )

    parser.add_argument(
        "--dataset_title",
        type=str,
        default="dataset",
        help="Name of the dataset, i.e. the name of the folder created in the `save_folder`."
    )

    parser.add_argument(
        "--filenames",
        nargs="+",
        default=None,
        help="List of .wav files to process (default: all .wav in audio_folder)."
    )

    parser.add_argument(
        "--slicelength_seconds",
        type=float,
        default=2.0,
        help="Slice length in seconds (default : 2 seconds)."
    )

    parser.add_argument(
        "--n_interpolated_slices",
        type=int,
        default=4,
        help="Number of interpolated slices (default : 4. For the default slicelength of 2 seconds, this means a slice every 0.5 seconds)."
    )

    parser.add_argument(
        "--save_spectrogram",
        action="store_true",
        help="Enable spectrogram saving in the pickle for later analysis (REQUIRES A LOT OF SPACE)."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve paths
    audio_folder = os.path.abspath(args.audio_folder)
    save_folder = os.path.abspath(args.save_folder)

    os.makedirs(save_folder, exist_ok=True)

    if args.filenames is None:
        args.filenames = [
            f for f in os.listdir(audio_folder)
            if f.lower().endswith(".wav")
        ]
        if not args.filenames:
            raise ValueError(f"No .wav files found in {audio_folder}")
    args.filenames = sorted(args.filenames)

    config = {
        "audio_folder": audio_folder,
        "save_folder": save_folder,
        "dataset_title": args.dataset_title,
        "filenames": args.filenames,
        "slicelength_seconds": args.slicelength_seconds,
        "n_interpolated_slices": args.n_interpolated_slices,
        "save_spectrogram": args.save_spectrogram,
    }

    print("\n=== CONFIGURATION ===")
    pprint.pp(config)
    print("=====================\n")

    # ---- YOUR PIPELINE ENTRY POINT HERE ----
    # Example (replace with your real function):
    run_analysis_pipeline(
        audio_folder=audio_folder,
        save_folder=save_folder,
        dataset_title=args.dataset_title,
        filenames=args.filenames,
        slicelength_seconds=args.slicelength_seconds,
        n_interpolated_slices=args.n_interpolated_slices,
        save_spectrogram=args.save_spectrogram
    )


if __name__ == "__main__":
    main()
