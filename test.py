import argparse

import scipy.io
from visualization_utils import confidence_plotter, show, save_as_npy

from utils import get_cm_backend

if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--backend",
        type=str,
        default="octave",
        help="Backend to use. Can be 'numpy' or 'octave'",
    )
    argparser.add_argument(
        "--sink_mode",
        type=str,
        default="mid",
        help="Sink mode to use. Can be 'all', 'mid' or 'min'",
    )
    # There is also the mask mode where you can provide a mask for the sink
    # but to keep this example simple we will not use it here

    args = argparser.parse_args()

    # Import confidence map function from the selected backend
    ConfidenceMap = get_cm_backend(args.backend)

    # Load neck data and call confidence estimation for B-mode with default parameters
    img = scipy.io.loadmat("data/neck.mat")["img"]
    cm = ConfidenceMap(
        alpha=2.0, beta=90.0, gamma=0.0, sink_mode=args.sink_mode
    )
    map_ = cm(img)
    save_as_npy(map_, "data/neck_result.npy")
    confidence_plotter(img, map_)

    # Load femur data and call confidence estimation for B-mode with default parameters
    img = scipy.io.loadmat("data/femur.mat")["img"]
    cm = ConfidenceMap(
        alpha=2.0, beta=90.0, gamma=0.06, sink_mode=args.sink_mode
    )
    map_ = cm(img)
    save_as_npy(map_, "data/femur_result.npy")

    confidence_plotter(img, map_)

    show()
