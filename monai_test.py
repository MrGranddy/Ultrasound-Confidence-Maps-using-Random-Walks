import scipy.io
from visualization_utils import confidence_plotter, show, save_as_npy

from confidence_map.confidence_monai import UltrasoundConfidenceMap

if __name__ == "__main__":

    # Load neck data and call confidence estimation for B-mode with default parameters
    img = scipy.io.loadmat("data/neck.mat")["img"]
    cm = UltrasoundConfidenceMap(
        alpha=2.0, beta=90.0, gamma=0.0
    )
    map_ = cm(img)
    save_as_npy(map_, "data/neck_result.npy")
    confidence_plotter(img, map_)

    # Load femur data and call confidence estimation for B-mode with default parameters
    img = scipy.io.loadmat("data/femur.mat")["img"]
    cm = UltrasoundConfidenceMap(
        alpha=2.0, beta=90.0, gamma=0.06
    )
    map_ = cm(img)
    save_as_npy(map_, "data/femur_result.npy")

    confidence_plotter(img, map_)

    show()
