from confidence_map.confidence_map_scipy import ConfidenceMap as ConfidenceMap_scipy
from confidence_map.confidence_map_oct import ConfidenceMap as ConfidenceMap_oct


def get_cm_backend(backend):
    """Get the confidence map function from the selected backend

    Args:
        backend (str): Backend to use. Can be 'numpy' or 'cupy' or 'octave'

    Raises:
        NotImplementedError: If the backend is not supported

    Returns:
        class: ConfidenceMap class from the selected backend
    """

    if backend == "numpy":
        ConfidenceMap = ConfidenceMap_scipy
    elif backend == "octave":
        ConfidenceMap = ConfidenceMap_oct
    else:
        # Give error message if the backend is not supported
        raise NotImplementedError(
            f'The backend "{backend}" is not supported.'
        )
    return ConfidenceMap