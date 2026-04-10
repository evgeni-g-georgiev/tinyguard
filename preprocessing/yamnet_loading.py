import warnings

from ai_edge_litert.interpreter import Interpreter

from config import YAMNET_PATH


def load_yamnet(path=YAMNET_PATH):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # experimental_preserve_all_tensors=True is required to read intermediate
        # tensors (e.g. the embedding layer at index EMB_IDX) after invoke().
        # Without it, only the final output tensor is accessible.
        interp = Interpreter(str(path), experimental_preserve_all_tensors=True)
    interp.allocate_tensors()
    return interp, interp.get_input_details()
