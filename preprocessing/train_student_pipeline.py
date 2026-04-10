

from preprocessing.extract_embeddings import extract_embeddings
from preprocessing.compute_mels import compute_mels
from preprocessing.split_mimii import split_mimii


def preprocessing():
    extract_embeddings()
    compute_mels()
    #split_mimii()



if __name__ == "__main__":
    preprocessing()
