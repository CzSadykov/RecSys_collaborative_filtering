import os
import os.path
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import pickle

from implicit.als import AlternatingLeastSquares as ALS
from scipy.sparse import csr_matrix
# from scipy import linalg -- if you want to do factorization via SVD

from user_item_matrix import UserItemMatrix
from normalizations import Normalization

# unnecessary part I added for my own convenience, feel free to delete
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
env_path = os.path.join(parent_dir, '.env')
load_dotenv(env_path, override=True)
# if you remove the lines above, uncomment the following line
# load_dotenv()

data_path = os.getenv("data_path")
embeddings_path = os.getenv("embeddings_path")

sales = pd.read_csv(data_path)
sales_df = pd.DataFrame(sales)

ui_matrix = UserItemMatrix(sales_df).csr_matrix


def normalize_matrix(
        ui_matrix: csr_matrix, method: str = 'bm_25'
        ) -> csr_matrix:
    """
    Normalizes the user-item interaction matrix.

    This function applies the chosen normalization method to the input matrix.
    Available normalization methods include: row-wise, column-wise, TF-IDF, and BM25.

    Parameters:
    ----------
    ui_matrix : csr_matrix
        Sparse user-item interaction matrix.
    method : str, default 'bm_25'
        Normalization method. Valid values: 'row', 'column', 'tf_idf', 'bm_25'.

    Returns:
    -------
    csr_matrix
        Normalized user-item interaction matrix.

    Raises:
    ------
    ValueError
        If an unknown normalization method is specified.
    """
    if method == 'row':
        return Normalization.by_row(ui_matrix)
    elif method == 'column':
        return Normalization.by_column(ui_matrix)
    elif method == 'tf_idf':
        return Normalization.tf_idf(ui_matrix)
    elif method == 'bm_25':
        return Normalization.bm_25(ui_matrix)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


normalized_matrix = normalize_matrix(ui_matrix, method='bm_25')

# initialize default ids if not provided
item_ids = np.array(range(ui_matrix.shape[1]))


def items_embeddings(
        ui_matrix: csr_matrix, item_ids: np.ndarray, dim: int, **kwargs
        ) -> np.ndarray:
    """
    Creates item embeddings using the Alternating Least Squares (ALS) algorithm.

    This function trains an ALS model on the normalized user-item interaction matrix
    and returns a dictionary of item embeddings.

    Parameters:
    ----------
    ui_matrix : csr_matrix
        Normalized user-item interaction matrix.
    item_ids : np.ndarray
        Array of item identifiers.
    dim : int
        Dimensionality of the embeddings.
    **kwargs : dict
        Additional parameters for configuring the ALS model:
        - regularization : float, default 0.1
            Regularization parameter.
        - iterations : int, default 10
            Number of training iterations.
        - random_state : int, default 42
            Seed for the random number generator.
        - dtype : data type, default np.float32
            Data type for computations.
        - use_gpu : bool, default False
            Whether to use GPU for computations.

    Returns:
    -------
    dict
        A dictionary where keys are item identifiers and values are their embeddings.
    """
    model = ALS(
        factors=dim,
        regularization=kwargs.get("regularization", 0.1),
        iterations=kwargs.get("iterations", 10),
        random_state=kwargs.get("random_state", 42),
        dtype=kwargs.get("dtype", np.float32),
        use_gpu=kwargs.get("use_gpu", False)
        )
    model.fit(ui_matrix, show_progress=True)
    embeddings = model.item_factors
    return {
        item_id: embedding for item_id, embedding
        in zip(item_ids, embeddings)
        }

# Using SVD from scipy.linalg might give you slightly better results

# def items_embeddings(ui_matrix: csr_matrix, dim: int) -> np.ndarray:
#     """Build items embedding using SVD"""
#     U, S, V_t = linalg.svds(ui_matrix, k=dim)
#     return V_t.T


embeddings = items_embeddings(normalized_matrix, item_ids, dim=75)

with open(embeddings_path, 'wb') as f:
    pickle.dump(embeddings, f)

# print(np.load(embeddings_path, allow_pickle=True))  # to check out the file
