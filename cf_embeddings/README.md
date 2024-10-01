# CF Embeddings

This folder contains essential components for creating and managing collaborative filtering embeddings. Let's dive into each file and its functions.

## create_embeddings.py

This script is the heart of our embedding creation process. It handles the creation of item embeddings using the Alternating Least Squares (ALS) algorithm (but you can use any other such as SVD).

### Key Functions:

1. `normalize_matrix(ui_matrix: csr_matrix, method: str = 'bm_25') -> csr_matrix`
   - Normalizes the user-item interaction matrix using various methods (row-wise, column-wise, TF-IDF, or BM25).
   - Parameters:
     - `ui_matrix`: Sparse user-item interaction matrix
     - `method`: Normalization method ('row', 'column', 'tf_idf', 'bm_25')
   - Returns: Normalized user-item interaction matrix

2. `items_embeddings(ui_matrix: csr_matrix, item_ids: np.ndarray, dim: int, **kwargs) -> np.ndarray`
   - Creates item embeddings using the ALS algorithm.
   - Parameters:
     - `ui_matrix`: User-item interaction matrix
     - `item_ids`: Array of item IDs
     - `dim`: Dimensionality of embeddings
     - `**kwargs`: Additional parameters for ALS
   - Returns: Array of item embeddings

## normalizations.py

This file contains various normalization techniques for the user-item matrix.

### Class: Normalization

Static methods:
1. `by_column(matrix: csr_matrix) -> csr_matrix`
   - Normalizes the matrix by column
2. `by_row(matrix: csr_matrix) -> csr_matrix`
   - Normalizes the matrix by row
3. `tf_idf(matrix: csr_matrix) -> csr_matrix`
   - Applies TF-IDF normalization
4. `bm_25(matrix: csr_matrix) -> csr_matrix`
   - Applies BM25 normalization (not shown in the provided context, but likely implemented)

## user_item_matrix.py

This file (not shown in the context) likely contains the `UserItemMatrix` class, which is responsible for creating the user-item interaction matrix from raw data.

### Presumed Class: UserItemMatrix

Methods:
1. `__init__(self, sales_df: pd.DataFrame)`
   - Initializes the UserItemMatrix object with sales data
2. `csr_matrix` property
   - Returns the user-item matrix in CSR (Compressed Sparse Row) format

## Usage

To create embeddings:
1. Load your sales data
2. Create a UserItemMatrix object
3. Normalize the matrix using the desired method
4. Use the `items_embeddings` function to generate embeddings

## Sample Data

In the `data` folder, you can find dummy samples of data and embeddings that can be used for testing and familiarization with the format:

1. Sample of raw sales data
2. Sample of generated item embeddings

These files will help you understand the structure of input data and the expected format of output embeddings, which will simplify the integration and use of the module in your project.




