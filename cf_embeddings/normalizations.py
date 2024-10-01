from scipy.sparse import csr_matrix
import numpy as np


class Normalization:
    @staticmethod
    def by_column(matrix: csr_matrix) -> csr_matrix:
        """Normalization by column

        Args:
            matrix (csr_matrix): User-Item matrix of size (N, M)

        Returns:
            csr_matrix: Normalized matrix of size (N, M)
        """
        col_sum = matrix.sum(axis=0)
        norm_matrix = csr_matrix(matrix.multiply(1 / col_sum))
        return norm_matrix

    @staticmethod
    def by_row(matrix: csr_matrix) -> csr_matrix:
        """Normalization by row

        Args:
            matrix (csr_matrix): User-Item matrix of size (N, M)

        Returns:
            csr_matrix: Normalized matrix of size (N, M)
        """
        row_sum = matrix.sum(axis=1)
        norm_matrix = csr_matrix(matrix.multiply(1 / row_sum))
        return norm_matrix

    @staticmethod
    def tf_idf(matrix: csr_matrix) -> csr_matrix:
        """Normalization using tf-idf

        Args:
            matrix (csr_matrix): User-Item matrix of size (N, M)

        Returns:
            csr_matrix: Normalized matrix of size (N, M)
        """

        N = matrix.shape[0]

        tf = Normalization.by_row(matrix)
        idf = np.log(N / (matrix > 0).sum(axis=0))

        norm_matrix = csr_matrix(tf.multiply(idf))

        return norm_matrix

    @staticmethod
    def bm_25(
        matrix: csr_matrix, k1: float = 2.0, b: float = 0.75
    ) -> csr_matrix:
        """Normalization based on BM-25

        Args:
            matrix (csr_matrix): User-Item matrix of size (N, M)

        Returns:
            csr_matrix: Normalized matrix of size (N, M)
        """
        N = matrix.shape[0]
        d_len = matrix.sum(axis=1)
        avg_d_len = d_len.mean()
        delta = k1 * ((1 - b) + b * d_len / avg_d_len)

        tf = Normalization.by_row(matrix)
        tf = tf.multiply(1 / delta).power(-1)
        tf.data += 1
        tf = tf.power(-1).multiply(k1 + 1)

        idf = np.log(N / (matrix > 0).sum(axis=0))

        norm_matrix = csr_matrix(tf.multiply(idf))

        return norm_matrix
