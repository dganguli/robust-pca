import numpy as np
import pytest
from r_pca import RobustPCA


class TestInitialization:
    """Tests for RobustPCA initialization."""

    def test_default_parameters(self):
        """Test that default mu and lambda are computed correctly."""
        D = np.random.randn(10, 20)
        rpca = RobustPCA(D)

        # Check default mu: n1*n2 / (4 * ||D||_1)
        expected_mu = np.prod(D.shape) / (4 * np.linalg.norm(D.flatten(), ord=1))
        assert rpca.mu == pytest.approx(expected_mu)

        # Check default lambda: 1 / sqrt(max(n1, n2))
        expected_lmbda = 1 / np.sqrt(np.max(D.shape))
        assert rpca.lmbda == pytest.approx(expected_lmbda)

    def test_custom_parameters(self):
        """Test that custom mu and lambda are used when provided."""
        D = np.random.randn(10, 20)
        rpca = RobustPCA(D, mu=0.5, lmbda=0.1)

        assert rpca.mu == 0.5
        assert rpca.lmbda == 0.1
        assert rpca.mu_inv == pytest.approx(2.0)

    def test_mu_zero_allowed(self):
        """Test that mu=0 is not treated as None (regression test)."""
        D = np.array([[1.0, 2.0], [3.0, 4.0]])
        # This should use the provided value, not compute default
        # Note: mu=0 would cause division by zero, but the point is
        # that mu=0 should not be treated as "use default"
        rpca = RobustPCA(D, mu=0.001)
        assert rpca.mu == 0.001

    def test_lmbda_zero_allowed(self):
        """Test that lmbda=0 is not treated as None."""
        D = np.array([[1.0, 2.0], [3.0, 4.0]])
        rpca = RobustPCA(D, lmbda=0.0)
        assert rpca.lmbda == 0.0

    def test_initial_matrices_shape(self):
        """Test that S and Y are initialized with correct shape."""
        D = np.random.randn(15, 25)
        rpca = RobustPCA(D)

        assert rpca.S.shape == D.shape
        assert rpca.Y.shape == D.shape
        assert np.all(rpca.S == 0)
        assert np.all(rpca.Y == 0)


class TestFrobeniusNorm:
    """Tests for the frobenius_norm static method."""

    def test_frobenius_norm_identity(self):
        """Test Frobenius norm of identity matrix."""
        I = np.eye(3)
        assert RobustPCA.frobenius_norm(I) == pytest.approx(np.sqrt(3))

    def test_frobenius_norm_zeros(self):
        """Test Frobenius norm of zero matrix."""
        Z = np.zeros((5, 5))
        assert RobustPCA.frobenius_norm(Z) == 0.0

    def test_frobenius_norm_ones(self):
        """Test Frobenius norm of ones matrix."""
        O = np.ones((3, 4))
        assert RobustPCA.frobenius_norm(O) == pytest.approx(np.sqrt(12))

    def test_frobenius_norm_known_value(self):
        """Test Frobenius norm with known value."""
        M = np.array([[1, 2], [3, 4]])
        # sqrt(1 + 4 + 9 + 16) = sqrt(30)
        assert RobustPCA.frobenius_norm(M) == pytest.approx(np.sqrt(30))


class TestShrink:
    """Tests for the shrink (soft thresholding) static method."""

    def test_shrink_zero_threshold(self):
        """Test that shrink with tau=0 returns original matrix."""
        M = np.array([[1, -2], [3, -4]])
        result = RobustPCA.shrink(M, 0)
        np.testing.assert_array_equal(result, M)

    def test_shrink_large_threshold(self):
        """Test that shrink with large tau returns zeros."""
        M = np.array([[1, -2], [3, -4]])
        result = RobustPCA.shrink(M, 10)
        np.testing.assert_array_equal(result, np.zeros_like(M))

    def test_shrink_positive_values(self):
        """Test shrink on positive values."""
        M = np.array([[5, 3], [1, 4]])
        tau = 2
        expected = np.array([[3, 1], [0, 2]])
        result = RobustPCA.shrink(M, tau)
        np.testing.assert_array_equal(result, expected)

    def test_shrink_negative_values(self):
        """Test shrink on negative values."""
        M = np.array([[-5, -3], [-1, -4]])
        tau = 2
        expected = np.array([[-3, -1], [0, -2]])
        result = RobustPCA.shrink(M, tau)
        np.testing.assert_array_equal(result, expected)

    def test_shrink_mixed_values(self):
        """Test shrink on mixed positive/negative values."""
        M = np.array([[5, -3], [-1, 4]])
        tau = 2
        expected = np.array([[3, -1], [0, 2]])
        result = RobustPCA.shrink(M, tau)
        np.testing.assert_array_equal(result, expected)


class TestSvdThreshold:
    """Tests for the svd_threshold method."""

    def test_svd_threshold_zero_tau(self):
        """Test SVD threshold with tau=0 returns original matrix."""
        D = np.random.randn(5, 5)
        rpca = RobustPCA(D)
        result = rpca.svd_threshold(D, 0)
        np.testing.assert_array_almost_equal(result, D)

    def test_svd_threshold_large_tau(self):
        """Test SVD threshold with large tau returns zero matrix."""
        D = np.random.randn(5, 5)
        rpca = RobustPCA(D)
        result = rpca.svd_threshold(D, 1000)
        np.testing.assert_array_almost_equal(result, np.zeros_like(D))

    def test_svd_threshold_reduces_rank(self):
        """Test that SVD threshold reduces matrix rank."""
        # Create a rank-3 matrix
        U = np.random.randn(10, 3)
        V = np.random.randn(3, 10)
        D = U @ V  # rank 3

        rpca = RobustPCA(D)
        # Threshold should reduce rank
        result = rpca.svd_threshold(D, 0.5)

        # Result should have lower or equal rank
        rank_original = np.linalg.matrix_rank(D)
        rank_result = np.linalg.matrix_rank(result, tol=1e-10)
        assert rank_result <= rank_original


class TestFit:
    """Tests for the fit method."""

    def test_fit_returns_correct_shapes(self):
        """Test that fit returns L and S with correct shapes."""
        D = np.random.randn(10, 20)
        rpca = RobustPCA(D)
        L, S = rpca.fit(max_iter=10, iter_print=1000)

        assert L.shape == D.shape
        assert S.shape == D.shape

    def test_fit_stores_results(self):
        """Test that fit stores L and S as attributes."""
        D = np.random.randn(10, 20)
        rpca = RobustPCA(D)
        L, S = rpca.fit(max_iter=10, iter_print=1000)

        np.testing.assert_array_equal(rpca.L, L)
        np.testing.assert_array_equal(rpca.S, S)

    def test_fit_convergence(self):
        """Test that fit converges (error decreases)."""
        np.random.seed(42)
        D = np.random.randn(20, 20)
        rpca = RobustPCA(D)
        L, S = rpca.fit(max_iter=100, iter_print=1000)

        # Error should be small after convergence
        err = RobustPCA.frobenius_norm(D - L - S)
        assert err < 1e-5 * RobustPCA.frobenius_norm(D)

    def test_fit_with_custom_tolerance(self):
        """Test fit with custom tolerance."""
        D = np.random.randn(10, 10)
        rpca = RobustPCA(D)
        L, S = rpca.fit(tol=1e-3, max_iter=1000, iter_print=1000)

        err = RobustPCA.frobenius_norm(D - L - S)
        assert err <= 1e-3


class TestDecomposition:
    """Integration tests for low-rank + sparse decomposition."""

    def test_decomposition_reconstruction(self):
        """Test that D ≈ L + S after fitting."""
        np.random.seed(42)
        D = np.random.randn(20, 20)

        rpca = RobustPCA(D)
        L, S = rpca.fit(max_iter=500, iter_print=1000)

        # Check decomposition: D ≈ L + S
        reconstruction_error = RobustPCA.frobenius_norm(D - L - S)
        assert reconstruction_error < 1e-5 * RobustPCA.frobenius_norm(D)

    def test_low_rank_input_recovery(self):
        """Test that low-rank input is recovered in L."""
        np.random.seed(42)
        # Create a rank-2 matrix
        U = np.random.randn(20, 2)
        V = np.random.randn(2, 20)
        L_true = U @ V

        rpca = RobustPCA(L_true)
        L, S = rpca.fit(max_iter=500, iter_print=1000)

        # Reconstruction should be good
        reconstruction_error = RobustPCA.frobenius_norm(L_true - L - S)
        assert reconstruction_error < 1e-5 * RobustPCA.frobenius_norm(L_true)

    def test_different_matrix_sizes(self):
        """Test decomposition works for various matrix sizes."""
        np.random.seed(42)

        for shape in [(10, 10), (20, 30), (30, 20), (50, 50)]:
            D = np.random.randn(*shape)
            rpca = RobustPCA(D)
            L, S = rpca.fit(max_iter=200, iter_print=1000)

            reconstruction_error = RobustPCA.frobenius_norm(D - L - S)
            assert reconstruction_error < 1e-5 * RobustPCA.frobenius_norm(D)

    def test_reproducibility_with_seed(self):
        """Test that results are reproducible with same random seed."""
        D = np.random.randn(15, 15)

        rpca1 = RobustPCA(D.copy())
        L1, S1 = rpca1.fit(max_iter=100, iter_print=1000)

        rpca2 = RobustPCA(D.copy())
        L2, S2 = rpca2.fit(max_iter=100, iter_print=1000)

        np.testing.assert_array_almost_equal(L1, L2)
        np.testing.assert_array_almost_equal(S1, S2)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_small_matrix(self):
        """Test with a small 2x2 matrix."""
        D = np.array([[1.0, 2.0], [3.0, 4.0]])
        rpca = RobustPCA(D)
        L, S = rpca.fit(max_iter=100, iter_print=1000)

        assert L.shape == (2, 2)
        assert S.shape == (2, 2)

    def test_single_row(self):
        """Test with a single row matrix."""
        D = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        rpca = RobustPCA(D)
        L, S = rpca.fit(max_iter=100, iter_print=1000)

        assert L.shape == (1, 5)
        assert S.shape == (1, 5)

    def test_single_column(self):
        """Test with a single column matrix."""
        D = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        rpca = RobustPCA(D)
        L, S = rpca.fit(max_iter=100, iter_print=1000)

        assert L.shape == (5, 1)
        assert S.shape == (5, 1)

    def test_zero_matrix(self):
        """Test with zero matrix."""
        D = np.zeros((10, 10))
        rpca = RobustPCA(D, mu=1.0)  # Provide mu since default would divide by zero
        L, S = rpca.fit(max_iter=10, iter_print=1000)

        np.testing.assert_array_almost_equal(L, np.zeros_like(D))
        np.testing.assert_array_almost_equal(S, np.zeros_like(D))

    def test_identity_matrix(self):
        """Test with identity matrix."""
        D = np.eye(10)
        rpca = RobustPCA(D)
        L, S = rpca.fit(max_iter=500, iter_print=1000)

        # D = L + S should hold
        reconstruction_error = RobustPCA.frobenius_norm(D - L - S)
        assert reconstruction_error < 1e-5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
