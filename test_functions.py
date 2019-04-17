import numpy as np

from functions import gauss


def test_gauss_value_at_mu():
    """
    The value of the Gaussian at ``x=mu`` should be $1/sqrt(2 * pi * sigma**2)$; check
    that that is the case for a couple non-zero values of sigma to within a tolerance.
    """
    # Nothing particularly special about this value of tolerance
    tolerance = 1e-8

    # choose x and mu; they need to be the same for this test
    x = 4
    mu = 4
    sig1 = 2.3
    norm1 = 1 / np.sqrt(2 * np.pi * sig1 ** 2)
    assert np.abs(gauss(x, mu, sig1) - norm1) <= tolerance


def test_gauss_x_mu_zero():
    """
    If $\mu=0$, $x=0$, and $\sigma=1$ the value of the Gaussian should
    be $1/\sqrt{2\pi}$. Check that it is.
    """
    tolerance = 1e-8
    x = 0
    mu = 0
    sigma = 1
    assert np.abs(gauss(x, mu, sigma) - 1 / np.sqrt(2 * np.pi)) <= tolerance


def test_gauss_x_zero_mu_is_2_sigma():
    """
    If $x=0$ and $\mu = \sqrt{2} \sigma$ then the Gaussian should have value
    $\frac{1}{e\sqrt{2\pi\sigma^2}}$. Make sure it does.
    """
    tolerance = 1e-8
    x = 0
    # Pick a value of sigma
    sigma = 2.5
    mu = np.sqrt(2) * sigma
    expected_result = 1 / (np.e * np.sqrt(2 * np.pi * sigma ** 2))
    assert np.abs(gauss(x, mu, sigma) - expected_result) <= tolerance
