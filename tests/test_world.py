"""Die Welt: Handwerte der exakten Bias-Rechnung, Konstanten und die Daten aus festen Strömen."""

import numpy as np
import pytest

import dfh_constants as K
import dfh_world as W
from kausal_generator import SplitMix64


def test_hand_values_of_the_exact_bias():
    var_z, cov_xz = 0.64 + 1, 0.8 * (0.64 + 1)
    var_x = 0.64 * var_z + 1
    assert var_z == pytest.approx(1.64) and var_x == pytest.approx(2.0496)
    assert W.POPULATION[K.Z, K.Z] == pytest.approx(var_z) and W.POPULATION[K.X, K.X] == pytest.approx(var_x) and W.POPULATION[K.X, K.Z] == pytest.approx(cov_xz)
    assert W.bias(()) == pytest.approx(0.6 * cov_xz / var_x, abs=1e-12) and W.bias(()) == pytest.approx(0.3841, abs=0.00006)
    assert W.bias((K.Z,)) == pytest.approx(0.0, abs=1e-12) and W.bias((K.S, K.Z, K.W)) == pytest.approx(0.0, abs=1e-12) and W.bias((K.Z, K.W)) == pytest.approx(0.0, abs=1e-12)
    # alle anderen: nach Anpassung an S, Z, W bleiben eX, eY, eC; C = 0,8 Y + eC; beta_x = 0,6 / 1,64
    assert W.x_coefficient(W.POPULATION, (K.S, K.Z, K.W, K.C_)) == pytest.approx(0.6 / 1.64, abs=1e-12) and W.bias((K.S, K.Z, K.W, K.C_)) == pytest.approx(-0.2341, abs=0.00006)


def test_missing_arrow_formula_and_the_hand_instance():
    for a in (0.05, 0.2, 0.5, 0.8):
        assert W.missing_arrow_bias(a) == pytest.approx(0.6 * 1.64 * a / (1.64 * a * a + 1), abs=1e-12)
    assert W.missing_arrow_bias(0.2) == pytest.approx(0.1847, abs=0.00006) and W.missing_arrow_bias(0.8) == pytest.approx(W.bias(()), abs=1e-12)
    assert W.missing_arrow_bias(0.0) == pytest.approx(0.0, abs=1e-12)


def test_hidden_cause_formula_and_the_hand_instances():
    for a, b in ((0.5, 0.5), (0.2, 0.3), (0.8, 0.1), (0.3, 0.8)):
        assert W.hidden_cause_bias(a, b) == pytest.approx(a * b / (1 + a * a), abs=1e-12)
    assert W.hidden_cause_bias(0.5, 0.5) == pytest.approx(0.2, abs=1e-12) and W.hidden_cause_bias(0.2, 0.3) == pytest.approx(0.0577, abs=0.00006)
    assert W.hidden_cause_bias(0.0, 0.5) == pytest.approx(0.0, abs=1e-12)


def test_coefficient_matrix_and_covariance_are_consistent():
    b = W.coefficient_matrix()
    assert b.shape == (6, 6) and np.count_nonzero(b) == 6 and b[K.X, K.Z] == 0.8 and b[K.Y, K.X] == 0.6 and b[K.C_, K.Y] == 0.8
    assert W.coefficient_matrix({(K.Z, K.X): 0.1})[K.X, K.Z] == 0.1 and W.coefficient_matrix(extra=1).shape == (7, 7)
    assert np.allclose(W.POPULATION, W.POPULATION.T) and np.all(np.linalg.eigvalsh(W.POPULATION) > 0)
    assert K.EFFECT == 0.6 and set(K.COEF) == set(K.TRUE_EDGES) and len(K.PAIRS) == 15 and len(K.NAMES) == len(K.LONG) == 6


def test_simulate_is_deterministic_and_has_the_population_covariance():
    d1, d2 = W.simulate(2000, SplitMix64(5)), W.simulate(2000, SplitMix64(5))
    assert d1.shape == (2000, 6) and np.array_equal(d1, d2) and not np.array_equal(d1, W.simulate(2000, SplitMix64(6)))
    big = W.simulate(300000, SplitMix64(11))
    assert np.allclose(np.cov(big, rowvar=False), W.POPULATION, atol=0.03)


def test_regression_estimate_by_hand():
    data = np.zeros((4, 6))
    data[:, K.X] = [1, 2, 3, 4]
    data[:, K.Y] = [2, 1, 4, 3]
    est, se = W.regression_estimate(data)
    assert est == pytest.approx(0.6) and se == pytest.approx(np.sqrt(1.6 / 5), abs=1e-9)             # Sxy = 3, Sxx = 5, SSR = 3,2, sigma² = 3,2 / 2
    data[:, K.Z] = [0, 0, 1, 1]
    est2, _ = W.regression_estimate(data, (K.Z,))
    z = np.column_stack([np.ones(4), data[:, K.X], data[:, K.Z]])
    assert est2 == pytest.approx(np.linalg.lstsq(z, data[:, K.Y], rcond=None)[0][1])
    exact = np.zeros((5, 6))
    exact[:, K.X] = [1, 2, 3, 4, 5]
    exact[:, K.Y] = 2 * exact[:, K.X]
    assert W.regression_estimate(exact) == pytest.approx((2.0, 0.0), abs=1e-9)
