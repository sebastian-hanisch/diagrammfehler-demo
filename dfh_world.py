"""Die Welt mit sechs Größen als lineares Gauß-Modell: Bevölkerungskovarianz, Koeffizient von X nach Anpassung (exakt), Daten aus festen Strömen und die zwei Fehlerformeln (fehlender Pfeil, verborgene Ursache). Kein Streamlit."""

from __future__ import annotations

import numpy as np

import dfh_constants as K
from kausal_generator import SplitMix64

N = 6


def coefficient_matrix(overrides: dict | None = None, extra: int = 0) -> np.ndarray:
    """B[j, i]: Koeffizient von i in j; overrides ersetzt Pfeilstärken; extra: zusätzliche Knoten am Ende (für eine verborgene Ursache)."""
    size = N + extra
    b = np.zeros((size, size))
    for (i, j), c in {**K.COEF, **(overrides or {})}.items():
        b[j, i] = c
    return b


def covariance(b: np.ndarray) -> np.ndarray:
    inv = np.linalg.inv(np.eye(b.shape[0]) - b)
    return inv @ inv.T


POPULATION = covariance(coefficient_matrix())


def x_coefficient(cov: np.ndarray, adjust=()) -> float:
    """Koeffizient von X in der Regression von Y auf X und die Menge (aus der Kovarianzmatrix, exakt)."""
    idx = [K.X] + sorted(adjust)
    return float(np.linalg.solve(cov[np.ix_(idx, idx)], cov[idx, K.Y])[0])


def bias(adjust=(), cov: np.ndarray | None = None) -> float:
    """Abweichung des Koeffizienten von X von der wahren Wirkung 0,6 (exakt, Bevölkerung)."""
    return x_coefficient(POPULATION if cov is None else cov, adjust) - K.EFFECT


def missing_arrow_bias(strength: float, adjust=()) -> float:
    """Bias, wenn der Pfeil Z → X in Wahrheit die Stärke `strength` hat (in der Welt 0,8) und man ohne ihn anpasst; mit leerer Menge 0,6 · Kov(X, Z) / Var(X)."""
    cov = covariance(coefficient_matrix({(K.Z, K.X): strength}))
    return x_coefficient(cov, adjust) - K.EFFECT


def hidden_cause_bias(a: float, b: float, adjust=(K.Z,)) -> float:
    """Bias bei einer verborgenen gemeinsamen Ursache U (Einheitsvarianz) von X (Stärke a) und Y (Stärke b); angepasst wird an die Menge (ohne U): nach Anpassung an Z gleich a b / (1 + a²)."""
    bm = coefficient_matrix({(N, K.X): a, (N, K.Y): b}, extra=1)
    cov = covariance(bm)[:N, :N]
    return x_coefficient(cov, adjust) - K.EFFECT


def simulate(n: int, rng: SplitMix64) -> np.ndarray:
    """n × 6 (S, Z, X, Y, W, C), Einheitsrauschen: x = (I − B)⁻¹ e."""
    e = rng.normal(n * N).reshape(n, N)
    return e @ np.linalg.inv(np.eye(N) - coefficient_matrix()).T


def regression_estimate(data: np.ndarray, adjust=()) -> tuple:
    """(Koeffizient von X, Standardfehler) der Regression von Y auf X und die Menge."""
    n = len(data)
    z = np.column_stack([np.ones(n), data[:, K.X]] + [data[:, j] for j in sorted(adjust)])
    beta, *_ = np.linalg.lstsq(z, data[:, K.Y], rcond=None)
    res = data[:, K.Y] - z @ beta
    cov = np.linalg.inv(z.T @ z) * (res @ res) / (n - z.shape[1])
    return float(beta[1]), float(np.sqrt(cov[1, 1]))
