"""Gemeinsame Fixtures: die Experimente einmal klein (für Struktur- und Verdikt-Tests) und einmal in voller Größe (nur für test_claims). Die Experimente 1 und 2 sind exakt und billig."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="session")
def single():
    import dfh_evaluation as ev

    return ev.single_error_experiment()


@pytest.fixture(scope="session")
def strength():
    import dfh_evaluation as ev

    return ev.strength_experiment()


@pytest.fixture(scope="session")
def small_detect():
    import dfh_evaluation as ev

    return ev.detection_experiment(reps=3, n=500, distance=1)


@pytest.fixture(scope="session")
def small_multi():
    import dfh_evaluation as ev

    return ev.multi_error_experiment(ks=(1, 2), samples=8), ev.rmse_experiment(ns=(200,), reps=5)


@pytest.fixture(scope="session")
def full_detect():
    import dfh_evaluation as ev

    return ev.detection_experiment()


@pytest.fixture(scope="session")
def full_multi():
    import dfh_evaluation as ev

    return ev.multi_error_experiment(), ev.rmse_experiment()
