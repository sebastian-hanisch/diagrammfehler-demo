"""Dauerhafte Orakel mit unabhängiger zweiter Rechnung: (1) der exakte Bias aus der Kovarianz stimmt mit einer großen Simulation überein; (2) eine Menge ist im wahren Diagramm gültig genau dann, wenn der Bias null ist
(Rechnung über Kovarianz gegen d-Separation: zwei verschiedene Wege zur selben Aussage), für alle 16 Mengen und für alle Mengen aller Regeln in allen 285 Diagrammen im Abstand 2; (3) die Zahl der azyklischen
Diagramme auf n benannten Knoten (1, 3, 25, 543, 29 281); (4) der Fisher-z-Test stimmt mit scipy überein."""

import itertools
import math

import numpy as np
import pytest
import scipy.stats as st

import dfh_constants as K
import dfh_evaluation as ev
import dfh_graphs as D
import dfh_world as W
import stl_pc as P
from kausal_generator import SplitMix64

OTHERS = [v for v in range(6) if v not in (K.X, K.Y)]


def test_exact_bias_agrees_with_a_large_simulation_for_every_set():
    data = W.simulate(400000, SplitMix64(77))
    for r in range(len(OTHERS) + 1):
        for sub in itertools.combinations(OTHERS, r):
            est, _ = W.regression_estimate(data, sub)
            assert est - K.EFFECT == pytest.approx(W.bias(sub), abs=0.012), sub


def test_valid_in_the_true_diagram_iff_the_bias_is_zero_for_all_16_sets():
    for r in range(len(OTHERS) + 1):
        for sub in itertools.combinations(OTHERS, r):
            assert D.valid_backdoor(K.TRUE_EDGES, sub) == (abs(W.bias(sub)) < 1e-9), sub


def test_valid_iff_bias_zero_for_every_rule_in_every_diagram_within_distance_two():
    seen = D.diagrams_within(2)
    assert len(seen) == 285
    mismatches = [(sorted(edges), rule) for edges in seen for rule in K.RULES
                  if D.valid_backdoor(K.TRUE_EDGES, D.rule_set(edges, rule)) != (abs(W.bias(D.rule_set(edges, rule))) < 1e-9)]
    assert mismatches == []


def test_counts_of_labelled_dags_by_enumeration():
    assert [D.count_dags(n) for n in (1, 2, 3, 4)] == [1, 3, 25, 543]
    assert D.count_dags(5) == 29281


def test_fisher_z_p_value_agrees_with_scipy():
    for r, n, k in ((0.1, 200, 0), (0.35, 1000, 2), (-0.05, 5000, 1), (0.9, 50, 3)):
        z = math.atanh(r) * math.sqrt(n - k - 3)
        assert P.fisher_z_pvalue(r, n, k) == pytest.approx(2 * st.norm.sf(abs(z)), rel=1e-9)


def test_the_bias_of_a_rule_in_any_diagram_is_the_bias_of_its_set():
    """Der Bias hängt nur von der Menge ab, nicht vom Diagramm, das sie gewählt hat."""
    for edges in list(D.diagrams_within(1))[:12]:
        for rule in K.RULES:
            adj = D.rule_set(edges, rule)
            assert W.bias(adj) == pytest.approx(W.x_coefficient(W.POPULATION, sorted(adj)) - 0.6, abs=1e-12)
    assert np.isclose(W.bias((K.Z,)), 0.0)


def test_the_analysis_rows_agree_with_the_simulation_for_wrong_diagrams():
    """Die Zeilen der Analyse (exakter Bias je Regel in einem falschen Diagramm) stimmen mit der Schätzung aus einer großen Stichprobe überein."""
    data = W.simulate(400000, SplitMix64(78))
    for edges in (K.TRUE_EDGES - {(K.Z, K.X)}, K.TRUE_EDGES | {(K.W, K.X)}, (K.TRUE_EDGES - {(K.X, K.Y)}) | {(K.Y, K.X)}, (K.TRUE_EDGES - {(K.Z, K.X), (K.S, K.Z)}) | {(K.X, K.Z), (K.Z, K.S)}):
        a = ev.analyse(ev.Settings(edges=edges, n=100))
        for row in a.rows:
            est, _ = W.regression_estimate(data, row.adjust)
            assert est - K.EFFECT == pytest.approx(row.bias, abs=0.012), (sorted(edges), row.rule)
            assert row.valid_true == (abs(est - K.EFFECT) < 0.012 and abs(row.bias) < 1e-9)


def test_the_experiment_tables_agree_with_the_simulation(single):
    data = W.simulate(400000, SplitMix64(79))
    for r in single["rows"][::4]:
        for rule in K.RULES:
            est, _ = W.regression_estimate(data, r["set"][rule])
            assert est - K.EFFECT == pytest.approx(r["bias"][rule], abs=0.012), (r["label"], rule)
