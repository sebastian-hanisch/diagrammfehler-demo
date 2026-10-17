"""Jede Zahl aus README.md und den App-Texten wird hier über dieselben Auswertungsfunktionen nachgerechnet (DEMO-PLAYBOOK §4).

Die Experimente 1 und 2 sind exakt (Kovarianz der Welt, keine Stichprobe): Dort gelten Toleranzen von der Rundung. Experimente 3 und 4 laufen über SplitMix64-Ströme (ganzzahlig, plattformstabil), die Tests und Schätzungen
rechnen aber in Gleitkomma; wo ein einzelner Datensatz oder eine einzelne Aussage je Plattform kippen kann (Anteile verworfener Diagramme, Anteile harmloser Zufallsdiagramme, Streuung), gelten Toleranzen von zwei Läufen.
Was Identität oder Konstruktion ist, steht ausdrücklich dabei: gültig im wahren Diagramm ⇔ Bias null (Identität dieser Welt, kein Befund), „alle anderen“ und „nichts“ (die Menge hängt nicht vom Diagramm ab, Konstruktion),
„zusätzliche Pfeile schaden nie“ (Eigenschaft dieser Welt: kein Sammelknoten vor dem Faktor, kein Befund über Diagramme im Allgemeinen).
"""

import itertools

import numpy as np
import pytest

import dfh_constants as K
import dfh_evaluation as ev
import dfh_graphs as D
import dfh_world as W
import dsep_graph as G
import stl_pc as P

S, Z, X, Y, WW, C = range(6)
SETS = [sub for r in range(5) for sub in itertools.combinations((S, Z, WW, C), r)]


def test_claim_hand_instances():
    assert W.bias(()) == pytest.approx(0.3841, abs=0.00006) and W.missing_arrow_bias(0.2) == pytest.approx(0.1847, abs=0.00006) and W.hidden_cause_bias(0.5, 0.5) == pytest.approx(0.2, abs=1e-9)
    assert W.hidden_cause_bias(0.2, 0.3) == pytest.approx(0.0577, abs=0.00006) and W.bias((S, Z, WW, C)) == pytest.approx(-0.2341, abs=0.00006) and len(P.class_members(D.parents_of(K.TRUE_EDGES))) == 3


def test_claim_valid_in_the_true_diagram_iff_bias_zero_is_an_identity_of_this_world():
    assert len(SETS) == 16 and all(D.valid_backdoor(K.TRUE_EDGES, s) == (abs(W.bias(s)) < 1e-9) for s in SETS)
    assert [s for s in SETS if D.valid_backdoor(K.TRUE_EDGES, s)] == [(Z,), (S, Z), (Z, WW), (S, Z, WW)]


def test_claim_the_true_diagram_with_each_rule(single):
    tb = single["true_bias"]
    assert tb["Eltern"] == pytest.approx(0, abs=1e-9) and tb["Nicht-Nachkommen"] == pytest.approx(0, abs=1e-9) and tb["kleinste gültige"] == pytest.approx(0, abs=1e-9)
    assert tb["alle anderen"] == pytest.approx(-0.2341, abs=0.00006) and tb["nichts"] == pytest.approx(0.3841, abs=0.00006)


def test_claim_single_errors_exact(single):
    sm = ev.single_summary(single, K.DEFAULT_TOL)
    assert sm["Eltern"]["total"] == 23 and {k: v[1] for k, v in sm["Eltern"]["by_kind"].items()} == {"fehlt": 6, "umgedreht": 5, "zu viel": 12}
    assert (sm["Eltern"]["harmless"], sm["kleinste gültige"]["harmless"], sm["Nicht-Nachkommen"]["harmless"], sm["alle anderen"]["harmless"], sm["nichts"]["harmless"]) == (21, 19, 18, 0, 0)
    assert sm["Eltern"]["by_kind"] == {"fehlt": (5, 6), "umgedreht": (4, 5), "zu viel": (12, 12)}
    harmful = [r["label"] for r in single["rows"] if abs(r["bias"]["Eltern"]) > K.DEFAULT_TOL]
    assert harmful == ["fehlt Z → X", "umgedreht Z → X"] and all(abs(r["bias"]["Eltern"]) == pytest.approx(0.3841, abs=0.00006) for r in single["rows"] if r["label"] in harmful)


def test_claim_the_biases_cluster_so_the_tolerance_matters_little_for_single_errors(single):
    values = sorted({round(abs(r["bias"][rule]), 3) for r in single["rows"] for rule in K.RULES})
    assert values == [0.0, 0.234, 0.293, 0.384] and [ev.single_summary(single, t)["Eltern"]["harmless"] for t in (0.02, 0.06, 0.2)] == [21, 21, 21]
    assert [ev.single_summary(single, t)["Nicht-Nachkommen"]["harmless"] for t in (0.02, 0.2)] == [18, 18] and ev.single_summary(single, 0.3)["Nicht-Nachkommen"]["harmless"] == 23


def test_claim_minimal_and_non_descendants_fail_where_eltern_do_not(single):
    by = {r["label"]: r["bias"] for r in single["rows"]}
    assert abs(by["fehlt Z → Y"]["kleinste gültige"]) == pytest.approx(0.3841, abs=0.00006) and abs(by["fehlt Z → Y"]["Eltern"]) < 1e-9
    assert abs(by["umgedreht X → Y"]["kleinste gültige"]) == pytest.approx(0.3841, abs=0.00006) and abs(by["umgedreht X → Y"]["Eltern"]) < 1e-9
    for label in ("fehlt X → Y", "umgedreht X → Y", "fehlt Y → C", "umgedreht Y → C"):
        assert by[label]["Nicht-Nachkommen"] == pytest.approx(-0.2341, abs=0.00006), label
    assert by["umgedreht Z → X"]["Nicht-Nachkommen"] == pytest.approx(12 / 41, abs=1e-9)


def test_claim_extra_arrows_are_harmless_in_this_world_only(single):
    zu = [r for r in single["rows"] if r["kind"] == "zu viel"]
    assert len(zu) == 12 and all(r["valid_true"]["Eltern"] and abs(r["bias"]["Eltern"]) < 1e-9 for r in zu)
    # Grund: Ein zusätzlicher Pfeil in den Faktor kann nur von S oder W kommen (Y → X und C → X schlössen einen Kreis); beide sind Nicht-Nachkommen des Faktors, die Menge bleibt eine gültige Rückwärtsmenge.
    assert all(set(r["set"]["Eltern"]) <= {S, Z, WW} and Z in r["set"]["Eltern"] for r in zu)
    assert {r["edge"] for r in zu if r["edge"][1] == X} == {(S, X), (WW, X)}


def test_claim_markov_neighbour_has_the_same_statements_and_a_harmful_member():
    members = P.class_members(D.parents_of(K.TRUE_EDGES))
    biases = sorted(round(abs(W.bias(D.rule_set(frozenset((p, c) for c, ps in enumerate(m) for p in ps), "Eltern"))), 3) for m in members)
    assert biases == [0.0, 0.0, 0.384]
    neighbour = (K.TRUE_EDGES - {(Z, X), (S, Z)}) | {(X, Z), (Z, S)}
    assert D.parents_of(neighbour) in members and ev.analyse(ev.Settings(edges=neighbour)).chosen.bias == pytest.approx(0.3841, abs=0.00006)
    assert G.statement_set(D.parents_of(neighbour)) == G.statement_set(D.parents_of(K.TRUE_EDGES))


def test_claim_tolerance_thresholds_for_a_missing_arrow_and_a_hidden_cause(strength):
    assert ev.missing_arrow_threshold(0.06) == pytest.approx(0.0614, abs=0.0001) and ev.missing_arrow_threshold(0.02) == pytest.approx(0.0203, abs=0.0001) and ev.missing_arrow_threshold(0.2) == pytest.approx(0.2193, abs=0.0001)
    ok = [(h["a"], h["b"]) for h in strength["hidden"] if abs(h["bias"]) <= 0.06]
    assert len(ok) == 11 and (0.1, 0.1) in ok and (0.5, 0.5) not in ok and (0.8, 0.1) in ok and (0.3, 0.3) not in ok
    biases = {r["a"]: r["bias"] for r in strength["missing"]}
    assert biases[0.05] <= 0.06 < biases[0.1] and biases[0.8] == pytest.approx(0.3841, abs=0.00006) and max(biases.values()) == biases[0.8]


def test_claim_detection(full_detect):
    sm = ev.detection_summary(full_detect, K.DEFAULT_TOL, "Eltern")
    assert sm["total"] == 284 and sm["harmful"] == 51 and sm["harmless"] == 233                                  # Zählung der Diagramme: exakt
    assert 46 <= sm["harmful_rejected"] <= 50 and 2 <= len(sm["missed"]) <= 5 and all(r["rate"] < 0.2 for r in sm["missed"])
    assert sm["harmless_rejected"] / sm["harmless"] == pytest.approx(144 / 233, abs=0.08)
    single_rows = [r for r in full_detect["rows"] if r["dist"] == 1]
    harmful_single = [r for r in single_rows if abs(r["bias"]["Eltern"]) > K.DEFAULT_TOL]
    assert len(harmful_single) == 2 and all(r["rate"] >= 0.95 for r in harmful_single)
    zu = [r for r in single_rows if len(r["edges"]) == 7]
    assert len(zu) == 12 and max(r["rate"] for r in zu) <= 0.1                                                  # Fehlalarm bei zusätzlichem Pfeil
    assert sum(1 for r in single_rows if r["rate"] >= K.EXP_DETECT_RATE) == pytest.approx(10, abs=2)           # 2 schädliche + 8 harmlose werden trotzdem verworfen


def test_claim_the_data_reject_the_diagrams_that_claim_a_false_independence(full_detect):
    truth = G.statement_set(D.parents_of(K.TRUE_EDGES))
    rows = full_detect["rows"]
    true_only = [r for r in rows if G.statement_set(D.parents_of(r["edges"])) <= truth]          # behaupten nur wahre Unabhängigkeiten
    false_claim = [r for r in rows if r not in true_only]
    assert len(true_only) == 92 and len(false_claim) == 192
    assert max(r["rate"] for r in true_only) <= 0.1 and min(r["rate"] for r in false_claim) >= 0.95
    sm = ev.detection_summary(full_detect, K.DEFAULT_TOL, "Eltern")
    assert all(G.statement_set(D.parents_of(r["edges"])) <= truth for r in sm["missed"]) and sum(1 for r in true_only if abs(r["bias"]["Eltern"]) > K.DEFAULT_TOL) == 3


def test_claim_detection_of_other_rules(full_detect):
    nn = ev.detection_summary(full_detect, K.DEFAULT_TOL, "Nicht-Nachkommen")
    mn = ev.detection_summary(full_detect, K.DEFAULT_TOL, "kleinste gültige")
    assert nn["harmful"] == 111 and mn["harmful"] == 88 and 2 <= len(nn["missed"]) <= 6 and 2 <= len(mn["missed"]) <= 6


def test_claim_multiple_errors(full_multi):
    res, _ = full_multi
    sm = ev.multi_summary(res, K.DEFAULT_TOL)
    assert [sm[k]["Eltern"] for k in (1, 2, 3, 4)] == pytest.approx([0.873, 0.833, 0.787, 0.68], abs=0.06)
    assert [sm[k]["Nicht-Nachkommen"] for k in (1, 2, 3, 4)] == pytest.approx([0.717, 0.637, 0.543, 0.44], abs=0.06)
    assert [sm[k]["kleinste gültige"] for k in (1, 2, 3, 4)] == pytest.approx([0.777, 0.707, 0.63, 0.537], abs=0.06)
    assert all(sm[k]["alle anderen"] == 0.0 and sm[k]["nichts"] == 0.0 for k in sm)                              # Konstruktion: die Menge hängt nicht vom Diagramm ab
    assert sm[1]["Eltern"] > sm[4]["Eltern"] and sm[1]["Nicht-Nachkommen"] > sm[4]["Nicht-Nachkommen"] and all(sm[k]["Eltern"] > sm[k]["kleinste gültige"] > sm[k]["Nicht-Nachkommen"] for k in sm)


def test_claim_variance_against_bias(full_multi):
    _, rmse = full_multi
    c = rmse["cases"]
    assert c[("richtiges Diagramm", 1000)]["rmse"] == pytest.approx(0.037, abs=0.006) and c[("richtiges Diagramm", 200)]["rmse"] == pytest.approx(0.088, abs=0.015)
    assert c[("fehlt Z → X", 1000)]["rmse"] == pytest.approx(0.384, abs=0.01) and c[("fehlt Z → X", 1000)]["mean"] == pytest.approx(0.983, abs=0.01)
    assert c[("zu viel W → X", 1000)]["rmse"] < c[("richtiges Diagramm", 1000)]["rmse"] and c[("zu viel W → X", 1000)]["rmse"] / c[("richtiges Diagramm", 1000)]["rmse"] == pytest.approx(1 / np.sqrt(1.36), abs=0.06)
    assert c[("zu viel S → X", 1000)]["rmse"] == pytest.approx(c[("richtiges Diagramm", 1000)]["rmse"], abs=0.006)
    assert all(abs(v["mean"] - (K.EFFECT + (0.384 if v["set"] == () else 0))) < 0.02 for v in c.values())
