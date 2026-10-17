"""Auswertung: Formatierung, Analyse eines Diagramms, die vier Experimente (Struktur und Invarianten)."""

import numpy as np
import pytest

import dfh_constants as K
import dfh_evaluation as ev
import dfh_graphs as D
import dfh_world as W
from kausal_generator import SplitMix64

S, Z, X, Y, WW, C = range(6)


def _a(**kw):
    return ev.analyse(ev.Settings(**kw))


def test_format_helpers():
    assert ev.de(1.5) == "1,50" and ev.de(-0.234, 3) == "−0,234" and ev.signed(0.384) == "+0,38" and ev.signed(-0.2341, 3) == "−0,234" and ev.signed(-1e-12, 3) == "+0,000"
    assert ev.pct(0.914) == "91 %" and ev.pct(0.0625, 1) == "6,2 %" and ev.names_of((4, 1)) == "{Z, W}" and ev.names_of(()) == "{}"
    assert ev.harmless(0.06, 0.06) and ev.harmless(-0.06, 0.06) and not ev.harmless(0.0601, 0.06)


def test_settings_default_and_hashable():
    s = ev.Settings()
    assert s.edges == K.TRUE_EDGES and s.rule == "Eltern" and s.tol == 0.06 and s.n == 1000 and s.seed == K.DEFAULT_SEED and hash(s) == hash(ev.Settings())


def test_analysis_of_the_true_diagram():
    a = _a()
    assert a.errors == 0 and a.chosen.rule == "Eltern" and a.chosen.adjust == (Z,) and a.chosen.valid_own and a.chosen.valid_true and a.chosen.harmless and a.chosen.bias == pytest.approx(0.0, abs=1e-12)
    assert [r.rule for r in a.rows] == list(K.RULES) and [r.adjust for r in a.rows] == [(Z,), (S, Z, WW), (Z,), (S, Z, WW, C), ()]
    assert [r.valid_true for r in a.rows] == [True, True, True, False, False] and [r.harmless for r in a.rows] == [True, True, True, False, False]
    assert [round(r.bias, 4) for r in a.rows] == [0.0, 0.0, 0.0, -0.2341, 0.3841] and a.reasons == []
    assert a.estimate == pytest.approx(0.6, abs=3 * a.se) and 0.03 < a.se < 0.06 and a.class_size == 3 and a.same_class and len(a.markov) == 11 and a.alpha_each == pytest.approx(0.05 / 11)
    assert not a.rejected and all(0 <= p <= 1 for _, _, _, p, _ in a.markov)


def test_analysis_is_deterministic_and_seeded():
    assert _a(seed=3).estimate == _a(seed=3).estimate and _a(seed=3).estimate != _a(seed=4).estimate
    assert _a(n=4000).se < _a(n=200).se


def test_rule_choice_selects_the_row():
    for rule in K.RULES:
        assert _a(rule=rule).chosen.rule == rule


def test_analysis_of_a_missing_arrow():
    a = _a(edges=K.TRUE_EDGES - {(Z, X)})
    assert a.errors == 1 and a.diff["fehlt"] == [(Z, X)] and a.chosen.adjust == () and not a.chosen.harmless and a.chosen.bias == pytest.approx(0.3841, abs=0.00006)
    assert a.chosen.valid_own and not a.chosen.valid_true and a.reasons == ["Offener Rückwärtspfad X ← Z → Y."] and a.rejected and not a.same_class and a.class_size == 2


def test_analysis_of_an_extra_arrow_is_harmless():
    a = _a(edges=K.TRUE_EDGES | {(WW, X)})
    assert a.errors == 1 and a.diff["zu viel"] == [(WW, X)] and a.chosen.adjust == (Z, WW) and a.chosen.valid_true and a.chosen.bias == pytest.approx(0.0, abs=1e-12)


def test_analysis_of_the_markov_neighbour_is_not_rejected_but_harmful():
    a = _a(edges=(K.TRUE_EDGES - {(Z, X), (S, Z)}) | {(X, Z), (Z, S)})
    assert a.errors == 2 and set(a.diff["umgedreht"]) == {(X, Z), (Z, S)} and a.same_class and not a.rejected and a.chosen.adjust == () and a.chosen.bias == pytest.approx(0.3841, abs=0.00006)
    assert a.markov and all(not rej for *_, rej in a.markov)


def test_reasons_name_descendants_and_open_paths():
    r = ev.reasons_invalid((Z, C))
    assert r == ["{C} ist im wahren Diagramm Nachkomme des Faktors (X → … → C)."]
    both = ev.reasons_invalid((C,))
    assert any("Nachkomme" in t for t in both) and any("Offener Rückwärtspfad" in t for t in both)
    assert ev.reasons_invalid((Z,)) == [] and ev.reasons_invalid((Z, WW)) == []
    assert ev.reasons_invalid(()) == ["Offener Rückwärtspfad X ← Z → Y."]


def test_a_cycle_is_refused():
    with pytest.raises(ValueError):
        _a(edges=K.TRUE_EDGES | {(Y, S)})


def test_markov_checks_reject_a_wrong_diagram_and_keep_the_true_one():
    corr = np.corrcoef(W.simulate(1000, SplitMix64(9)), rowvar=False)
    rows, alpha, rej = ev.markov_checks(K.TRUE_EDGES, corr, 1000)
    assert len(rows) == 11 and alpha == pytest.approx(0.05 / 11) and not rej
    rows2, _, rej2 = ev.markov_checks(set(), corr, 1000)
    assert len(rows2) == 15 and rej2 and any(r[4] for r in rows2)


def test_single_error_experiment_structure(single):
    assert len(single["rows"]) == 23 and set(single["true_bias"]) == set(K.RULES) and single["true_bias"]["Eltern"] == pytest.approx(0.0, abs=1e-12)
    for r in single["rows"]:
        assert set(r["bias"]) == set(K.RULES) and r["label"].startswith(r["kind"]) and set(r["valid_true"]) == set(K.RULES)
        assert all((abs(r["bias"][rule]) < 1e-9) == r["valid_true"][rule] for rule in K.RULES)


def test_single_summary_counts_and_monotone_in_the_tolerance(single):
    sm = ev.single_summary(single, 0.06)
    assert sm["Eltern"]["harmless"] == 21 and sm["kleinste gültige"]["harmless"] == 19 and sm["Nicht-Nachkommen"]["harmless"] == 18 and sm["alle anderen"]["harmless"] == 0 and sm["nichts"]["harmless"] == 0
    assert sm["Eltern"]["by_kind"] == {"fehlt": (5, 6), "umgedreht": (4, 5), "zu viel": (12, 12)} and sm["Eltern"]["total"] == 23 and sm["Eltern"]["share"] == pytest.approx(21 / 23)
    for rule in K.RULES:
        counts = [ev.single_summary(single, t)["%s" % rule]["harmless"] for t in (0.02, 0.1, 0.2, 0.3, 0.4)]
        assert counts == sorted(counts), rule
    assert ev.single_summary(single, 0.4)["nichts"]["harmless"] == 23 and ev.single_summary(single, 0.02)["Eltern"]["harmless"] == 21 and ev.single_summary(single, 0.2)["Nicht-Nachkommen"]["harmless"] == 18


def test_missing_arrow_threshold_is_where_the_bias_equals_the_tolerance():
    for tol in (0.02, 0.06, 0.1, 0.2):
        thr = ev.missing_arrow_threshold(tol)
        assert W.missing_arrow_bias(thr) == pytest.approx(tol, abs=1e-9) and 0 < thr < 0.5
    assert ev.missing_arrow_threshold(0.06) == pytest.approx(0.0614, abs=0.00006)
    assert ev.missing_arrow_threshold(0.02) < ev.missing_arrow_threshold(0.06) < ev.missing_arrow_threshold(0.2)


def test_strength_experiment_structure(strength):
    assert [r["a"] for r in strength["missing"]] == list(K.EXP_STRENGTHS) and len(strength["hidden"]) == len(K.EXP_HIDDEN) ** 2 and strength["true_strength"] == 0.8
    for h in strength["hidden"]:
        assert h["bias"] == pytest.approx(h["a"] * h["b"] / (1 + h["a"] ** 2), abs=1e-12)
    biases = [r["bias"] for r in strength["missing"]]
    assert biases[0] < biases[1] < biases[2] < biases[3]


def test_detection_experiment_structure(small_detect):
    res = small_detect
    assert res["reps"] == 3 and res["n"] == 500 and res["distance"] == 1 and len(res["rows"]) == 23
    for r in res["rows"]:
        assert r["dist"] == 1 and 0 <= r["rate"] <= 1 and set(r["bias"]) == set(K.RULES) and D.is_acyclic(r["edges"]) and r["edges"] != K.TRUE_EDGES
    sm = ev.detection_summary(res, 0.06, "Eltern")
    assert sm["total"] == 23 and sm["harmful"] + sm["harmless"] == 23 and sm["harmful"] == 2 and len(sm["missed"]) <= sm["harmful"]
    assert sum(sm["cnt"].values()) == 23 and sm["harmful_rejected"] + len(sm["missed"]) == sm["harmful"]


def test_detection_summary_depends_on_rule_and_tolerance(small_detect):
    assert ev.detection_summary(small_detect, 0.06, "nichts")["harmful"] == 23 and ev.detection_summary(small_detect, 0.5, "nichts")["harmful"] == 0
    assert ev.detection_summary(small_detect, 0.06, "Nicht-Nachkommen")["harmful"] == 5


def test_multi_error_experiment_structure(small_multi):
    res, rmse = small_multi
    assert res["ks"] == (1, 2) and res["samples"] == 8 and set(res["abs_bias"]) == {1, 2}
    for k in (1, 2):
        assert set(res["abs_bias"][k]) == set(K.RULES) and all(len(v) == 8 and np.all(v >= 0) for v in res["abs_bias"][k].values())
        assert np.allclose(res["abs_bias"][k]["alle anderen"], 0.2341, atol=6e-5) and np.allclose(res["abs_bias"][k]["nichts"], 0.3841, atol=6e-5)      # die Menge dieser Regeln hängt nicht vom Diagramm ab
    sm = ev.multi_summary(res, 0.06)
    assert all(0 <= v <= 1 for k in sm.values() for v in k.values()) and all(v == 1.0 for k in ev.multi_summary(res, 0.6).values() for v in k.values())
    assert rmse["ns"] == (200,) and rmse["reps"] == 5 and set(rmse["cases"]) == {(label, 200) for label in ev.RMSE_CASES}
    assert rmse["cases"][("richtiges Diagramm", 200)]["set"] == (Z,) and rmse["cases"][("zu viel W → X", 200)]["set"] == (Z, WW) and rmse["cases"][("fehlt Z → X", 200)]["set"] == ()


def test_multi_error_experiment_is_deterministic():
    a = ev.multi_error_experiment(ks=(2,), samples=6)
    b = ev.multi_error_experiment(ks=(2,), samples=6)
    assert all(np.array_equal(a["abs_bias"][2][r], b["abs_bias"][2][r]) for r in K.RULES)


def test_signed_zeroes_only_values_that_would_print_as_zero():
    assert ev.signed(-4e-7, 3) == "+0,000" and ev.signed(-6e-7, 3) == "−0,000" and ev.signed(6e-7, 3) == "+0,000"
    assert ev.signed(-(0.5 * 10 ** -3 * 1e-3), 3) == "−0,000" and ev.signed(-4e-6, 2) == "+0,00" and ev.signed(-4e-4, 2) == "−0,00" and ev.signed(-0.004, 2) == "−0,00" and ev.signed(-0.0004, 3) == "−0,000"


def test_detection_rates_are_fractions_of_the_number_of_datasets():
    res = ev.detection_experiment(reps=4, n=150, distance=1)
    rates = [r["rate"] for r in res["rows"]]
    assert all(abs(x * 4 - round(x * 4)) < 1e-9 for x in rates) and any(0 < x < 1 for x in rates) and res["reps"] == 4


def test_detection_summary_counts_a_rate_of_exactly_one_half_as_rejected():
    biases = {rule: 0.384 for rule in K.RULES}
    res = {"rows": [{"edges": frozenset({(0, 1)}), "dist": 1, "rate": 0.5, "bias": biases}, {"edges": frozenset({(0, 2)}), "dist": 1, "rate": 0.49, "bias": biases},
                    {"edges": frozenset({(0, 3)}), "dist": 2, "rate": 0.0, "bias": {rule: 0.06 for rule in K.RULES}}]}
    sm = ev.detection_summary(res, 0.06, "Eltern")
    assert sm["harmful"] == 2 and sm["harmful_rejected"] == 1 and len(sm["missed"]) == 1 and sm["missed"][0]["rate"] == 0.49 and sm["harmless"] == 1 and sm["harmless_rejected"] == 0


def test_multi_summary_counts_a_bias_exactly_at_the_tolerance_as_harmless():
    res = {"ks": (1,), "abs_bias": {1: {rule: np.array([0.06, 0.07]) for rule in K.RULES}}}
    assert ev.multi_summary(res, 0.06)[1]["Eltern"] == 0.5 and ev.multi_summary(res, 0.07)[1]["Eltern"] == 1.0 and ev.multi_summary(res, 0.05)[1]["Eltern"] == 0.0


def test_rmse_uses_every_dataset():
    res = ev.rmse_experiment(ns=(100,), reps=2)
    assert res["cases"][("richtiges Diagramm", 100)]["sd"] > 0
    res3 = ev.rmse_experiment(ns=(100, 200), reps=3)
    assert set(n for _, n in res3["cases"]) == {100, 200}
