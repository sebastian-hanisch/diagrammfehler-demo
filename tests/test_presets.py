"""Presets, Reglergrenzen, Permalink-Raster und Konstanten."""

import pytest

import dfh_constants as K
import dfh_evaluation as ev
import dfh_graphs as D
import dfh_presets as P

S, Z, X, Y, W, C = range(6)
FIRST = "Das richtige Diagramm (Voreinstellung)"


def _settings(preset):
    return ev.Settings(**preset)


def test_every_preset_has_help_and_all_keys():
    assert set(K.PRESETS) == set(K.PRESET_HELP)
    for name, p in K.PRESETS.items():
        assert set(p) == set(P.PRESET_KEYS) | {"edges"}, name
        assert K.PRESET_HELP[name] and isinstance(p["edges"], frozenset)


def test_preset_values_are_valid_and_on_the_slider_grid():
    for name, p in K.PRESETS.items():
        assert D.is_acyclic(p["edges"]), name
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            v = p[key]
            if spec.choices:
                assert v in spec.choices, (name, key)
                continue
            assert spec.lo <= v <= spec.hi, (name, key)
            assert P.snap(v, spec) == pytest.approx(v), (name, key)


def test_first_preset_is_the_default_setting():
    p = K.PRESETS[FIRST]
    assert _settings(p) == ev.Settings() and next(iter(K.PRESETS)) == FIRST and p["edges"] == K.TRUE_EDGES
    assert all(P.SETTING_SPECS[sk].default == p[k] for k, sk in P.PRESET_KEYS.items())


def test_presets_differ_and_each_changes_at_most_the_diagram_or_the_rule():
    assert len({(tuple(sorted(p["edges"])), p["rule"]) for p in K.PRESETS.values()}) == len(K.PRESETS)
    base = K.PRESETS[FIRST]
    for name, p in K.PRESETS.items():
        changed = [k for k in p if p[k] != base[k]]
        assert set(changed) <= {"edges", "rule"}, name


def test_the_presets_show_what_their_help_says():
    a = {name: ev.analyse(_settings(p)) for name, p in K.PRESETS.items()}
    base = a[FIRST]
    assert base.errors == 0 and base.chosen.harmless and [round(r.bias, 4) for r in base.rows] == [0.0, 0.0, 0.0, -0.2341, 0.3841]
    miss = a["Pfeil Nachfrage → Faktor fehlt"]
    assert miss.chosen.adjust == () and round(miss.chosen.bias, 2) == 0.38 and miss.rejected
    rev = a["Pfeil Nachfrage → Faktor umgedreht"]
    assert rev.diff["umgedreht"] == [(X, Z)] and round(rev.chosen.bias, 2) == 0.38
    extra = a["Pfeil Wetter → Faktor zu viel"]
    assert extra.chosen.adjust == (Z, W) and extra.chosen.harmless and extra.chosen.valid_true and extra.se < base.se * 1.05
    markov = a["Markov-Nachbar (Kette umgedreht)"]
    assert markov.same_class and not markov.rejected and round(markov.chosen.bias, 2) == 0.38
    nn = a["Pfeil Faktor → Servicegrad fehlt (Nicht-Nachkommen)"]
    assert nn.chosen.rule == "Nicht-Nachkommen" and nn.chosen.adjust == (S, Z, W, C) and round(nn.chosen.bias, 2) == -0.23
    everything = a["Alles anpassen"]
    assert everything.errors == 0 and everything.chosen.adjust == (S, Z, W, C) and round(everything.chosen.bias, 2) == -0.23


def test_snap_clamps_rounds_and_maps_the_choices():
    tol = P.SETTING_SPECS["tol_slider"]
    assert P.snap(9.0, tol) == K.TOL_MAX and P.snap(-1.0, tol) == K.TOL_MIN and P.snap(0.07, tol) == pytest.approx(0.06) and P.snap(0.075, tol) == pytest.approx(0.08)
    n = P.SETTING_SPECS["n_slider"]
    assert P.snap(50, n) == K.N_MIN and P.snap(10 ** 6, n) == K.N_MAX and P.snap(1049, n) == 1000 and P.snap(1051, n) == 1100
    r = P.SETTING_SPECS["rule_radio"]
    assert P.snap(0, r) == "Eltern" and P.snap(4, r) == "nichts" and P.snap(9, r) == "nichts" and P.snap(-3, r) == "Eltern" and P.url_value("rule_radio", "kleinste gültige") == 2 and P.url_value("tol_slider", 0.1) == 0.1


def test_bounds_defaults_and_params():
    assert P.bounds("n_slider") == (K.N_MIN, K.N_MAX) and P.bounds("seed_input") == (0, K.SEED_MAX) and P.bounds("tol_slider") == (K.TOL_MIN, K.TOL_MAX) and P.bounds("rule_radio") == (0, 4)
    assert {s.url_param for s in P.SETTING_SPECS.values()} == {"r", "tol", "n", "seed"} and P.DIAGRAM_PARAM == "g" and P.DIAGRAM_PARAM not in {s.url_param for s in P.SETTING_SPECS.values()}
    assert K.DEFAULT_N in range(K.N_MIN, K.N_MAX + 1, K.N_STEP) and P.snap(K.DEFAULT_TOL, P.SETTING_SPECS["tol_slider"]) == pytest.approx(K.DEFAULT_TOL) and K.DEFAULT_RULE == K.RULES[0]


def test_pair_keys_and_error_labels():
    assert list(P.PAIR_KEYS) == list(K.PAIRS) and len(set(P.PAIR_KEYS.values())) == 15 and P.PAIR_KEYS[(0, 1)] == "pair_0_1"
    assert len(P.ERROR_LABELS) == 23 and P.NO_ERROR not in P.ERROR_LABELS and P.ERROR_LABELS["fehlt Z → X"] == K.TRUE_EDGES - {(Z, X)} and P.ERROR_LABELS["zu viel W → X"] == K.TRUE_EDGES | {(W, X)}
    assert P.ERROR_LABELS["umgedreht Z → X"] == (K.TRUE_EDGES - {(Z, X)}) | {(X, Z)}


def test_experiment_constants_are_consistent():
    assert K.EXP_STRENGTHS == tuple(sorted(K.EXP_STRENGTHS)) and K.EXP_STRENGTHS[-1] == K.COEF[(Z, X)] and K.EXP_HIDDEN == tuple(sorted(K.EXP_HIDDEN))
    assert K.EXP_DETECT_REPS >= 30 and K.EXP_DETECT_DISTANCE == 2 and K.EXP_MULTI_KS == (1, 2, 3, 4) and K.EXP_MULTI_SAMPLES >= 100 and K.EXP_RMSE_NS == tuple(sorted(K.EXP_RMSE_NS)) and K.EXP_RMSE_REPS >= 100
    assert 0 < K.EXP_DETECT_RATE <= 1 and K.SEED_BASE > 10 ** 6 and 0 < K.TOL_MIN < K.DEFAULT_TOL < K.TOL_MAX and K.DEFAULT_TOL == pytest.approx(0.1 * K.EFFECT)
    assert set(K.RULE_HELP) == set(K.RULES)
