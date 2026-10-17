"""End-to-end Smoke-Test via Streamlits AppTest: lädt app.py, klickt jeden Preset-Button, fährt die Regler an die Grenzen und prüft, dass kein Python-Fehler auftritt."""

import os

from streamlit.testing.v1 import AppTest

import dfh_constants as K
import dfh_graphs as D
import dfh_presets as P

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = 300
SLIDERS = ("tol_slider", "n_slider")
S, Z, X, Y, W, C = range(6)


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _metrics(at):
    return {m.label: m.value for m in at.metric}


def _diagram(at):
    return D.edges_from_states({pair: at.selectbox(key=key).value for pair, key in P.PAIR_KEYS.items()})


def test_app_loads_without_exception_and_shows_the_numbers():
    at = _fresh_app()
    m = _metrics(at)
    assert m["Fehler im Diagramm"] == "0" and m["Menge nach der Regel"] == "{Z}" and m["Bias (exakt)"] == "+0,000" and m["Schätzung aus den Daten"] == "0,58"
    assert at.radio(key="rule_radio").value == K.DEFAULT_RULE and at.slider(key="tol_slider").value == K.DEFAULT_TOL and at.slider(key="n_slider").value == K.DEFAULT_N and at.number_input(key="seed_input").value == K.DEFAULT_SEED
    assert _diagram(at) == K.TRUE_EDGES and any("Das gezeichnete Diagramm ist das wahre" in s.value for s in at.success)


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if label.startswith("▶"):
            continue                                                         # die Experimente laufen in eigenen Tests
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_the_sliders_at_min_and_max_do_not_raise():
    for key in SLIDERS:
        for edge in ("min", "max"):
            at = _fresh_app()
            s = at.slider(key=key)
            s.set_value(s.min if edge == "min" else s.max).run(timeout=TIMEOUT)
            assert not at.exception, (key, edge, [str(e) for e in at.exception])


def test_every_preset_is_applied_to_the_widgets():
    labels = {b.label: i for i, b in enumerate(_fresh_app().button)}
    for name, preset in K.PRESETS.items():
        at = _fresh_app()
        at.button[labels[name]].click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])
        assert at.radio(key="rule_radio").value == preset["rule"] and at.slider(key="tol_slider").value == preset["tol"] and at.slider(key="n_slider").value == preset["n"] and at.number_input(key="seed_input").value == preset["seed"]
        assert _diagram(at) == preset["edges"], name


def test_missing_arrow_preset_shows_the_error_and_the_numbers():
    at = _fresh_app()
    at.button[[b.label for b in at.button].index("Pfeil Nachfrage → Faktor fehlt")].click().run(timeout=TIMEOUT)
    m = _metrics(at)
    assert not at.exception and m["Fehler im Diagramm"] == "1" and m["Menge nach der Regel"] == "{}" and m["Bias (exakt)"] == "+0,384" and any("Offener Rückwärtspfad X ← Z → Y" in e.value for e in at.error)


def test_pair_selectboxes_change_the_diagram_and_a_cycle_stops_the_analysis():
    at = _fresh_app()
    at.selectbox(key=P.PAIR_KEYS[(Z, X)]).set_value(0).run(timeout=TIMEOUT)
    assert not at.exception and _metrics(at)["Fehler im Diagramm"] == "1" and _metrics(at)["Bias (exakt)"] == "+0,384"
    at = _fresh_app()
    at.selectbox(key=P.PAIR_KEYS[(S, Y)]).set_value(2).run(timeout=TIMEOUT)          # Y → S schließt den Kreis S → Z → X → Y → S
    assert not at.exception and any("enthält einen Kreis" in e.value for e in at.error) and not at.metric
    at.selectbox(key=P.PAIR_KEYS[(S, Y)]).set_value(0).run(timeout=TIMEOUT)
    assert not at.exception and not any("enthält einen Kreis" in e.value for e in at.error) and _metrics(at)["Fehler im Diagramm"] == "0"


def test_the_single_error_selection_sets_the_diagram_and_resets_itself():
    at = _fresh_app()
    at.selectbox(key="error_select").set_value("zu viel W → X").run(timeout=TIMEOUT)
    assert not at.exception and _diagram(at) == K.TRUE_EDGES | {(W, X)} and at.selectbox(key="error_select").value == P.NO_ERROR and _metrics(at)["Menge nach der Regel"] == "{Z, W}"


def test_choice_widgets_work():
    at = _fresh_app()
    at.radio(key="rule_radio").set_value("alle anderen").run(timeout=TIMEOUT)
    assert not at.exception and at.radio(key="rule_radio").value == "alle anderen" and _metrics(at)["Menge nach der Regel"] == "{S, Z, W, C}" and _metrics(at)["Bias (exakt)"] == "−0,234"
    at.slider(key="tol_slider").set_value(0.2).run(timeout=TIMEOUT)
    assert not at.exception and at.slider(key="tol_slider").value == 0.2


def test_every_experiment_runs_on_click():
    at = _fresh_app()
    assert "exp3_ready" not in at.session_state
    assert any("Von 23 Einzelfehlern" in i.value for i in at.info) and any("Fehlt der Pfeil Nachfrage → Faktor" in i.value for i in at.info)          # Experimente 1 und 2 sind exakt und stehen gleich da
    for key, text in (("exp3_run", "Alle 284 Diagramme im Abstand 1 bis 2"), ("exp4_run", "zufälligen Fehlern sinkt der Anteil harmloser Diagramme")):
        at.button(key=key).click().run(timeout=TIMEOUT)
        assert not at.exception, (key, [str(e) for e in at.exception])
        assert any(text in i.value for i in at.info), key


def test_pdf_is_created_only_after_the_click():
    at = _fresh_app()
    assert "pdf_made" not in at.session_state
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    settings, data = at.session_state["pdf_made"]
    assert data[:5] == b"%PDF-" and len(data) > 1000


def test_permalink_values_are_taken_and_clamped():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["n"] = "50"
    at.query_params["tol"] = "9"
    at.query_params["r"] = "3"
    at.query_params["g"] = str(D.encode(K.TRUE_EDGES - {(Z, X)}))
    at.query_params["seed"] = "abc"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.slider(key="n_slider").value == K.N_MIN and at.slider(key="tol_slider").value == K.TOL_MAX and at.radio(key="rule_radio").value == "alle anderen" and at.number_input(key="seed_input").value == K.DEFAULT_SEED
    assert _diagram(at) == K.TRUE_EDGES - {(Z, X)}


def test_a_broken_diagram_code_falls_back_to_the_truth():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["g"] = "xyz"
    at.run()
    assert not at.exception and _diagram(at) == K.TRUE_EDGES


def test_the_address_bar_mirrors_the_settings():
    at = _fresh_app()
    assert at.query_params["g"] in (str(D.encode(K.TRUE_EDGES)), [str(D.encode(K.TRUE_EDGES))]) and at.query_params["r"] in ("0", ["0"])
    at.slider(key="tol_slider").set_value(0.1).run(timeout=TIMEOUT)
    assert at.query_params["tol"] in ("0.1", ["0.1"])
    at.selectbox(key=P.PAIR_KEYS[(Z, X)]).set_value(0).run(timeout=TIMEOUT)
    assert at.query_params["g"] in (str(D.encode(K.TRUE_EDGES - {(Z, X)})), [str(D.encode(K.TRUE_EDGES - {(Z, X)}))])
    at.radio(key="rule_radio").set_value("nichts").run(timeout=TIMEOUT)
    assert at.query_params["r"] in ("4", ["4"])
