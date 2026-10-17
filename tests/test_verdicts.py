"""Verdikte: Art und Text je Lage; jede Zahl im Text stammt aus dem Befund, nicht aus einer Konstanten."""

import dataclasses

import dfh_constants as K
import dfh_evaluation as ev

S, Z, X, Y, W, C = range(6)


def _a(**kw):
    return ev.analyse(ev.Settings(**kw))


def test_analysis_verdict_true_diagram_is_a_success():
    kind, text = ev.verdict_analysis(_a())
    assert kind == "success" and "Regel „Eltern“ im gezeichneten Diagramm: {Z}, Bias +0,00" in text and "gegen wahre Wirkung 0,60" in text and "Das gezeichnete Diagramm ist das wahre: Die Menge ist gültig, der Bias ist null." in text


def test_analysis_verdict_right_diagram_with_a_bad_rule_is_a_warning_that_names_the_reason():
    kind, text = ev.verdict_analysis(_a(rule="alle anderen"))
    assert kind == "warning" and "{S, Z, W, C}, Bias −0,23" in text and "Das Diagramm ist richtig, aber diese Regel ist keine gültige Anpassung" in text and "{C} ist im wahren Diagramm Nachkomme des Faktors" in text
    kind, text = ev.verdict_analysis(_a(rule="nichts"))
    assert kind == "warning" and "{}, Bias +0,38" in text and "Offener Rückwärtspfad X ← Z → Y" in text


def test_analysis_verdict_a_missing_arrow_is_an_error_that_the_data_reject():
    kind, text = ev.verdict_analysis(_a(edges=K.TRUE_EDGES - {(Z, X)}))
    assert kind == "error" and "Bias +0,38" in text and "Bias über der Toleranz 0,06" in text and "Offener Rückwärtspfad X ← Z → Y." in text and text.endswith("Die Daten verwerfen das gezeichnete Diagramm.")


def test_analysis_verdict_the_markov_neighbour_is_an_error_the_data_do_not_show():
    kind, text = ev.verdict_analysis(_a(edges=(K.TRUE_EDGES - {(Z, X), (S, Z)}) | {(X, Z), (Z, S)}))
    assert kind == "error" and text.endswith("Es gehört zur selben Markov-Klasse wie das wahre.")
    a = _a(edges=K.TRUE_EDGES - {(Z, X)})
    text2 = ev.verdict_analysis(dataclasses.replace(a, rejected=False, same_class=False))[1]
    assert text2.endswith("Die Daten sprechen nicht gegen das gezeichnete Diagramm.")


def test_analysis_verdict_an_extra_arrow_is_harmless():
    kind, text = ev.verdict_analysis(_a(edges=K.TRUE_EDGES | {(W, X)}))
    assert kind == "success" and "1 Fehler im Diagramm, aber die Menge ist im wahren Diagramm gültig (Bias null): der Fehler ist hier harmlos." in text and "{Z, W}" in text
    rejected = ev.verdict_analysis(dataclasses.replace(_a(edges=K.TRUE_EDGES | {(W, X)}), rejected=True))[1]
    assert rejected.endswith("Die Daten verwerfen das Diagramm trotzdem.")


def test_analysis_verdict_invalid_but_inside_the_tolerance_is_info():
    edges = (K.TRUE_EDGES - {(Z, X), (Y, C)}) | {(X, Z)}
    kind, text = ev.verdict_analysis(_a(edges=edges, rule="Nicht-Nachkommen", tol=0.10))
    assert kind == "info" and "Bias −0,10" in text and "im wahren Diagramm nicht gültig, aber der Bias bleibt innerhalb der Toleranz 0,10" in text
    assert ev.verdict_analysis(_a(edges=edges, rule="Nicht-Nachkommen", tol=0.08))[0] == "error"


def test_single_verdict(single):
    kind, text = ev.verdict_single(single, 0.06)
    assert kind == "info" and "Von 23 Einzelfehlern" in text and "bei der Regel „Eltern“ 21 (91 %) harmlos (|Bias| ≤ 0,06)" in text and "„kleinste gültige“ 19" in text and "„Nicht-Nachkommen“ 18" in text
    assert "„alle anderen“ 0 und „nichts“ 0" in text and "Zusätzliche Pfeile schaden hier nie (12 von 12)" in text and "Eigenschaft dieser Welt" in text
    assert "„Eltern“ 23 (100 %)" in ev.verdict_single(single, 0.4)[1]


def test_strength_verdict(strength):
    kind, text = ev.verdict_strength(strength, 0.06)
    assert kind == "info" and "schwächer ist als 0,061 (in der Welt 0,80)" in text and "11 von 25 Stärkepaaren (44 %)" in text
    assert "schwächer ist als 0,020" in ev.verdict_strength(strength, 0.02)[1] and "3 von 25 Stärkepaaren (12 %)" in ev.verdict_strength(strength, 0.02)[1]
    assert "schwächer ist als 0,219" in ev.verdict_strength(strength, 0.2)[1] and "21 von 25 Stärkepaaren (84 %)" in ev.verdict_strength(strength, 0.2)[1]


def test_detection_verdict(small_detect):
    kind, text = ev.verdict_detection(small_detect, 0.06, "Eltern")
    assert kind == "info" and "Alle 23 Diagramme im Abstand 1 bis 1 (n = 500, 3 Datensätze), Regel „Eltern“: 2 schädlich, davon 2 von den Daten verworfen, 0 nicht." in text
    assert "Von 21 harmlosen werden 8 (38 %) trotzdem verworfen: Die Daten prüfen das Diagramm, nicht die Wirkung." in text


def test_multi_verdict(small_multi):
    res, rmse = small_multi
    kind, text = ev.verdict_multi(res, rmse, 0.06)
    assert kind == "info" and "Mit 1 bis 2 zufälligen Fehlern sinkt der Anteil harmloser Diagramme bei „Eltern“ von " in text and "Streuung gegen Bias (n = 200)" in text and "richtig RMSE" in text and "fehlendes Z → X" in text


def test_multi_verdict_names_the_largest_sample_size():
    res = ev.multi_error_experiment(ks=(1, 2), samples=6)
    rmse = ev.rmse_experiment(ns=(100, 200), reps=3)
    assert "Streuung gegen Bias (n = 200)" in ev.verdict_multi(res, rmse, 0.06)[1]
