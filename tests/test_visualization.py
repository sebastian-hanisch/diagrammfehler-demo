"""Figuren und Tabellen: bauen ohne Fehler, und die gezeichneten Zahlen sind die der Auswertung."""

import re

import pytest

import dfh_constants as K
import dfh_evaluation as ev
import dfh_visualization as V

S, Z, X, Y, W, C = range(6)


def _a(**kw):
    return ev.analyse(ev.Settings(**kw))


def _arrows(fig):
    return [ann for ann in fig.layout.annotations if ann.showarrow]


def _fills(fig):
    return {t.text[0]: t.marker.color for t in fig.data if t.mode == "markers+text"}


def test_true_diagram_draws_every_arrow_and_marks_x_and_y():
    fig = V.build_true_diagram()
    assert len(_arrows(fig)) == 6
    nodes = {t.text[0]: t for t in fig.data if t.mode == "markers+text"}
    assert set(nodes) == set(K.NAMES) and nodes["X"].marker.line.color == K.COLOR_GOOD and nodes["Y"].marker.line.color == K.COLOR_TRUTH


def test_drawn_diagram_of_the_truth_is_all_grey_and_marks_the_adjustment_set():
    fig = V.build_drawn_diagram(_a())
    assert {ann.arrowcolor for ann in _arrows(fig)} == {V.KIND_COLORS["richtig"]} and len(_arrows(fig)) == 6
    assert {t.name for t in fig.data if t.name} == {"richtig", "angepasst (gewählte Regel)"}
    fills = _fills(fig)
    assert fills["Z"] == V.SET_FILL and all(fills[v] != V.SET_FILL for v in "SXYWC")
    assert _fills(V.build_drawn_diagram(_a(rule="alle anderen")))["C"] == V.SET_FILL and _fills(V.build_drawn_diagram(_a(rule="nichts")))["Z"] != V.SET_FILL


def test_drawn_diagram_colours_each_kind_of_error():
    fig = V.build_drawn_diagram(_a(edges=(K.TRUE_EDGES - {(Z, X), (W, Y)}) | {(X, Z), (S, X)}))
    colours = sorted(ann.arrowcolor for ann in _arrows(fig))
    assert V.KIND_COLORS["umgedreht"] in colours and V.KIND_COLORS["zu viel"] in colours and colours.count(V.KIND_COLORS["richtig"]) == 4 and len(_arrows(fig)) == 6
    names = {t.name for t in fig.data if t.name}
    assert {"richtig", "umgedreht", "zu viel", "fehlt"} <= names
    dashed = [t for t in fig.data if t.mode == "lines" and t.line.dash == "dash" and t.x[0] is not None]
    assert len(dashed) == 1


def test_rules_chart_and_table_show_the_exact_biases():
    a = _a(edges=K.TRUE_EDGES - {(Z, X)})
    fig = V.build_rules_chart(a)
    bar = fig.data[0]
    assert list(bar.x) == list(K.RULES) and list(bar.y) == pytest.approx([r.bias for r in a.rows]) and list(bar.marker.color) == [K.COLOR_BAD, K.COLOR_GOOD, K.COLOR_BAD, K.COLOR_BAD, K.COLOR_BAD]
    df = V.build_rules_table(a)
    assert list(df["Regel"]) == list(K.RULES) and list(df["Menge"])[0] == "{}" and df["Bias (exakt)"][0] == "+0,384" and list(df["innerhalb der Toleranz"]) == ["nein", "ja", "nein", "nein", "nein"]
    assert list(df["gültig im wahren Diagramm"]) == ["nein", "ja", "nein", "nein", "nein"] and list(df["gültig im gezeichneten Diagramm"])[0] == "ja"
    assert V.style_rules_table(df).to_html().count("rgba(192,57,43,0.15)") > 0


def test_differences_markov_and_summary_tables():
    a = _a(edges=(K.TRUE_EDGES - {(Z, X)}) | {(X, Z), (W, X)})
    df = V.build_differences_table(a)
    assert sorted(zip(df["Art"], df["Pfeil im wahren Diagramm"])) == [("umgedreht", "X → Z"), ("zu viel", "W → X")]
    assert list(V.build_differences_table(_a())["Art"]) == ["—"]
    m = V.build_markov_table(_a())
    assert len(m) == 11 and "S ⊥ W | {}" in list(m["Aussage des gezeichneten Diagramms"]) and set(m["verworfen"]) == {"nein"}
    s = V.build_summary_table(_a())
    assert list(s["Größe"])[0].startswith("Fehler") and s["Wert"][0] == "0" and "{Z}" in s["Wert"][1] and "3; das wahre gehört dazu: ja" in s["Wert"][4] and s["Wert"][5] == "0,06"


def test_single_experiment_figures(single):
    fig = V.build_single_chart(single, 0.06)
    assert list(fig.data[0].x) == list(K.RULES) and list(fig.data[0].y) == pytest.approx([100 * 21 / 23, 100 * 18 / 23, 100 * 19 / 23, 0, 0]) and fig.data[0].text[0] == "21 von 23"
    df = V.build_single_table(single, 0.06)
    assert len(df) == 23 and list(df.columns) == ["Einzelfehler", *K.RULES] and df["Einzelfehler"][0] == "fehlt S → Z"
    html = V.style_single_table(df, 0.06).to_html()
    style = html.split("</style>")[0]
    assert "rgba(192,57,43,0.18)" in style and len(re.findall(r"_row\d+_col\d+", style)) == sum(1 for r in single["rows"] for rule in K.RULES if abs(r["bias"][rule]) > 0.06)
    kinds = V.build_single_kind_table(single, 0.06)
    assert list(kinds.columns) == ["Regel", "alle", "fehlt", "umgedreht", "zu viel"] and kinds["alle"][0] == "21 von 23" and kinds["zu viel"][0] == "12 von 12"


def test_strength_figures(strength):
    fig = V.build_strength_chart(strength, 0.06)
    assert list(fig.data[0].x) == list(K.EXP_STRENGTHS) and len(fig.layout.shapes) == 2
    tables = V.build_strength_tables(strength, 0.06)
    assert len(tables[0]) == len(K.EXP_STRENGTHS) and tables[0]["innerhalb der Toleranz"][0] == "ja" and tables[0]["innerhalb der Toleranz"][3] == "nein"
    assert tables[1].shape == (5, 6) and tables[1]["b = 0,5"][3].startswith("+0,200") and tables[1]["b = 0,5"][3].endswith("✗")


def test_detection_figures(small_detect):
    fig = V.build_detection_chart(small_detect, 0.06, "Eltern")
    assert [t.name for t in fig.data] == ["harmlos, nicht verworfen", "harmlos, verworfen", "schädlich, verworfen", "schädlich, NICHT verworfen"] and sum(sum(t.y) for t in fig.data) == 23
    df = V.build_detection_table(small_detect, 0.06, "Eltern")
    assert list(df.columns) == ["Abweichungen vom wahren Diagramm", "Abstand", "verworfen in", "Bias"] and len(df) >= 1
    empty = V.build_detection_table(small_detect, 0.6, "Eltern")
    assert empty["Abweichungen vom wahren Diagramm"][0] == "keines"


def test_multi_figures(small_multi):
    res, rmse = small_multi
    fig = V.build_multi_chart(res, 0.06)
    assert [t.name for t in fig.data] == list(K.RULES) and list(fig.data[0].x) == [1, 2]
    multi, rm = V.build_multi_tables(res, rmse, 0.06)
    assert list(multi.columns) == ["Fehler", *K.RULES] and len(multi) == 2 and multi["alle anderen"][0] == "0 %"
    assert len(rm) == 4 and list(rm["n"]) == [200] * 4 and rm["Menge (Regel Eltern)"][0] == "{Z}"
