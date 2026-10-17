"""Plotly-Figuren und Tabellen. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängen bleibt."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

import dfh_constants as K
import dfh_evaluation as E
import dfh_graphs as D
from dfh_evaluation import Analysis, de, harmless, names_of, pct, signed

POS = {K.S: (0.0, 1.0), K.Z: (0.32, 1.0), K.W: (0.95, 1.0), K.X: (0.32, 0.0), K.Y: (0.66, 0.5), K.C_: (1.0, 0.0)}
KIND_COLORS = {"richtig": "#6b6b6b", "zu viel": K.COLOR_BAD, "umgedreht": K.COLOR_TRUTH, "fehlt": K.COLOR_NEUTRAL}
SET_FILL = "rgba(76,120,168,0.35)"


def _fixed(fig: go.Figure, height: int = 380, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.22), **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _segment(p0, p1, r: float = 0.1):
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    ln = (dx * dx + dy * dy) ** 0.5 or 1.0
    ux, uy = dx / ln, dy / ln
    return (p0[0] + ux * r, p0[1] + uy * r), (p1[0] - ux * r, p1[1] - uy * r)


def _arrow(fig: go.Figure, a: int, b: int, color: str, width: float = 3):
    s, e = _segment(POS[a], POS[b])
    fig.add_annotation(x=e[0], y=e[1], ax=s[0], ay=s[1], xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3, arrowsize=1.3, arrowwidth=width, arrowcolor=color, text="")


def _nodes(fig: go.Figure, adjust=()):
    outlines = {K.X: K.COLOR_GOOD, K.Y: K.COLOR_TRUTH}
    for v, (x, y) in POS.items():
        fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers+text", text=[K.NAMES[v]], textposition="middle center", textfont=dict(size=12),
                                 marker=dict(size=58, color=SET_FILL if v in adjust else "rgba(47,107,101,0.08)", line=dict(width=2, color=outlines.get(v, "#1b1b1b"))), hoverinfo="skip", showlegend=False))


def _frame(fig: go.Figure, height: int = 330, **layout) -> go.Figure:
    fig.update_xaxes(visible=False, range=[-0.2, 1.2])
    fig.update_yaxes(visible=False, range=[-0.3, 1.3])
    return _fixed(fig, height, **layout)


def _legend_line(fig: go.Figure, name: str, color: str, dash: str = "solid"):
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=color, width=4, dash=dash), name=name))


def build_true_diagram() -> go.Figure:
    fig = go.Figure()
    for a, b in sorted(K.TRUE_EDGES):
        _arrow(fig, a, b, "#6b6b6b")
    _nodes(fig)
    return _frame(fig, showlegend=False)


def build_drawn_diagram(a: Analysis) -> go.Figure:
    """Das gezeichnete Diagramm nach Urteil gegen das wahre gefärbt; fehlende wahre Pfeile gestrichelt grau; Knoten der gewählten Menge blau gefüllt."""
    fig = go.Figure()
    edges = a.settings.edges
    present = set()
    for e in sorted(edges):
        if e in K.TRUE_EDGES:
            kind = "richtig"
        elif (e[1], e[0]) in K.TRUE_EDGES:
            kind = "umgedreht"
        else:
            kind = "zu viel"
        present.add(kind)
        _arrow(fig, e[0], e[1], KIND_COLORS[kind], 3 if kind == "richtig" else 4)
    for e in a.diff["fehlt"]:
        present.add("fehlt")
        s, t = _segment(POS[e[0]], POS[e[1]])
        fig.add_trace(go.Scatter(x=[s[0], t[0]], y=[s[1], t[1]], mode="lines", line=dict(color=K.COLOR_NEUTRAL, width=2, dash="dash"), hoverinfo="skip", showlegend=False))
    _nodes(fig, a.chosen.adjust)
    for kind in ("richtig", "umgedreht", "zu viel", "fehlt"):
        if kind in present:
            _legend_line(fig, kind, KIND_COLORS[kind], "dash" if kind == "fehlt" else "solid")
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=14, color=SET_FILL, line=dict(width=1, color="#1b1b1b")), name="angepasst (gewählte Regel)"))
    return _frame(fig, showlegend=True)


def build_rules_chart(a: Analysis) -> go.Figure:
    fig = go.Figure()
    tol = a.settings.tol
    fig.add_trace(go.Bar(x=[r.rule for r in a.rows], y=[r.bias for r in a.rows], marker_color=[K.COLOR_GOOD if r.harmless else K.COLOR_BAD for r in a.rows], text=[signed(r.bias) for r in a.rows], textposition="outside"))
    fig.add_hrect(y0=-tol, y1=tol, fillcolor=K.COLOR_GOOD, opacity=0.1, line_width=0)
    fig.add_hline(y=0, line=dict(color="#888888", width=1))
    fig.update_yaxes(title="Bias der geschätzten Wirkung (wahr 0,6)", range=[-0.5, 0.6])
    fig.update_xaxes(title="Regel zur Wahl der Menge (grün: innerhalb der Toleranz)")
    return _fixed(fig, 340, showlegend=False)


def build_rules_table(a: Analysis) -> pd.DataFrame:
    return pd.DataFrame([{"Regel": r.rule, "Menge": names_of(r.adjust), "gültig im gezeichneten Diagramm": "ja" if r.valid_own else "nein", "gültig im wahren Diagramm": "ja" if r.valid_true else "nein",
                          "Bias (exakt)": signed(r.bias, 3), "innerhalb der Toleranz": "ja" if r.harmless else "nein"} for r in a.rows])


def style_rules_table(df: pd.DataFrame):
    def row_style(row):
        return ["background-color: rgba(192,57,43,0.15)" if row["innerhalb der Toleranz"] == "nein" else ""] * len(row)

    return df.style.apply(row_style, axis=1)


def build_differences_table(a: Analysis) -> pd.DataFrame:
    rows = [{"Art": k, "Pfeil im wahren Diagramm": D.name(e)} for k, es in a.diff.items() for e in es]
    return pd.DataFrame(rows or [{"Art": "—", "Pfeil im wahren Diagramm": "keine Abweichung"}])


def build_markov_table(a: Analysis) -> pd.DataFrame:
    rows = [{"Aussage des gezeichneten Diagramms": f"{K.NAMES[x]} ⊥ {K.NAMES[y]} | {names_of(c)}", "p-Wert (Fisher-z)": de(p, 3), "verworfen": "ja" if rej else "nein"} for x, y, c, p, rej in a.markov]
    return pd.DataFrame(rows)


def build_summary_table(a: Analysis) -> pd.DataFrame:
    s = a.settings
    return pd.DataFrame([
        {"Größe": "Fehler gegenüber dem wahren Diagramm", "Wert": str(a.errors)},
        {"Größe": "Gewählte Regel, Menge", "Wert": f"{s.rule}, {names_of(a.chosen.adjust)}"},
        {"Größe": "Schätzung der Wirkung (n, 95-%-Intervall)", "Wert": f"{de(a.estimate)} ({s.n}, ± {de(1.96 * a.se)}); wahr {de(K.EFFECT)}"},
        {"Größe": f"Lokale Markov-Aussagen, Niveau je Aussage (Bonferroni, 0,05 gesamt)", "Wert": f"{len(a.markov)}, {de(a.alpha_each, 4)}"},
        {"Größe": "Mitglieder der Markov-Klasse des gezeichneten Diagramms", "Wert": f"{a.class_size}; das wahre gehört dazu: {'ja' if a.same_class else 'nein'}"},
        {"Größe": "Toleranz für den Bias", "Wert": de(s.tol)},
    ])


# ----------------------------------------------------------------------------- Experimente

def build_single_chart(res: dict, tol: float) -> go.Figure:
    sm = E.single_summary(res, tol)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(K.RULES), y=[100 * sm[r]["share"] for r in K.RULES], marker_color=[K.COLOR_GOOD if sm[r]["share"] > 0.5 else K.COLOR_BAD for r in K.RULES],
                         text=[f"{sm[r]['harmless']} von {sm[r]['total']}" for r in K.RULES], textposition="outside"))
    fig.update_yaxes(title="harmlose Einzelfehler (%)", range=[0, 115])
    fig.update_xaxes(title="Regel zur Wahl der Menge")
    return _fixed(fig, 340, showlegend=False)


def build_single_table(res: dict, tol: float) -> pd.DataFrame:
    rows = []
    for r in res["rows"]:
        rows.append({"Einzelfehler": r["label"], **{rule: signed(r["bias"][rule], 3) for rule in K.RULES}})
    return pd.DataFrame(rows)


def style_single_table(df: pd.DataFrame, tol: float):
    def bad(v):
        try:
            return abs(float(v.replace("−", "-").replace(",", "."))) > tol + 1e-12
        except ValueError:
            return False

    def cell(v):
        return "background-color: rgba(192,57,43,0.18)" if bad(v) else ""

    return df.style.map(cell, subset=list(K.RULES))


def build_single_kind_table(res: dict, tol: float) -> pd.DataFrame:
    sm = E.single_summary(res, tol)
    return pd.DataFrame([{"Regel": r, "alle": f"{sm[r]['harmless']} von {sm[r]['total']}", **{k: f"{v[0]} von {v[1]}" for k, v in sm[r]["by_kind"].items()}} for r in K.RULES])


def build_strength_chart(res: dict, tol: float) -> go.Figure:
    xs = [r["a"] for r in res["missing"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["bias"] for r in res["missing"]], mode="lines+markers", name="Bias ohne Anpassung", line=dict(color=K.COLOR_BAD, width=3)))
    fig.add_hrect(y0=-tol, y1=tol, fillcolor=K.COLOR_GOOD, opacity=0.12, line_width=0)
    fig.add_vline(x=E.missing_arrow_threshold(tol), line=dict(color=K.COLOR_GOOD, width=2, dash="dash"), annotation_text=f"Grenze {de(E.missing_arrow_threshold(tol), 3)}", annotation_position="top right")
    fig.update_xaxes(title="Stärke des fehlenden Pfeils Nachfrage → Faktor (in der Welt 0,8)")
    fig.update_yaxes(title="Bias der geschätzten Wirkung", range=[-0.02, 0.45])
    return _fixed(fig, 340)


def build_strength_tables(res: dict, tol: float) -> list:
    missing = pd.DataFrame([{"Stärke des fehlenden Pfeils": de(r["a"], 2), "Bias": signed(r["bias"], 3), "innerhalb der Toleranz": "ja" if harmless(r["bias"], tol) else "nein"} for r in res["missing"]])
    bs = sorted({h["b"] for h in res["hidden"]})
    rows = []
    for a_ in sorted({h["a"] for h in res["hidden"]}):
        row = {"a (Ursache → Faktor)": de(a_, 1)}
        for b in bs:
            bias = next(h["bias"] for h in res["hidden"] if h["a"] == a_ and h["b"] == b)
            row[f"b = {de(b, 1)}"] = signed(bias, 3) + ("" if harmless(bias, tol) else " ✗")
        rows.append(row)
    return [missing, pd.DataFrame(rows)]


def build_detection_chart(res: dict, tol: float, rule: str) -> go.Figure:
    sm = E.detection_summary(res, tol, rule)
    cats = (("harmlos, nicht verworfen", False, False, K.COLOR_GOOD), ("harmlos, verworfen", False, True, K.COLOR_NEUTRAL), ("schädlich, verworfen", True, True, K.COLOR_CONTROL), ("schädlich, NICHT verworfen", True, False, K.COLOR_BAD))
    fig = go.Figure()
    dists = list(range(1, res["distance"] + 1))
    for label, harm, det, color in cats:
        fig.add_trace(go.Bar(x=[f"Abstand {d}" for d in dists], y=[sm["cnt"].get((d, harm, det), 0) for d in dists], name=label, marker_color=color))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title="Zahl der Diagramme")
    return _fixed(fig, 360)


def build_detection_table(res: dict, tol: float, rule: str) -> pd.DataFrame:
    sm = E.detection_summary(res, tol, rule)
    rows = [{"Abweichungen vom wahren Diagramm": ", ".join(f"{k} {D.name(e)}" for k, es in D.difference(r["edges"]).items() for e in es), "Abstand": r["dist"], "verworfen in": pct(r["rate"]), "Bias": signed(r["bias"][rule], 3)}
            for r in sm["missed"]]
    return pd.DataFrame(rows or [{"Abweichungen vom wahren Diagramm": "keines", "Abstand": "—", "verworfen in": "—", "Bias": "—"}])


def build_multi_chart(res: dict, tol: float) -> go.Figure:
    sm = E.multi_summary(res, tol)
    fig = go.Figure()
    colors = {"Eltern": K.COLOR_GOOD, "Nicht-Nachkommen": K.COLOR_CONTROL, "kleinste gültige": K.COLOR_TRUTH, "alle anderen": K.COLOR_NEUTRAL, "nichts": K.COLOR_BAD}
    for rule in K.RULES:
        fig.add_trace(go.Scatter(x=list(res["ks"]), y=[100 * sm[k][rule] for k in res["ks"]], mode="lines+markers", name=rule, line=dict(color=colors[rule], width=3)))
    fig.update_xaxes(title="Zahl der zufälligen Fehler im Diagramm", dtick=1)
    fig.update_yaxes(title="Diagramme mit Bias innerhalb der Toleranz (%)", range=[-2, 104])
    return _fixed(fig, 360)


def build_multi_tables(res: dict, rmse: dict, tol: float) -> list:
    sm = E.multi_summary(res, tol)
    multi = pd.DataFrame([{"Fehler": k, **{r: pct(sm[k][r]) for r in K.RULES}} for k in res["ks"]])
    rows = []
    for (label, n), v in rmse["cases"].items():
        rows.append({"Diagramm": label, "n": n, "Menge (Regel Eltern)": names_of(v["set"]), "Mittel": de(v["mean"], 3), "Streuung": de(v["sd"], 3), "RMSE": de(v["rmse"], 3)})
    return [multi, pd.DataFrame(rows)]
