"""Auswertung: ein gezeichnetes Diagramm gegen die Welt (Mengen nach fünf Regeln, exakter Bias, Schätzung, Prüfung an den Daten) und vier Experimente: Einzelfehler, Stärke des Fehlers, Erkennung an den Daten, mehrere Fehler. Nur numpy und die Module der Linie, kein Streamlit."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import dfh_constants as K
import dfh_graphs as D
import dfh_world as W
import dsep_graph as G
import stl_pc as P
from kausal_generator import SplitMix64


def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma und typografischem Minuszeichen."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def signed(x: float, digits: int = 2) -> str:
    x = 0.0 if abs(x) < 0.5 * 10 ** -digits * 1e-3 else x
    return ("+" if x >= 0 else "−") + de(abs(x), digits)


def pct(x: float, digits: int = 0) -> str:
    return de(100 * x, digits) + " %"


def names_of(cols) -> str:
    cols = sorted(cols)
    return "{" + ", ".join(K.NAMES[c] for c in cols) + "}" if cols else "{}"


def harmless(b: float, tol: float) -> bool:
    return abs(b) <= tol + 1e-12


@dataclass(frozen=True)
class Settings:
    edges: frozenset = K.TRUE_EDGES
    rule: str = K.DEFAULT_RULE
    tol: float = K.DEFAULT_TOL
    n: int = K.DEFAULT_N
    seed: int = K.DEFAULT_SEED


# ----------------------------------------------------------------------------- Analyse eines Diagramms

@dataclass
class RuleRow:
    rule: str
    adjust: tuple
    valid_own: bool               # gültig im gezeichneten Diagramm
    valid_true: bool              # gültig im wahren Diagramm
    bias: float                   # exakt (Bevölkerung)
    harmless: bool


@dataclass
class Analysis:
    settings: Settings
    diff: dict                    # fehlt / zu viel / umgedreht gegenüber dem wahren Diagramm
    errors: int
    rows: list
    chosen: RuleRow
    estimate: float
    se: float
    markov: list                  # (a, b, cond, p, verworfen)
    alpha_each: float
    rejected: bool
    class_size: int
    same_class: bool              # das wahre Diagramm gehört zur Markov-Klasse des gezeichneten
    reasons: list                 # warum die gewählte Menge im wahren Diagramm nicht gültig ist


def rule_row(edges, rule: str, tol: float) -> RuleRow:
    adj = D.rule_set(edges, rule)
    valid_true = D.valid_backdoor(K.TRUE_EDGES, adj)
    return RuleRow(rule, adj, D.valid_backdoor(edges, adj), valid_true, W.bias(adj), harmless(W.bias(adj), tol))


def reasons_invalid(adj) -> list:
    """Gründe, warum `adj` im wahren Diagramm keine gültige Menge ist: Nachkommen des Faktors, offene Rückwärtspfade."""
    out = []
    parents = D.parents_of(K.TRUE_EDGES)
    desc = G.descendants(parents, K.X) & set(adj)
    if desc:
        out.append(f"{names_of(desc)} {'ist' if len(desc) == 1 else 'sind'} im wahren Diagramm Nachkomme des Faktors (X → … → {', '.join(K.NAMES[v] for v in sorted(desc))}).")
    cut = tuple(tuple(p for p in ps if p != K.X) for ps in parents)
    for info in G.explain_paths(cut, K.X, K.Y, tuple(sorted(adj))):
        if not info.blocked:
            text = K.NAMES[info.nodes[0]] + "".join(f" {a} {K.NAMES[v]}" for a, v in zip(info.arrows, info.nodes[1:]))
            out.append(f"Offener Rückwärtspfad {text}.")
    return out


def markov_checks(edges, corr: np.ndarray, n: int) -> tuple:
    """Lokale Markov-Aussagen des Diagramms gegen die Daten: (Aussagen, α je Aussage, mindestens eine verworfen). Fisher-z, Bonferroni."""
    stmts = D.local_statements(edges)
    alpha = 0.05 / max(len(stmts), 1)
    rows = []
    for a, b, cond in stmts:
        p = P.fisher_z_pvalue(P.partial_correlation(corr, a, b, cond), n, len(cond))
        rows.append((a, b, cond, p, p < alpha))
    return rows, alpha, any(r[4] for r in rows)


def analyse(s: Settings) -> Analysis:
    if not D.is_acyclic(s.edges):
        raise ValueError("Das Diagramm enthält einen Kreis.")
    data = W.simulate(s.n, SplitMix64(s.seed))
    rows = [rule_row(s.edges, r, s.tol) for r in K.RULES]
    chosen = next(r for r in rows if r.rule == s.rule)
    est, se = W.regression_estimate(data, chosen.adjust)
    markov, alpha, rej = markov_checks(s.edges, np.corrcoef(data, rowvar=False), s.n)
    members = P.class_members(D.parents_of(s.edges))
    diff = D.difference(s.edges)
    return Analysis(s, diff, sum(len(v) for v in diff.values()), rows, chosen, est, se, markov, alpha, rej, len(members), D.parents_of(K.TRUE_EDGES) in members,
                    [] if chosen.valid_true else reasons_invalid(chosen.adjust))


# ----------------------------------------------------------------------------- Experiment 1: alle 23 Einzelfehler (exakt)

def single_error_experiment() -> dict:
    """Alle azyklischen Einzelfehler des wahren Diagramms, je Regel der exakte Bias (kein Zufall)."""
    rows = []
    for kind, edge, edges in D.single_errors():
        rows.append({"kind": kind, "edge": edge, "label": f"{kind} {D.name(edge)}", "bias": {r: W.bias(D.rule_set(edges, r)) for r in K.RULES},
                     "set": {r: D.rule_set(edges, r) for r in K.RULES}, "valid_true": {r: D.valid_backdoor(K.TRUE_EDGES, D.rule_set(edges, r)) for r in K.RULES}})
    return {"rows": rows, "true_bias": {r: W.bias(D.rule_set(K.TRUE_EDGES, r)) for r in K.RULES}}


def single_summary(res: dict, tol: float) -> dict:
    """Anteil harmloser Einzelfehler je Regel (gesamt und je Art) bei Toleranz `tol`."""
    out = {}
    for rule in K.RULES:
        by_kind = {}
        for kind in ("fehlt", "umgedreht", "zu viel"):
            rs = [r for r in res["rows"] if r["kind"] == kind]
            by_kind[kind] = (sum(harmless(r["bias"][rule], tol) for r in rs), len(rs))
        total = sum(v[0] for v in by_kind.values())
        count = sum(v[1] for v in by_kind.values())
        out[rule] = {"harmless": total, "total": count, "share": total / count, "by_kind": by_kind}
    return out


# ----------------------------------------------------------------------------- Experiment 2: wie stark darf der Fehler sein (exakt)

def missing_arrow_threshold(tol: float) -> float:
    """Größte Stärke des fehlenden Pfeils Z → X, bis zu der das Weglassen (Menge leer) noch den Bias `tol` nicht überschreitet: kleinere Wurzel von 1,64 tol a² − 0,984 a + tol = 0 (0,6 · 1,64 a / (1,64 a² + 1) = tol)."""
    v = 1.0 + 0.8 ** 2                                                  # Var(Z) = 0,64 + 1
    b = K.EFFECT * v
    disc = b * b - 4 * v * tol * tol
    return (b - math.sqrt(disc)) / (2 * v * tol)


def strength_experiment() -> dict:
    """Tabellen: fehlender Pfeil Z → X nach Stärke (Menge leer) und verborgene Ursache U mit den Stärken a (nach X) und b (nach Y) bei Anpassung an Z."""
    missing = [{"a": a, "bias": W.missing_arrow_bias(a)} for a in K.EXP_STRENGTHS]
    hidden = [{"a": a, "b": b, "bias": W.hidden_cause_bias(a, b)} for a in K.EXP_HIDDEN for b in K.EXP_HIDDEN]
    return {"missing": missing, "hidden": hidden, "true_strength": K.COEF[(K.Z, K.X)]}


# ----------------------------------------------------------------------------- Experiment 3: Erkennung an den Daten (Monte-Carlo)

def detection_experiment(reps: int = K.EXP_DETECT_REPS, n: int = K.EXP_DETECT_N, distance: int = K.EXP_DETECT_DISTANCE) -> dict:
    """Alle Diagramme im Abstand 1 bis `distance` vom wahren: Anteil der Datensätze, in denen eine lokale Markov-Aussage verworfen wird, und exakter Bias je Regel."""
    seen = D.diagrams_within(distance)
    corrs = [np.corrcoef(W.simulate(n, SplitMix64(K.SEED_BASE + r)), rowvar=False) for r in range(reps)]
    rows = []
    for edges, dist in seen.items():
        if dist == 0:
            continue
        stmts = D.local_statements(edges)
        alpha = 0.05 / max(len(stmts), 1)
        rate = float(np.mean([any(P.fisher_z_pvalue(P.partial_correlation(c, a, b, cond), n, len(cond)) < alpha for a, b, cond in stmts) for c in corrs]))
        rows.append({"edges": edges, "dist": dist, "rate": rate, "bias": {r: W.bias(D.rule_set(edges, r)) for r in K.RULES}})
    return {"reps": reps, "n": n, "distance": distance, "rows": rows}


def detection_summary(res: dict, tol: float, rule: str) -> dict:
    cnt = {}
    missed = []
    for r in res["rows"]:
        harm = not harmless(r["bias"][rule], tol)
        det = r["rate"] >= K.EXP_DETECT_RATE
        cnt[(r["dist"], harm, det)] = cnt.get((r["dist"], harm, det), 0) + 1
        if harm and not det:
            missed.append(r)
    total_harm = sum(v for (_, h, _), v in cnt.items() if h)
    total_less = sum(v for (_, h, _), v in cnt.items() if not h)
    caught = sum(v for (_, h, d), v in cnt.items() if h and d)
    less_rej = sum(v for (_, h, d), v in cnt.items() if not h and d)
    return {"cnt": cnt, "harmful": total_harm, "harmless": total_less, "harmful_rejected": caught, "harmless_rejected": less_rej, "missed": missed, "total": len(res["rows"])}


# ----------------------------------------------------------------------------- Experiment 4: mehrere Fehler und Streuung (Monte-Carlo)

def multi_error_experiment(ks=K.EXP_MULTI_KS, samples: int = K.EXP_MULTI_SAMPLES) -> dict:
    """k zufällige Einzelfehler nacheinander (azyklisch): |Bias| je Regel."""
    out = {}
    for ki, k in enumerate(ks):
        rng = SplitMix64(K.SEED_BASE + 1000000 + ki)
        us = rng.uniform(samples * k).reshape(samples, k)
        biases = {r: [] for r in K.RULES}
        for row in us:
            edges = K.TRUE_EDGES
            for u in row:
                errs = D.single_errors(edges)
                edges = errs[min(int(u * len(errs)), len(errs) - 1)][2]
            for r in K.RULES:
                biases[r].append(abs(W.bias(D.rule_set(edges, r))))
        out[k] = {r: np.array(v) for r, v in biases.items()}
    return {"samples": samples, "ks": tuple(ks), "abs_bias": out}


def multi_summary(res: dict, tol: float) -> dict:
    return {k: {r: float(np.mean(v <= tol + 1e-12)) for r, v in rules.items()} for k, rules in res["abs_bias"].items()}


RMSE_CASES = {"richtiges Diagramm": K.TRUE_EDGES, "fehlt Z → X": K.TRUE_EDGES - {(K.Z, K.X)}, "zu viel W → X": K.TRUE_EDGES | {(K.W, K.X)}, "zu viel S → X": K.TRUE_EDGES | {(K.S, K.X)}}


def rmse_experiment(ns=K.EXP_RMSE_NS, reps: int = K.EXP_RMSE_REPS) -> dict:
    """Streuung gegen Bias: Schätzung mit der Menge der Regel Eltern in einigen Diagrammen, `reps` Datensätze je n."""
    out = {}
    for ni, n in enumerate(ns):
        datas = [W.simulate(n, SplitMix64(K.SEED_BASE + 2000000 + 10000 * ni + r)) for r in range(reps)]
        for label, edges in RMSE_CASES.items():
            adj = D.rule_set(edges, "Eltern")
            ests = np.array([W.regression_estimate(d, adj)[0] for d in datas])
            out[(label, n)] = {"set": adj, "mean": float(ests.mean()), "sd": float(ests.std()), "rmse": float(np.sqrt(np.mean((ests - K.EFFECT) ** 2)))}
    return {"ns": tuple(ns), "reps": reps, "cases": out}


# ----------------------------------------------------------------------------- Verdikte

def verdict_analysis(a: Analysis) -> tuple:
    s, c = a.settings, a.chosen
    head = f"Regel „{c.rule}“ im gezeichneten Diagramm: {names_of(c.adjust)}, Bias {signed(c.bias)} (Schätzung {de(a.estimate)} ± {de(1.96 * a.se)} gegen wahre Wirkung {de(K.EFFECT)}). "
    if a.errors == 0:
        if c.valid_true and c.harmless:
            return "success", head + "Das gezeichnete Diagramm ist das wahre: Die Menge ist gültig, der Bias ist null."
        return "warning", head + f"Das Diagramm ist richtig, aber diese Regel ist keine gültige Anpassung: {' '.join(a.reasons)}"
    if c.valid_true and c.harmless:
        extra = " Die Daten verwerfen das Diagramm trotzdem." if a.rejected else ""
        return "success", head + f"{a.errors} Fehler im Diagramm, aber die Menge ist im wahren Diagramm gültig (Bias null): der Fehler ist hier harmlos.{extra}"
    if c.harmless:
        return "info", head + f"Die Menge ist im wahren Diagramm nicht gültig, aber der Bias bleibt innerhalb der Toleranz {de(s.tol)}."
    tail = " Die Daten verwerfen das gezeichnete Diagramm." if a.rejected else (" Die Daten sprechen nicht gegen das gezeichnete Diagramm" + (": Es gehört zur selben Markov-Klasse wie das wahre." if a.same_class else "."))
    return "error", head + f"Bias über der Toleranz {de(s.tol)}: {' '.join(a.reasons)}{tail}"


def verdict_single(res: dict, tol: float) -> tuple:
    sm = single_summary(res, tol)
    e, m, nn, al, no = sm["Eltern"], sm["kleinste gültige"], sm["Nicht-Nachkommen"], sm["alle anderen"], sm["nichts"]
    return "info", (f"Von {e['total']} Einzelfehlern (fehlender, umgedrehter oder zusätzlicher Pfeil) sind bei der Regel „Eltern“ {e['harmless']} ({pct(e['share'])}) harmlos (|Bias| ≤ {de(tol)}), bei „kleinste gültige“ {m['harmless']}, bei „Nicht-Nachkommen“ {nn['harmless']}; "
                    f"„alle anderen“ {al['harmless']} und „nichts“ {no['harmless']}: Diese beiden verzerren schon im richtigen Diagramm. Ein einzelner zusätzlicher Pfeil schadet hier nie ({e['by_kind']['zu viel'][0]} von {e['by_kind']['zu viel'][1]}), "
                    f"das ist eine Eigenschaft dieser Welt (kein Sammelknoten vor dem Faktor).")


def verdict_strength(res: dict, tol: float) -> tuple:
    thr = missing_arrow_threshold(tol)
    hidden_ok = [(h["a"], h["b"]) for h in res["hidden"] if harmless(h["bias"], tol)]
    share = len(hidden_ok) / len(res["hidden"])
    return "info", (f"Fehlt der Pfeil Nachfrage → Faktor, ist das Weglassen nur harmlos, wenn er schwächer ist als {de(thr, 3)} (in der Welt {de(res['true_strength'])}): Bias = 0,6 · 1,64 a / (1,64 a² + 1). "
                    f"Eine verborgene gemeinsame Ursache von Faktor und Servicegrad hat nach Anpassung an die Nachfrage den Bias a b / (1 + a²); davon bleiben {len(hidden_ok)} von {len(res['hidden'])} Stärkepaaren ({pct(share)}) innerhalb der Toleranz.")


def verdict_detection(res: dict, tol: float, rule: str) -> tuple:
    sm = detection_summary(res, tol, rule)
    return "info", (f"Alle {sm['total']} Diagramme im Abstand 1 bis {res['distance']} (n = {res['n']}, {res['reps']} Datensätze), Regel „{rule}“: {sm['harmful']} schädlich, davon {sm['harmful_rejected']} von den Daten verworfen, {len(sm['missed'])} nicht. "
                    f"Von {sm['harmless']} harmlosen werden {sm['harmless_rejected']} ({pct(sm['harmless_rejected'] / max(sm['harmless'], 1))}) trotzdem verworfen: Die Daten prüfen das Diagramm, nicht die Wirkung.")


def verdict_multi(res: dict, rmse: dict, tol: float) -> tuple:
    sm = multi_summary(res, tol)
    ks = res["ks"]
    c = rmse["cases"]
    n_hi = max(rmse["ns"])
    return "info", (f"Mit {ks[0]} bis {ks[-1]} zufälligen Fehlerschritten sinkt der Anteil harmloser Diagramme bei „Eltern“ von {pct(sm[ks[0]]['Eltern'])} auf {pct(sm[ks[-1]]['Eltern'])}, bei „Nicht-Nachkommen“ von {pct(sm[ks[0]]['Nicht-Nachkommen'])} auf {pct(sm[ks[-1]]['Nicht-Nachkommen'])}. "
                    f"Streuung gegen Bias (n = {n_hi}): richtig RMSE {de(c[('richtiges Diagramm', n_hi)]['rmse'], 3)}, fehlendes Z → X {de(c[('fehlt Z → X', n_hi)]['rmse'], 3)}, zusätzliches W → X {de(c[('zu viel W → X', n_hi)]['rmse'], 3)}: ein Fehler kann die Streuung senken.")
