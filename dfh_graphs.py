"""Diagramme aus Pfeilmengen: Zyklen, Einzelfehler, Abstand, Gültigkeit (Backdoor), die fünf Regeln zur Mengenwahl, lokale Markov-Aussagen. Kein Streamlit."""

from __future__ import annotations

import itertools

import dfh_constants as K
import dsep_graph as G

N = 6


def parents_of(edges) -> tuple:
    """Elterntupel je Knoten (Knoten müssen nicht topologisch nummeriert sein)."""
    return tuple(tuple(sorted(i for i, j in edges if j == c)) for c in range(N))


def is_acyclic(edges) -> bool:
    return G._acyclic([list(p) for p in parents_of(edges)])


def edges_from_states(states: dict) -> frozenset:
    """states: {(i, j): 0 kein Pfeil | 1 i → j | 2 j → i} für i < j."""
    return frozenset(((i, j) if s == 1 else (j, i)) for (i, j), s in states.items() if s)


def states_from_edges(edges) -> dict:
    edges = set(edges)
    return {(i, j): (1 if (i, j) in edges else 2 if (j, i) in edges else 0) for i, j in K.PAIRS}


def encode(edges) -> int:
    """Diagramm als ganze Zahl (Ziffern zur Basis 3 über die 15 Paare), für die Adresszeile."""
    return sum(s * 3 ** k for k, (_, s) in enumerate(states_from_edges(edges).items()))


def decode(code: int) -> frozenset:
    code = max(0, min(3 ** len(K.PAIRS) - 1, int(code)))
    states = {}
    for pair in K.PAIRS:
        code, s = divmod(code, 3)
        states[pair] = s
    return edges_from_states(states)


def descendants_of_x(edges) -> set:
    return G.descendants(parents_of(edges), K.X)


def valid_backdoor(edges, adjust) -> bool:
    """Gültig im Diagramm `edges`: keine Nachkommen von X, und X und Y sind im Diagramm ohne die Pfeile aus X durch die Menge getrennt."""
    parents = parents_of(edges)
    if set(adjust) & G.descendants(parents, K.X):
        return False
    cut = tuple(tuple(p for p in ps if p != K.X) for ps in parents)
    return G.d_separated(cut, K.X, K.Y, tuple(sorted(adjust)))


def rule_set(edges, rule: str) -> tuple:
    """Menge nach der Regel im (möglicherweise falschen) Diagramm."""
    parents = parents_of(edges)
    desc = G.descendants(parents, K.X)
    others = [v for v in range(N) if v not in (K.X, K.Y)]
    if rule == "Eltern":
        return tuple(p for p in parents[K.X] if p != K.Y)
    if rule == "Nicht-Nachkommen":
        return tuple(v for v in others if v not in desc)
    if rule == "alle anderen":
        return tuple(others)
    if rule == "nichts":
        return ()
    if rule == "kleinste gültige":
        cands = [v for v in others if v not in desc]
        for r in range(len(cands) + 1):
            for sub in itertools.combinations(cands, r):
                if valid_backdoor(edges, sub):
                    return sub
        return ()
    raise ValueError(rule)


def single_errors(edges=K.TRUE_EDGES) -> list:
    """Alle azyklischen Einzelfehler: (Art, Pfeil, neue Pfeilmenge); Art: fehlt | umgedreht | zu viel."""
    edges = frozenset(edges)
    out = []
    for e in sorted(edges):
        out.append(("fehlt", e, edges - {e}))
    for (i, j) in sorted(edges):
        new = (edges - {(i, j)}) | {(j, i)}
        if is_acyclic(new):
            out.append(("umgedreht", (i, j), new))
    for i in range(N):
        for j in range(N):
            if i != j and (i, j) not in edges and (j, i) not in edges:
                new = edges | {(i, j)}
                if is_acyclic(new):
                    out.append(("zu viel", (i, j), new))
    return out


def diagrams_within(distance: int, edges=K.TRUE_EDGES) -> dict:
    """Alle azyklischen Diagramme, die höchstens `distance` Einzelfehler vom Ausgangsdiagramm entfernt sind: {Pfeilmenge: Abstand}."""
    edges = frozenset(edges)
    seen = {edges: 0}
    level = {edges}
    for d in range(1, distance + 1):
        nxt = set()
        for e in level:
            for _, _, new in single_errors(e):
                if new not in seen:
                    seen[new] = d
                    nxt.add(new)
        level = nxt
    return seen


def difference(edges, truth=K.TRUE_EDGES) -> dict:
    """Abweichungen vom wahren Diagramm: fehlende, zu viele, umgedrehte Pfeile."""
    edges, truth = frozenset(edges), frozenset(truth)
    reversed_ = sorted(e for e in edges if (e[1], e[0]) in truth)
    return {"fehlt": sorted(e for e in truth if e not in edges and (e[1], e[0]) not in edges), "zu viel": sorted(e for e in edges if e not in truth and (e[1], e[0]) not in truth), "umgedreht": reversed_}


def local_statements(edges) -> list:
    """Lokale Markov-Aussagen des Diagramms (a, b, Menge): jede Größe ist von ihren Nicht-Nachkommen (ohne Eltern) gegeben ihre Eltern unabhängig."""
    return G.local_markov_basis(parents_of(edges))


def name(edge) -> str:
    return f"{K.NAMES[edge[0]]} → {K.NAMES[edge[1]]}"


def count_dags(n: int) -> int:
    """Zahl der azyklischen Diagramme auf n benannten Knoten durch Aufzählung aller Pfeilzustände (n ≤ 5)."""
    pairs = list(itertools.combinations(range(n), 2))
    total = 0
    for states in itertools.product(range(3), repeat=len(pairs)):
        ps: list = [[] for _ in range(n)]
        for (i, j), s in zip(pairs, states):
            if s == 1:
                ps[j].append(i)
            elif s == 2:
                ps[i].append(j)
        total += G._acyclic(ps)
    return total
