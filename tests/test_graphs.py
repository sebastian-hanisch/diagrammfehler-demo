"""Diagramme aus Pfeilmengen: Zyklen, Zustände und Adresszahl, Gültigkeit, die fünf Regeln, Einzelfehler und Abstand."""

import itertools
import random

import pytest

import dfh_constants as K
import dfh_graphs as D
import stl_pc as P

S, Z, X, Y, W, C = range(6)


def test_parents_and_cycles():
    assert D.parents_of(K.TRUE_EDGES) == ((), (S,), (Z,), (Z, X, W), (), (Y,))
    assert D.parents_of({(X, Z)})[Z] == (X,) and D.parents_of({(X, Z)})[X] == ()
    assert D.is_acyclic(K.TRUE_EDGES) and not D.is_acyclic(K.TRUE_EDGES | {(Y, S)}) and not D.is_acyclic({(0, 1), (1, 0)}) and D.is_acyclic(set())
    assert not D.is_acyclic({(0, 1), (1, 2), (2, 0)}) and D.is_acyclic({(0, 1), (1, 2), (0, 2)})


def test_states_round_trip_and_the_address_number():
    states = D.states_from_edges(K.TRUE_EDGES)
    assert len(states) == 15 and sum(1 for s in states.values() if s) == 6 and states[(S, Z)] == 1 and states[(Z, X)] == 1 and states[(S, X)] == 0
    assert D.states_from_edges({(Z, S)})[(S, Z)] == 2 and D.edges_from_states({(S, Z): 2}) == frozenset({(Z, S)}) and D.edges_from_states(states) == K.TRUE_EDGES
    assert D.encode(set()) == 0 and D.encode({(0, 1)}) == 1 and D.encode({(1, 0)}) == 2 and D.encode({(0, 2)}) == 3 and D.decode(0) == frozenset()
    assert D.encode(K.TRUE_EDGES) == 2677861 and D.decode(2677861) == K.TRUE_EDGES
    rng = random.Random(3)
    for _ in range(200):
        code = rng.randrange(3 ** 15)
        assert D.encode(D.decode(code)) == code
    assert D.decode(-5) == frozenset() and D.decode(3 ** 15 + 7) == D.decode(3 ** 15 - 1) == frozenset((j, i) for i, j in K.PAIRS)


def test_valid_backdoor_hand_cases():
    assert D.valid_backdoor(K.TRUE_EDGES, (Z,)) and D.valid_backdoor(K.TRUE_EDGES, (S, Z, W)) and D.valid_backdoor(K.TRUE_EDGES, (Z, W))
    assert not D.valid_backdoor(K.TRUE_EDGES, ()) and not D.valid_backdoor(K.TRUE_EDGES, (S,)) and not D.valid_backdoor(K.TRUE_EDGES, (W,))
    assert not D.valid_backdoor(K.TRUE_EDGES, (Z, C)) and not D.valid_backdoor(K.TRUE_EDGES, (Z, Y)) and not D.valid_backdoor(K.TRUE_EDGES, (S, Z, W, C))
    assert D.valid_backdoor(K.TRUE_EDGES - {(Z, X)}, ()) and D.valid_backdoor(K.TRUE_EDGES | {(W, X)}, (Z, W))


def test_the_five_rules_in_the_true_diagram():
    assert D.rule_set(K.TRUE_EDGES, "Eltern") == (Z,) and D.rule_set(K.TRUE_EDGES, "Nicht-Nachkommen") == (S, Z, W) and D.rule_set(K.TRUE_EDGES, "kleinste gültige") == (Z,)
    assert D.rule_set(K.TRUE_EDGES, "alle anderen") == (S, Z, W, C) and D.rule_set(K.TRUE_EDGES, "nichts") == ()
    with pytest.raises(ValueError):
        D.rule_set(K.TRUE_EDGES, "Zufall")


def test_the_rules_in_wrong_diagrams():
    assert D.rule_set(K.TRUE_EDGES - {(Z, X)}, "Eltern") == () and D.rule_set(K.TRUE_EDGES | {(W, X)}, "Eltern") == (Z, W)
    assert D.rule_set(K.TRUE_EDGES - {(Z, X)}, "kleinste gültige") == ()                                  # ohne den Pfeil gibt es im Diagramm keinen Rückwärtspfad
    assert D.rule_set(K.TRUE_EDGES - {(X, Y)}, "Nicht-Nachkommen") == (S, Z, W, C)                       # C ist dann kein Nachkomme von X
    assert D.rule_set((K.TRUE_EDGES - {(X, Y)}) | {(Y, X)}, "Eltern") == (Z,)                            # Y als Elternteil von X wird weggelassen
    assert D.rule_set((K.TRUE_EDGES - {(Z, X)}) | {(X, Z)}, "Nicht-Nachkommen") == (S, W)


def test_single_errors_counts_and_acyclicity():
    errs = D.single_errors()
    kinds = [k for k, _, _ in errs]
    pairs = [(k, edge) for k, edge, _ in errs]
    assert len(errs) == 23 and kinds.count("fehlt") == 6 and kinds.count("umgedreht") == 5 and kinds.count("zu viel") == 12
    assert all(D.is_acyclic(e) for _, _, e in errs) and all(e != K.TRUE_EDGES for _, _, e in errs) and len({e for _, _, e in errs}) == 23
    assert ("umgedreht", (Z, Y)) not in pairs                                                            # Y → Z schlösse den Kreis Z → X → Y → Z
    assert ("zu viel", (W, X)) in pairs and ("zu viel", (X, S)) not in pairs
    assert [edge for k, edge, _ in errs if k == "fehlt"] == sorted(K.TRUE_EDGES) and [edge for k, edge, _ in errs if k == "umgedreht"] == [(S, Z), (Z, X), (X, Y), (Y, C), (W, Y)]
    assert [len(e ^ K.TRUE_EDGES) for k, _, e in errs if k == "fehlt"] == [1] * 6 and [len(e ^ K.TRUE_EDGES) for k, _, e in errs if k == "umgedreht"] == [2] * 5


def test_diagrams_within_distance():
    d1, d2 = D.diagrams_within(1), D.diagrams_within(2)
    assert len(d1) == 24 and len(d2) == 285 and d1[K.TRUE_EDGES] == 0 and sum(1 for v in d2.values() if v == 1) == 23 and sum(1 for v in d2.values() if v == 2) == 261
    assert all(D.is_acyclic(e) for e in d2) and set(d1) <= set(d2) and D.diagrams_within(0) == {K.TRUE_EDGES: 0}
    assert D.diagrams_within(1, set())[frozenset({(0, 1)})] == 1


def test_difference_names_each_kind():
    assert D.difference(K.TRUE_EDGES) == {"fehlt": [], "zu viel": [], "umgedreht": []}
    d = D.difference((K.TRUE_EDGES - {(Z, X), (W, Y)}) | {(X, Z), (W, X)})
    assert d["fehlt"] == [(W, Y)] and d["zu viel"] == [(W, X)] and d["umgedreht"] == [(X, Z)]


def test_difference_lists_are_sorted_ascending():
    edges = (K.TRUE_EDGES - {(S, Z), (Z, X), (X, Y), (W, Y)}) | {(S, X), (S, W), (Y, X), (Y, W)}
    d = D.difference(edges)
    assert d["fehlt"] == [(S, Z), (Z, X)] and d["zu viel"] == [(S, X), (S, W)] and d["umgedreht"] == [(Y, X), (Y, W)]


def test_the_smallest_valid_set_can_be_all_non_descendants():
    edges = frozenset({(v, X) for v in (S, Z, W, C)} | {(v, Y) for v in (S, Z, W, C)} | {(X, Y)})
    assert D.rule_set(edges, "kleinste gültige") == (S, Z, W, C) == D.rule_set(edges, "Nicht-Nachkommen") == D.rule_set(edges, "Eltern")
    assert D.valid_backdoor(edges, (S, Z, W, C)) and not any(D.valid_backdoor(edges, sub) for r in range(4) for sub in itertools.combinations((S, Z, W, C), r))


def test_local_statements_of_the_true_diagram():
    st = D.local_statements(K.TRUE_EDGES)
    assert len(st) == 11 and (S, W, ()) in st and (X, W, (Z,)) in st and (Z, W, (S,)) in st and (W, C, (Y,)) in st and (W, C, ()) not in st
    assert len(D.local_statements(set())) == 15


def test_markov_class_of_the_true_diagram_and_its_neighbour():
    members = P.class_members(D.parents_of(K.TRUE_EDGES))
    assert len(members) == 3 and D.parents_of((K.TRUE_EDGES - {(Z, X), (S, Z)}) | {(X, Z), (Z, S)}) in members
    assert D.parents_of(K.TRUE_EDGES - {(Z, X)}) not in members


def test_name():
    assert D.name((Z, X)) == "Z → X" and D.name((W, Y)) == "W → Y"
