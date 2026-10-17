"""Konstanten: die Welt mit sechs Größen, Regeln, Regler-Grenzen, feste Experimente, Farben und Presets."""

SEED_MAX, DEFAULT_SEED = 999999, 35

# Welt (Linie Kausale Diagramme und Adjustierung): S → Z → X → Y, Z → Y, W → Y, Y → C, Einheitsrauschen; wahre Wirkung X → Y = 0,6 (kein Vermittler)
S, Z, X, Y, W, C_ = range(6)
NAMES = ("S", "Z", "X", "Y", "W", "C")
LONG = ("Saison", "Nachfrage", "Faktor", "Servicegrad", "Wetter", "Beschwerden")
COEF = {(S, Z): 0.8, (Z, X): 0.8, (X, Y): 0.6, (Z, Y): 0.6, (W, Y): 0.6, (Y, C_): 0.8}
TRUE_EDGES = frozenset(COEF)
EFFECT = COEF[(X, Y)]
PAIRS = tuple((i, j) for i in range(6) for j in range(i + 1, 6))          # 15 Paare, i < j
PAIR_STATES = ("kein Pfeil", "→", "←")

# Regeln, aus einem (falschen) Diagramm eine Anpassungsmenge zu bilden
RULES = ("Eltern", "Nicht-Nachkommen", "kleinste gültige", "alle anderen", "nichts")
RULE_HELP = {
    "Eltern": "Die Eltern des Faktors im gezeichneten Diagramm (ohne Y): im richtigen Diagramm immer gültig.",
    "Nicht-Nachkommen": "Alle Größen außer X und Y, die im gezeichneten Diagramm keine Nachkommen des Faktors sind.",
    "kleinste gültige": "Die kleinste Menge unter den Nicht-Nachkommen, die im gezeichneten Diagramm den Rückwärtspfad zwischen X und Y blockiert.",
    "alle anderen": "Alle übrigen Größen außer X und Y (auch Wirkungen des Servicegrads).",
    "nichts": "Keine Anpassung: die rohe Differenz.",
}
DEFAULT_RULE = RULES[0]

TOL_MIN, TOL_MAX, TOL_STEP, DEFAULT_TOL = 0.02, 0.20, 0.02, 0.06       # tolerierter Bias in Einheiten der Wirkung (0,06 = 10 % von 0,6)
N_MIN, N_MAX, N_STEP, DEFAULT_N = 100, 5000, 100, 1000

# Experimente: feste Ströme (Seeds aus SEED_BASE), nie ein einzelner Seed als Befund.
SEED_BASE = 1060000
EXP_STRENGTHS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8)                     # Stärke des fehlenden Pfeils Z → X
EXP_HIDDEN = (0.1, 0.2, 0.3, 0.5, 0.8)                                   # Stärken a (U → X) und b (U → Y) der verborgenen Ursache
EXP_DETECT_N, EXP_DETECT_REPS, EXP_DETECT_DISTANCE = 1000, 40, 2          # Experiment 3
EXP_DETECT_RATE = 0.5                                                    # „verworfen“: in mindestens der Hälfte der Datensätze
EXP_MULTI_KS, EXP_MULTI_SAMPLES = (1, 2, 3, 4), 300                      # Experiment 4: k zufällige Fehler
EXP_RMSE_NS, EXP_RMSE_REPS = (200, 1000), 300

COLOR_TRUTH = "#D68A2E"
COLOR_GOOD = "#2F6B65"
COLOR_BAD = "#c0392b"
COLOR_NEUTRAL = "#8a8a8a"
COLOR_CONTROL = "#4c78a8"


def preset(**kw) -> dict:
    base = dict(edges=TRUE_EDGES, rule=DEFAULT_RULE, tol=DEFAULT_TOL, n=DEFAULT_N, seed=DEFAULT_SEED)
    base.update(kw)
    return base


PRESETS = {
    "Das richtige Diagramm (Voreinstellung)": preset(),
    "Pfeil Nachfrage → Faktor fehlt": preset(edges=TRUE_EDGES - {(Z, X)}),
    "Pfeil Nachfrage → Faktor umgedreht": preset(edges=(TRUE_EDGES - {(Z, X)}) | {(X, Z)}),
    "Pfeil Wetter → Faktor zu viel": preset(edges=TRUE_EDGES | {(W, X)}),
    "Markov-Nachbar (Kette umgedreht)": preset(edges=(TRUE_EDGES - {(Z, X), (S, Z)}) | {(X, Z), (Z, S)}),
    "Pfeil Faktor → Servicegrad fehlt (Nicht-Nachkommen)": preset(edges=TRUE_EDGES - {(X, Y)}, rule="Nicht-Nachkommen"),
    "Alles anpassen": preset(rule="alle anderen"),
}
PRESET_HELP = {
    "Das richtige Diagramm (Voreinstellung)": "Das wahre Diagramm der Welt: Die Eltern des Faktors, {Nachfrage}, sind gültig, der Bias ist null. Alle fünf Regeln nebeneinander: nur „alle anderen“ und „nichts“ verzerren, auch bei richtigem Diagramm.",
    "Pfeil Nachfrage → Faktor fehlt": "Der Pfeil, der den Rückwärtspfad über die Nachfrage trägt, fehlt: Die Eltern des Faktors sind dann leer, der Bias ist +0,38. Die Daten verwerfen dieses Diagramm.",
    "Pfeil Nachfrage → Faktor umgedreht": "Dieselbe Stelle, falsch herum: Der Faktor gilt als Ursache der Nachfrage, die Nachfrage wird nicht angepasst, Bias +0,38.",
    "Pfeil Wetter → Faktor zu viel": "Ein Pfeil zu viel: Die Menge wird {Nachfrage, Wetter}. Das ist gültig (der Bias bleibt null) und verkleinert sogar die Streuung, weil das Wetter den Servicegrad miterklärt.",
    "Markov-Nachbar (Kette umgedreht)": "Saison und Nachfrage in der Kette umgedreht (Nachfrage → Saison, Faktor → Nachfrage): dieselben Aussagen über Unabhängigkeit wie das wahre Diagramm, die Daten verwerfen es nicht, aber der Bias ist +0,38.",
    "Pfeil Faktor → Servicegrad fehlt (Nicht-Nachkommen)": "Ohne den Pfeil Faktor → Servicegrad gelten die Beschwerden nicht mehr als Nachkomme des Faktors: Die Regel „Nicht-Nachkommen“ passt an sie an, Bias −0,23.",
    "Alles anpassen": "Das richtige Diagramm, aber alle übrigen Größen angepasst: Die Beschwerden sind eine Wirkung des Servicegrads, der Bias ist −0,23.",
}
