"""Permalink (Begrenzen), Presets, Einzelfehler-Auswahl und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio). Die 15 Paare des Diagramms stehen als eine Zahl zur Basis 3 in der Adresszeile (Parameter g),
die Regel als Index (r); alle anderen Regler wie sonst."""

import random
from dataclasses import dataclass

import streamlit as st

import dfh_constants as K
import dfh_graphs as D


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float
    hi: float
    step: float
    choices: tuple | None = None


def _choice(param: str, choices: tuple, default) -> SettingSpec:
    return SettingSpec(param, int, default, 0, len(choices) - 1, 1, choices)


SETTING_SPECS = {
    "rule_radio": _choice("r", K.RULES, K.DEFAULT_RULE),
    "tol_slider": SettingSpec("tol", float, K.DEFAULT_TOL, K.TOL_MIN, K.TOL_MAX, K.TOL_STEP),
    "n_slider": SettingSpec("n", int, K.DEFAULT_N, K.N_MIN, K.N_MAX, K.N_STEP),
    "seed_input": SettingSpec("seed", int, K.DEFAULT_SEED, 0, K.SEED_MAX, 1),
}
PRESET_KEYS = {"rule": "rule_radio", "tol": "tol_slider", "n": "n_slider", "seed": "seed_input"}
PAIR_KEYS = {pair: f"pair_{pair[0]}_{pair[1]}" for pair in K.PAIRS}
DIAGRAM_PARAM = "g"
ERROR_LABELS = {f"{kind} {D.name(edge)}": edges for kind, edge, edges in D.single_errors()}
NO_ERROR = "— Einzelfehler einsetzen —"


def snap(value, spec: SettingSpec):
    """Auf das Raster des Reglers legen und in die Grenzen klemmen (bei einer Auswahl: auf einen gültigen Index)."""
    value = max(spec.lo, min(spec.hi, value))
    if spec.step > 0 and (spec.caster is float or spec.step > 1 or spec.choices):
        value = round((value - spec.lo) / spec.step) * spec.step + spec.lo
        value = round(value, 6) if spec.caster is float else int(value)
    return spec.choices[int(value)] if spec.choices else value


def set_diagram(edges):
    for pair, state in D.states_from_edges(edges).items():
        st.session_state[PAIR_KEYS[pair]] = state


def current_edges() -> frozenset:
    return D.edges_from_states({pair: st.session_state[key] for pair, key in PAIR_KEYS.items()})


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default
    if any(key not in st.session_state for key in PAIR_KEYS.values()):
        set_diagram(K.TRUE_EDGES)
    st.session_state.setdefault("error_select", NO_ERROR)


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt."""
    if "permalink_loaded" in st.session_state:
        return
    init_session_state_defaults()
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param not in qp:
            continue
        try:
            value = snap(spec.caster(float(qp[spec.url_param])), spec)
        except (ValueError, TypeError):
            continue
        st.session_state[state_key] = value
    if DIAGRAM_PARAM in qp:
        try:
            set_diagram(D.decode(int(float(qp[DIAGRAM_PARAM]))))
        except (ValueError, TypeError, OverflowError):
            pass
    st.session_state["permalink_loaded"] = True


def url_value(state_key: str, value):
    spec = SETTING_SPECS[state_key]
    return spec.choices.index(value) if spec.choices else value


def sync_query_params(values: dict, edges):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(url_value(state_key, value))
        st.query_params[DIAGRAM_PARAM] = str(D.encode(edges))
    except Exception:
        pass


def apply_preset(name):
    p = K.PRESETS[name]
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = p[key]
    set_diagram(p["edges"])


def apply_error():
    label = st.session_state.get("error_select", NO_ERROR)
    if label in ERROR_LABELS:
        set_diagram(ERROR_LABELS[label])
    st.session_state["error_select"] = NO_ERROR


def reset_diagram():
    set_diagram(K.TRUE_EDGES)


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, K.SEED_MAX)
