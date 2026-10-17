"""Wie falsch darf das Diagramm sein? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Sechzehntes Stück der Linie Kausale Diagramme und Adjustierung. Die Linie hat das Diagramm als gegeben genommen; hier wird es falsch gezeichnet: ein Pfeil fehlt, zeigt verkehrt herum oder ist zu viel.
Gemessen wird der Bias der geschätzten Wirkung, exakt aus der Kovarianz der Welt, bei fünf Regeln, aus dem Diagramm eine Anpassungsmenge zu bilden, dazu was die Daten über das Diagramm verraten.
Kern: dfh_world.py, dfh_graphs.py, dfh_evaluation.py; d-Separation und Markov-Klassen: dsep_graph.py, stl_pc.py (wortgleich aus Stück 2 und 13).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import dfh_constants as K
import dfh_graphs as D
from dfh_evaluation import (
    Settings,
    analyse,
    de,
    detection_experiment,
    multi_error_experiment,
    names_of,
    rmse_experiment,
    signed,
    single_error_experiment,
    strength_experiment,
    verdict_analysis,
    verdict_detection,
    verdict_multi,
    verdict_single,
    verdict_strength,
)
from dfh_pdf_export import build_pdf
from dfh_presets import (
    ERROR_LABELS,
    NO_ERROR,
    PAIR_KEYS,
    apply_error,
    apply_preset,
    bounds,
    current_edges,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    reset_diagram,
    sync_query_params,
)
from dfh_visualization import (
    build_detection_chart,
    build_detection_table,
    build_differences_table,
    build_drawn_diagram,
    build_markov_table,
    build_multi_chart,
    build_multi_tables,
    build_rules_chart,
    build_rules_table,
    build_single_chart,
    build_single_kind_table,
    build_single_table,
    build_strength_chart,
    build_strength_tables,
    build_summary_table,
    build_true_diagram,
    style_rules_table,
    style_single_table,
)

st.set_page_config(page_title="Wie falsch darf das Diagramm sein? – Sebastian Hanisch", layout="wide")

SPINNER = "Experiment rechnet über feste Zufallsströme …"


@st.cache_data(show_spinner=False)
def _analyse(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _exp1():
    return single_error_experiment()


@st.cache_data(show_spinner=False)
def _exp2():
    return strength_experiment()


@st.cache_data(show_spinner=SPINNER)
def _exp3():
    return detection_experiment()


@st.cache_data(show_spinner=SPINNER)
def _exp4():
    return multi_error_experiment(), rmse_experiment()


def verdict(kind_text):
    kind, text = kind_text
    getattr(st, kind)(text)


def experiment_button(label, key):
    if st.button(label, key=f"{key}_run"):
        st.session_state[f"{key}_ready"] = True
    return st.session_state.get(f"{key}_ready", False)


st.title("📐 Wie falsch darf das Diagramm sein?")
st.markdown(
    """
**Die ganze Linie hat angenommen, dass das Diagramm stimmt.** Dies ist das sechzehnte Stück der Linie **Kausale Diagramme und Adjustierung**. Fünf Vorgänger haben die Frage ausdrücklich hierher verschoben: Was kostet es, wenn ein **Pfeil fehlt**, **verkehrt herum** zeigt oder **zu viel** gezeichnet ist?
Gemessen wird der **Bias** der geschätzten Wirkung des Faktors auf den Servicegrad, **exakt** aus der Kovarianz der Welt (kein Zufall), bei fünf Regeln, aus dem gezeichneten Diagramm eine Anpassungsmenge zu bilden.
Der Befund ist unbequem: Die meisten Fehler sind harmlos, aber **der eine schädliche ist genau der, den die Daten nicht immer zeigen**.

Weiter unten: vier Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So liest man das Ergebnis", expanded=True):
    st.markdown(
        """
1. **Die Welt** (links): die der Linie mit sechs Größen: Saison S → Nachfrage Z → Faktor X → Servicegrad Y, dazu Z → Y, Wetter W → Y und die Beschwerden C als Wirkung des Servicegrads. Die wahre Wirkung X → Y ist 0,6.
2. **Das gezeichnete Diagramm** (rechts) baut man unten Paar für Paar: **grau** richtig, **orange** umgedreht, **rot** zu viel, **grau gestrichelt** fehlend. Blau gefüllte Knoten sind die Größen, an die die gewählte Regel anpasst.
3. **Fünf Regeln** machen aus dem Diagramm eine Menge: die Eltern des Faktors, alle Nicht-Nachkommen, die kleinste gültige Menge, alle anderen Größen, nichts. Die Balken zeigen den **exakten Bias** je Regel; grün liegt innerhalb der Toleranz (Seitenleiste).
4. **Gültig heißt zweierlei:** im gezeichneten Diagramm (so wurde die Menge gewählt) und im wahren (so wird sie bezahlt). Nur eine im wahren Diagramm gültige Menge gibt Bias null.
5. **Die Daten** prüfen die lokalen Markov-Aussagen des gezeichneten Diagramms. Ein Diagramm derselben Markov-Klasse wie das wahre wird nie verworfen, kann aber falsch anpassen.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
names = list(K.PRESETS)
for chunk in (names[:4], names[4:]):
    cols = st.columns(4)
    for col, name in zip(cols, chunk):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=K.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Regel**")
    rule = st.radio("Regel zur Wahl der Anpassungsmenge", K.RULES, key="rule_radio", help="Wie aus dem gezeichneten Diagramm eine Menge von Größen wird, an die angepasst wird. " + " ".join(f"{r}: {K.RULE_HELP[r]}" for r in K.RULES))
    tol = st.slider("Tolerierter Bias", *bounds("tol_slider"), step=K.TOL_STEP, key="tol_slider", help="Bis zu diesem Betrag (in Einheiten der Wirkung 0,6) gilt der Bias als harmlos. 0,06 sind 10 % der Wirkung.")
    st.markdown("**Daten**")
    n = st.slider("Beobachtungen", *bounds("n_slider"), step=K.N_STEP, key="n_slider", help="Zeilen der Stichprobe für Schätzung und Prüfung an den Daten.")
    seed = st.number_input("Zufalls-Seed der Stichprobe", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Stichprobe würfeln", width="stretch", on_click=randomize_seed)

with st.expander("✏️ Das gezeichnete Diagramm bearbeiten", expanded=True):
    c_a, c_b = st.columns([3, 1])
    with c_a:
        st.selectbox("Einzelfehler vom wahren Diagramm einsetzen", [NO_ERROR, *ERROR_LABELS], key="error_select", on_change=apply_error, help="Setzt das wahre Diagramm mit genau diesem einen Fehler ein (23 azyklische Möglichkeiten).")
    with c_b:
        st.button("↺ Wahres Diagramm", width="stretch", on_click=reset_diagram, key="reset_truth")
    cols = st.columns(3)
    for i, ((a_, b_), key) in enumerate(PAIR_KEYS.items()):
        with cols[i % 3]:
            st.selectbox(f"{K.NAMES[a_]} – {K.NAMES[b_]}  ({K.LONG[a_]}, {K.LONG[b_]})", (0, 1, 2), key=key, format_func=lambda s, a_=a_, b_=b_: ("kein Pfeil", f"{K.NAMES[a_]} → {K.NAMES[b_]}", f"{K.NAMES[a_]} ← {K.NAMES[b_]}")[s])

edges = current_edges()
sync_query_params({"rule_radio": rule, "tol_slider": round(float(tol), 2), "n_slider": int(n), "seed_input": int(seed)}, edges)

if not D.is_acyclic(edges):
    st.error("Das gezeichnete Diagramm enthält einen Kreis: Ein Diagramm für Ursachen muss azyklisch sein. Ändern Sie einen Pfeil im Baukasten oben, die Auswertung folgt dann.")
    st.stop()

settings = Settings(edges, rule, round(float(tol), 2), int(n), int(seed))
a = _analyse(settings)

# --- Ergebnis -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Das gezeichnete Diagramm gegen die Welt")
(c1, c2), (c3, c4) = st.columns(2), st.columns(2)
c1.metric("Fehler im Diagramm", str(a.errors), delta=f"fehlt {len(a.diff['fehlt'])}, umgedreht {len(a.diff['umgedreht'])}, zu viel {len(a.diff['zu viel'])}", delta_color="off", help="Abweichungen vom wahren Diagramm der Welt.")
c2.metric("Menge nach der Regel", names_of(a.chosen.adjust), delta=f"gültig im wahren Diagramm: {'ja' if a.chosen.valid_true else 'nein'}", delta_color="off", help="An diese Größen passt die gewählte Regel im gezeichneten Diagramm an.")
c3.metric("Bias (exakt)", signed(a.chosen.bias, 3), delta=f"Toleranz ± {de(settings.tol)}", delta_color="off", help="Exakter Wert aus der Kovarianz der Welt: Koeffizient von X nach der Anpassung minus wahre Wirkung 0,6.")
c4.metric("Schätzung aus den Daten", de(a.estimate), delta=f"± {de(1.96 * a.se)} (wahr {de(K.EFFECT)})", delta_color="off", help="Koeffizient von X in der linearen Regression von Y auf X und die Menge, mit 95-%-Intervall.")

left, right = st.columns(2)
with left:
    st.plotly_chart(build_true_diagram(), width="stretch", key="true_chart")
    st.caption("Das wahre Diagramm. X (grün umrandet) ist der Faktor, Y (orange umrandet) der Servicegrad, C die Beschwerden.")
with right:
    st.plotly_chart(build_drawn_diagram(a), width="stretch", key="drawn_chart")
    st.caption("Das gezeichnete Diagramm, gefärbt nach dem Urteil gegen das wahre. Blau gefüllt: die Menge der gewählten Regel.")
verdict(verdict_analysis(a))

st.markdown("### Die fünf Regeln nebeneinander")
st.plotly_chart(build_rules_chart(a), width="stretch", key="rules_chart")
st.dataframe(style_rules_table(build_rules_table(a)), hide_index=True, width="stretch")
st.caption("Der Bias ist exakt (Bevölkerung), nicht aus der Stichprobe. Schadet ein Pfeil, steht es in der Spalte „gültig im wahren Diagramm“: Nur eine dort gültige Menge gibt Bias null.")

st.markdown("### 🔍 Was die Daten über das Diagramm sagen")
verdict(("info", (f"Von {len(a.markov)} lokalen Markov-Aussagen des gezeichneten Diagramms (Fisher-z, Bonferroni, 0,05 gesamt) wird mindestens eine verworfen: {'ja' if a.rejected else 'nein'}. "
                  f"Das Diagramm hat {a.class_size} Mitglieder in seiner Markov-Klasse; das wahre gehört dazu: {'ja, die Daten können sie nie unterscheiden' if a.same_class else 'nein'}.")))
st.dataframe(build_markov_table(a), hide_index=True, width="stretch")
st.dataframe(build_differences_table(a), hide_index=True, width="stretch")
st.dataframe(build_summary_table(a), hide_index=True, width="stretch")

if st.button("📄 PDF der aktuellen Auswertung erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="diagrammfehler.pdf", mime="application/pdf", key="pdf_download")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Alle 23 Einzelfehler")
st.caption("Jeder azyklische Einzelfehler des wahren Diagramms (6 fehlende, 5 umgedrehte, 12 zusätzliche Pfeile), je Regel der exakte Bias. Kein Zufall, keine Stichprobe; die Toleranz stellt die Seitenleiste.")
res1 = _exp1()
st.plotly_chart(build_single_chart(res1, settings.tol), width="stretch", key="single_chart")
verdict(verdict_single(res1, settings.tol))
st.dataframe(build_single_kind_table(res1, settings.tol), hide_index=True, width="stretch")
st.dataframe(style_single_table(build_single_table(res1, settings.tol), settings.tol), hide_index=True, width="stretch")

st.subheader("🔬 Experiment 2: Wie stark darf der Fehler sein?")
st.caption("Ein fehlender Pfeil ist harmlos, solange er schwach ist; eine verborgene gemeinsame Ursache von Faktor und Servicegrad gibt nach Anpassung an die Nachfrage den Bias a b / (1 + a²). Exakt, keine Stichprobe.")
res2 = _exp2()
st.plotly_chart(build_strength_chart(res2, settings.tol), width="stretch", key="strength_chart")
verdict(verdict_strength(res2, settings.tol))
for table in build_strength_tables(res2, settings.tol):
    st.dataframe(table, hide_index=True, width="stretch")

st.subheader("🔬 Experiment 3: Sieht man den Fehler an den Daten?")
st.caption(f"Alle 284 Diagramme im Abstand 1 und 2 vom wahren, {K.EXP_DETECT_REPS} feste Datensätze mit n = {K.EXP_DETECT_N}; „verworfen“ heißt: in mindestens der Hälfte der Datensätze wird eine lokale Markov-Aussage verworfen (Fisher-z, Bonferroni). Regel und Toleranz wie links.")
if experiment_button("▶ Experiment 3 rechnen (etwa 5 Sekunden)", "exp3"):
    res3 = _exp3()
    st.plotly_chart(build_detection_chart(res3, settings.tol, rule), width="stretch", key="detection_chart")
    verdict(verdict_detection(res3, settings.tol, rule))
    st.dataframe(build_detection_table(res3, settings.tol, rule), hide_index=True, width="stretch")

st.subheader("🔬 Experiment 4: Mehrere Fehler und die Streuung")
st.caption(f"{K.EXP_MULTI_SAMPLES} zufällige Diagramme mit 1 bis 4 Fehlern nacheinander (azyklisch); dazu die Streuung der Schätzung ({K.EXP_RMSE_REPS} feste Datensätze bei n = {K.EXP_RMSE_NS[0]} und {K.EXP_RMSE_NS[1]}), Regel Eltern.")
if experiment_button("▶ Experiment 4 rechnen (etwa 5 Sekunden)", "exp4"):
    res4, rmse4 = _exp4()
    st.plotly_chart(build_multi_chart(res4, settings.tol), width="stretch", key="multi_chart")
    verdict(verdict_multi(res4, rmse4, settings.tol))
    for table in build_multi_tables(res4, rmse4, settings.tol):
        st.dataframe(table, hide_index=True, width="stretch")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Zusätzliche Pfeile sind harmlos** | Gilt in dieser Welt, weil kein Sammelknoten (Collider) vor dem Faktor steht; mit einem solchen Knoten (M-Bias) schadet gerade das Anpassen an ihn. Die Welt hat keinen. | nicht gebaut |
| **Linear, Gauß'sches Rauschen** | Der Bias wird exakt aus der Kovarianz gerechnet; nichtlineare Wirkungen und die Schätzer der Stücke 6 und 7 (doppelt robust) sind nicht Teil dieses Stücks. | Welches Verfahren, gemessen woran? |
| **Alle Größen beobachtet** | Eine verborgene Ursache wird nur als Stärketabelle behandelt (Experiment 2), nicht als gezeichneter Knoten. | Wann geht es gar nicht? |
| **Fehler stammen aus Einzelschritten** | Experimente 3 und 4 verwenden Diagramme in kleinem Abstand zur Wahrheit; gelernte Diagramme (PC, LiNGAM) als Fehlerquelle sind nicht gebaut. | nicht gebaut |
| **Die Daten prüfen das Diagramm, nicht den Bias** | Ein Diagramm, das nur wahre Unabhängigkeiten behauptet (die Markov-Klasse des wahren, dichtere Diagramme), wird nie verworfen und kann doch schädlich sein; viele harmlose Diagramme werden verworfen (Experiment 3). | Welches Verfahren, gemessen woran? |
| **Eine Wirkung, eine Welt** | Das Ergebnis gilt für diese sechs Größen und ihre Koeffizienten; die Toleranz ist ein Maßstab des Nutzers, kein Naturgesetz. | nicht gebaut |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** Lineares Gauß-Modell $x = Bx + e$ mit Einheitsrauschen; Kovarianz $\Sigma = (I-B)^{-1}(I-B)^{-\top}$. Nach Anpassung an die Menge $A$ ist der Koeffizient von $X$ in der Regression von $Y$ auf $(X, A)$ gleich dem ersten Eintrag von
$\Sigma_{(X,A),(X,A)}^{-1}\,\Sigma_{(X,A),Y}$; der **Bias** ist dieser Wert minus der wahren Wirkung $0{,}6$. Das ist exakt, nicht geschätzt.

**Gültigkeit.** Eine Menge $A$ ist im Diagramm $G$ gültig (Backdoor-Kriterium), wenn sie keinen Nachkommen von $X$ enthält und $X \perp Y \mid A$ in $G$ ohne die Pfeile aus $X$ gilt. In der Welt gilt: $A$ gültig im wahren Diagramm $\Leftrightarrow$ Bias $= 0$ (für alle 16 Mengen geprüft).

**Regeln.** Eltern: $\mathrm{Pa}_G(X)\setminus\{Y\}$; Nicht-Nachkommen: alle Größen außer $X, Y$ und $\mathrm{De}_G(X)$; kleinste gültige Menge unter den Nicht-Nachkommen; alle anderen; nichts.

**Fehlender Pfeil.** Hat $Z \to X$ in Wahrheit die Stärke $a$ und wird nichts angepasst, ist $\mathrm{Bias} = 0{,}6\,\dfrac{1{,}64\,a}{1{,}64\,a^2 + 1}$ (denn $\mathrm{Var}(Z) = 1{,}64$); harmlos bis $a \approx 0{,}061$ bei Toleranz $0{,}06$.
**Verborgene Ursache.** Eine unbeobachtete Ursache $U$ (Varianz 1) mit Stärke $a$ auf $X$ und $b$ auf $Y$ gibt nach Anpassung an $Z$ den Bias $\dfrac{a\,b}{1 + a^2}$.

**Prüfung an den Daten.** Die lokalen Markov-Aussagen von $G$ ($v \perp u \mid \mathrm{Pa}(v)$ für jeden Nicht-Nachkommen $u$ von $v$) werden mit dem Fisher-z-Test auf der Teilkorrelation geprüft, $z = \sqrt{n-k-3}\,\operatorname{artanh}(r)$, Bonferroni über alle Aussagen.
Zwei Diagramme mit gleichen v-Strukturen und gleichem Skelett (Markov-Klasse) haben dieselben Aussagen: Die Daten unterscheiden sie nicht.

**Literatur.** J. Pearl, *Causality*, 2009 (Backdoor-Kriterium); T. Verma, J. Pearl, *Equivalence and synthesis of causal models*, UAI 1990; S. Greenland, J. Pearl, J. Robins, *Causal diagrams for epidemiologic research*, Epidemiology 1999; J. Pearl, *Myth, confusion, and science in causal analysis*, 2009 (alle aus dem Gedächtnis, nicht geprüft).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
