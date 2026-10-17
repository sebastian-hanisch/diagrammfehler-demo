# Wie falsch darf das Diagramm sein? – Streamlit-Demo

Sechzehntes Stück der Linie **Kausale Diagramme und Adjustierung** in der „Konzepte“-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) (Operations Research und Machine Learning). Nachfolger von [lingam-icp-demo](https://github.com/sebastian-hanisch/lingam-icp-demo): Die ganze Linie hat angenommen, dass das gezeichnete Diagramm stimmt. Vier Vorgänger haben die Frage ausdrücklich hierher verschoben; hier wird das Diagramm falsch gezeichnet und der Preis gemessen.

## Kernfrage

Was kostet es, wenn im Diagramm ein **Pfeil fehlt**, **verkehrt herum** zeigt oder **zu viel** gezeichnet ist? Gemessen wird der **Bias** der geschätzten Wirkung des Faktors auf den Servicegrad, **exakt** aus der Kovarianz der Welt (kein Zufall), bei **fünf Regeln**, aus dem gezeichneten Diagramm eine Anpassungsmenge zu bilden: die Eltern des Faktors, alle Nicht-Nachkommen, die kleinste gültige Menge, alle übrigen Größen, nichts. Dazu: **Sieht man den Fehler an den Daten?** Die meisten Fehler sind harmlos; der schädliche ist ausgerechnet der, den die Daten nicht immer zeigen.

## Modell

- **Welt:** die der Linie mit sechs Größen: Saison S → Nachfrage Z → Faktor X → Servicegrad Y, dazu Z → Y, Wetter W → Y und die Beschwerden C als Wirkung des Servicegrads (Y → C); Pfeilstärken 0,8 / 0,8 / 0,6 / 0,6 / 0,6 / 0,8, Einheitsrauschen. Wahre Wirkung X → Y = **0,6**.
- **Bias:** Koeffizient von X in der Regression von Y auf X und die Menge, aus der Bevölkerungskovarianz (I − B)⁻¹(I − B)⁻ᵀ, minus 0,6. **Gültig im wahren Diagramm ⇔ Bias null** (Identität dieser Welt, für alle 16 Mengen geprüft).
- **Fünf Regeln:** Eltern (ohne Y), Nicht-Nachkommen, kleinste gültige Menge unter den Nicht-Nachkommen, alle anderen, nichts.
- **Das gezeichnete Diagramm:** ein Baukasten über alle 15 Paare mit drei Zuständen (kein Pfeil, →, ←); Kreise werden erkannt und abgelehnt. Einzelfehler lassen sich einsetzen (23 azyklische Möglichkeiten).
- **Toleranz:** Regler von 0,02 bis 0,20 (voreingestellt 0,06 = 10 % der Wirkung); ein Diagramm ist „harmlos“, wenn |Bias| unter der Toleranz liegt.
- **Prüfung an den Daten:** lokale Markov-Aussagen des gezeichneten Diagramms (jede Größe ist von ihren Nicht-Nachkommen ohne Eltern unabhängig, gegeben ihre Eltern), Fisher-z-Test auf der Teilkorrelation, Bonferroni über alle Aussagen (0,05 gesamt).

## Methodik

- **Exakt statt gewürfelt** (`dfh_world.py`): Der Bias jeder Menge folgt aus der Kovarianzmatrix; Experimente 1 und 2 haben keinen Zufall.
- **Gültigkeit** (`dfh_graphs.py`): Backdoor-Kriterium im gezeichneten Diagramm (kein Nachkomme von X; X und Y getrennt im Diagramm ohne die Pfeile aus X), mit `dsep_graph.py` aus Stück 2. Die Regeln werden im gezeichneten Diagramm gebildet, der Bias wird in der wahren Welt bezahlt.
- **Experimente 3 und 4** laufen über feste SplitMix64-Ströme; „verworfen“ heißt: in mindestens der Hälfte der Datensätze wird eine lokale Markov-Aussage verworfen.
- **Fehlerformeln:** fehlender Pfeil Z → X der Stärke a, nichts angepasst: Bias = 0,6 · 1,64 a / (1,64 a² + 1); verborgene Ursache (Stärken a auf X, b auf Y) nach Anpassung an Z: Bias = a b / (1 + a²).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py`. Experimente 1 und 2 sind exakt (Rundung auf vier Stellen); die Läufe der Experimente 3 und 4 sind fest, ihre Zahl steht dabei, Raten sind auf etwa ±3 bis 6 Punkte genau.

| Frage | Befund | Test |
|---|---|---|
| Handwerte | Nichts angepasst: Bias **0,3841** (= 0,6 · 1,312 / 2,0496); fehlender Pfeil Z → X der Stärke 0,2: **0,1847**; verborgene Ursache (0,5; 0,5): **0,2**, (0,2; 0,3): **0,0577**; alle anderen angepasst: **−0,2341** (= 0,6 / 1,64 − 0,6); Markov-Klasse des wahren Diagramms: **3** Diagramme. | `test_claim_hand_instances` |
| Gültigkeit und Bias | Von den 16 Mengen aus {S, Z, W, C} sind genau vier im wahren Diagramm gültig ({Z}, {S, Z}, {Z, W}, {S, Z, W}), und genau diese haben Bias null. **Identität dieser Welt**, kein Befund. | `test_claim_valid_in_the_true_diagram_iff_bias_zero_is_an_identity_of_this_world` |
| Das richtige Diagramm | Eltern, Nicht-Nachkommen und kleinste gültige Menge: Bias **0**; „alle anderen“: **−0,2341** (C ist Nachkomme von X); „nichts“: **+0,3841**. | `test_claim_the_true_diagram_with_each_rule` |
| Alle **23 Einzelfehler** (6 fehlend, 5 umgedreht, 12 zu viel; Toleranz 0,06) | Harmlos bei der Regel **Eltern 21 (91 %)**, kleinste gültige 19, Nicht-Nachkommen 18, „alle anderen“ 0, „nichts“ 0. Eltern: fehlend 5 von 6, umgedreht 4 von 5, zu viel **12 von 12**. Schädlich sind nur der **fehlende** und der **umgedrehte Pfeil Z → X** (Bias +0,384). | `test_claim_single_errors_exact` |
| Die Toleranz | Die Biaswerte häufen sich bei 0, 0,234, 0,293 und 0,384: Für die Einzelfehler ändert der Regler zwischen 0,02 und 0,20 nichts (Eltern 21, Nicht-Nachkommen 18); erst bei 0,30 gilt alles als harmlos. | `test_claim_the_biases_cluster_so_the_tolerance_matters_little_for_single_errors` |
| Wo die anderen Regeln scheitern | Kleinste gültige Menge: schädlich auch beim fehlenden Z → Y und beim umgedrehten X → Y (Bias +0,384, die Eltern bleiben gültig). Nicht-Nachkommen: schädlich bei fehlendem oder umgedrehtem X → Y und Y → C (−0,234, C gilt dann nicht als Nachkomme) und beim umgedrehten Z → X (+0,293). | `test_claim_minimal_and_non_descendants_fail_where_eltern_do_not` |
| Zusätzliche Pfeile | **Nie schädlich** (12 von 12 bei Eltern). **Eigenschaft dieser Welt:** Ein zusätzlicher Pfeil in den Faktor kann nur von S oder W kommen (Y → X und C → X schlössen einen Kreis), beide sind Nicht-Nachkommen; es gibt keinen Sammelknoten vor dem Faktor. Kein Befund über Diagramme im Allgemeinen. | `test_claim_extra_arrows_are_harmless_in_this_world_only` |
| Markov-Klasse | Die Klasse des wahren Diagramms hat 3 Mitglieder mit Bias **0, 0 und +0,384**: Das Nachbardiagramm (Nachfrage → Saison, Faktor → Nachfrage) hat dieselben Unabhängigkeitsaussagen, wird von den Daten nie verworfen und schadet. | `test_claim_markov_neighbour_has_the_same_statements_and_a_harmful_member` |
| Wie stark darf der Fehler sein? | Fehlt Z → X, ist das Weglassen nur bis zur Stärke **0,061** harmlos (Toleranz 0,06; bei 0,02: 0,020; bei 0,20: 0,219; in der Welt 0,8). Eine verborgene Ursache bleibt in **11 von 25** Stärkepaaren (a, b ∈ {0,1; 0,2; 0,3; 0,5; 0,8}) unter 0,06. | `test_claim_tolerance_thresholds_for_a_missing_arrow_and_a_hidden_cause` |
| Sieht man den Fehler? (alle **284** Diagramme im Abstand 1 bis 2, **40 Datensätze**, n = 1 000, Eltern) | **51** schädlich, davon **48** (46 bis 50) verworfen, **3** nicht (Rate 0; siehe die nächste Zeile). Von 233 harmlosen werden **144 (62 %)** trotzdem verworfen: Die Daten prüfen das Diagramm, nicht den Bias. Beide schädlichen Einzelfehler werden in mindestens 95 % der Datensätze verworfen; der Fehlalarm bei einem zusätzlichen Pfeil liegt bei höchstens 10 %. | `test_claim_detection` |
| Was die Daten verwerfen | Ein Diagramm wird verworfen, wenn es eine **Unabhängigkeit behauptet, die in der Welt falsch ist**: Alle **192** Diagramme, die das tun, werden in mindestens 95 % der Datensätze verworfen. Die **92** übrigen behaupten nur wahre Unabhängigkeiten (Mitglieder der Markov-Klasse und dichtere Diagramme mit mehr Pfeilen): Sie werden höchstens im Rahmen des Fehlalarms (unter 10 %) verworfen. Alle unentdeckten schädlichen Diagramme liegen unter diesen 92. | `test_claim_the_data_reject_the_diagrams_that_claim_a_false_independence` |
| Andere Regeln, gleicher Versuch | Schädlich: Nicht-Nachkommen **111**, kleinste gültige **88** von 284; unentdeckt je 2 bis 6. | `test_claim_detection_of_other_rules` |
| **Mehrere Fehler** (1 bis 4 zufällige, je 300 Diagramme) | Harmloser Anteil bei Eltern **87 / 83 / 79 / 68 %**, kleinste gültige 78 / 71 / 63 / 54 %, Nicht-Nachkommen 72 / 64 / 54 / 44 %; „alle anderen“ und „nichts“ 0 % (die Menge hängt nicht vom Diagramm ab: Konstruktion). | `test_claim_multiple_errors` |
| Streuung gegen Bias (n = 1 000, **300 Datensätze**) | Richtiges Diagramm RMSE **0,037** (n = 200: 0,088); fehlendes Z → X **0,384** (Mittel 0,983); zusätzliches W → X **0,030**: ein Fehler, der die Streuung senkt, weil W den Servicegrad miterklärt (Faktor 1/√1,36 ≈ 0,86); zusätzliches S → X 0,037. | `test_claim_variance_against_bias` |

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Zusätzliche Pfeile sind harmlos** | Gilt in dieser Welt, weil kein Sammelknoten (Collider) vor dem Faktor steht; mit einem solchen Knoten (M-Bias) schadet gerade das Anpassen an ihn. Die Welt hat keinen. | nicht gebaut |
| **Linear, Gauß'sches Rauschen** | Der Bias wird exakt aus der Kovarianz gerechnet; nichtlineare Wirkungen und die Schätzer der Stücke 6 und 7 (doppelt robust) sind nicht Teil dieses Stücks. | Welches Verfahren, gemessen woran? |
| **Alle Größen beobachtet** | Eine verborgene Ursache wird nur als Stärketabelle behandelt (Experiment 2), nicht als gezeichneter Knoten. | Wann geht es gar nicht? |
| **Fehler stammen aus Einzelschritten** | Experimente 3 und 4 verwenden Diagramme in kleinem Abstand zur Wahrheit; gelernte Diagramme (PC, LiNGAM) als Fehlerquelle sind nicht gebaut. | nicht gebaut |
| **Die Daten prüfen das Diagramm, nicht den Bias** | Ein Diagramm, das nur wahre Unabhängigkeiten behauptet (die Markov-Klasse des wahren, dichtere Diagramme), wird nie verworfen und kann doch schädlich sein; viele harmlose Diagramme werden verworfen (Experiment 3). | Welches Verfahren, gemessen woran? |
| **Eine Wirkung, eine Welt** | Das Ergebnis gilt für diese sechs Größen und ihre Koeffizienten; die Toleranz ist ein Maßstab des Nutzers, kein Naturgesetz. | nicht gebaut |

Was **Identität, Konstruktion oder Eigenschaft der Welt** ist, steht in `test_claims.py` als solches: gültig im wahren Diagramm ⇔ Bias null (Identität), „alle anderen“ und „nichts“ (die Menge hängt nicht vom Diagramm ab), „zusätzliche Pfeile schaden nie“ (Eigenschaft dieser Welt). Der Anteil der Diagramme, die „unentdeckt“ bleiben, hängt an Datensatzgröße (n = 1 000), Niveau (Bonferroni, 0,05) und der Schwelle „verworfen in mindestens der Hälfte“; die Aussage ist die Richtung (schädlich und unsichtbar gibt es), nicht die Dezimale.
Das Backdoor-Kriterium (Pearl 2009), die Markov-Klassen (Verma/Pearl 1990) und die Fisher-z-Prüfung sind aus dem Gedächtnis geschrieben und nicht gegen die Quellen geprüft; sie sind belegt durch die Orakel in `tests/test_oracle_bias.py` (Simulation, scipy, Aufzählung der Diagramme).

## Bezug zu den Nachbarn im Portfolio

- [backdoor-demo](https://github.com/sebastian-hanisch/backdoor-demo) (Stück 3), [schlechte-kontrollen-demo](https://github.com/sebastian-hanisch/schlechte-kontrollen-demo) (Stück 4), [adjustierungsmenge-demo](https://github.com/sebastian-hanisch/adjustierungsmenge-demo) (Stück 5) und [identifizierbarkeit-demo](https://github.com/sebastian-hanisch/identifizierbarkeit-demo) (Stück 10): setzen das Diagramm voraus und verweisen in ihren Grenzen auf dieses Stück. Das Backdoor-Kriterium, `dsep_graph.py` und die Regeln der Mengenwahl kommen hier als Werkzeug.
- [sensitivitaet-demo](https://github.com/sebastian-hanisch/sensitivitaet-demo) (Stück 11): wie stark eine **verborgene** Ursache sein müsste; hier der Fehler im **gezeichneten** Diagramm, die verborgene Ursache nur als Stärketabelle.
- [strukturlernen-demo](https://github.com/sebastian-hanisch/strukturlernen-demo) (Stück 13) und [lingam-icp-demo](https://github.com/sebastian-hanisch/lingam-icp-demo) (Stück 15): gelernte Diagramme sind eine Quelle falscher Pfeile; hier die Gegenseite, was ein gegebener Fehler der Adjustierung antut (`stl_pc.py` wortgleich, für die Markov-Klassen).
- [d-separation-demo](https://github.com/sebastian-hanisch/d-separation-demo) (Stück 2): testbare Aussagen eines Diagramms; hier als Falsifikation der Annahme.

## Tests

Hand-Instanzen (Bias 0,3841, 0,1847, 0,2, 0,0577, −0,2341; Rechnungen von Hand für die Regression), der exakte Bias gegen eine Simulation mit 400 000 Zeilen für alle 16 Mengen, „gültig ⇔ Bias null“ für alle Regeln in allen 285 Diagrammen im Abstand 2, die Zahl der azyklischen Diagramme (1, 3, 25, 543, 29 281) durch Aufzählung, Fisher-z gegen scipy, Einzelfehler, Regeln, Permalink (Diagramm als Zahl zur Basis 3), Auswertung, Verdikte, jede README-Zahl in `test_claims.py`, AppTest (Presets, Baukasten, Kreise, Regler, Experimente, Permalink, PDF), Footer und Laufzeit-Importe.

```bash
python -m pytest tests -q
```

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `kausal_generator.py` | gemeinsamer Generator der Linie (SplitMix64, Normalverteilung, Strukturmodell) |
| `dsep_graph.py` | d-Separation, lokale Markov-Basis (wortgleich aus Stück 2) |
| `stl_pc.py` | Fisher-z, Teilkorrelation, Markov-Klassen (wortgleich aus Stück 13) |
| `dfh_world.py` | Bevölkerungskovarianz, exakter Bias, Daten, Fehlerformeln |
| `dfh_graphs.py` | Diagramme aus Pfeilmengen, Gültigkeit, die fünf Regeln, Einzelfehler, Abstand, Adresszahl |
| `dfh_evaluation.py` | Einstellungen, Analyse, vier Experimente, Verdikte |
| `dfh_visualization.py`, `dfh_pdf_export.py` | Figuren, Tabellen, PDF |
| `dfh_constants.py`, `dfh_presets.py` | Regler, Regeln, Experimente, Presets, Permalink |
| `tests/` | Hand-Instanzen, Orakel, Claims, App |

## Bewusst nicht umgesetzt

- **Gelernte Diagramme (PC, LiNGAM) als Fehlerquelle:** Die Fehler sind hier gegebene Einzelschritte; die gelernten stehen in Stück 13 und 15.
- **Nichtlineare Wirkungen, Schätzer mit maschinellem Lernen, verborgene Knoten im Diagramm:** nur im Grenzen-Absatz bzw. als Stärketabelle.
- **Praxisdaten:** nicht gebaut; alle Zahlen sind erzeugt.

## Lokal ausführen

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit Cloud: Python 3.12.

Gebaut mit Streamlit, Plotly, numpy, pandas und fpdf2.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
