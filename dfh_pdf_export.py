"""PDF-Export der aktuellen Auswertung (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und Sonderzeichen werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

import dfh_constants as K
import dfh_graphs as D
from dfh_evaluation import Analysis, de, names_of, signed, verdict_analysis

_REPLACE = {"→": "->", "←": "<-", "–": "-", "—": "-", "−": "-", "„": '"', "“": '"', "×": "x", "·": "*", "**": "", "α": "alpha", "σ": "sigma", "²": "2", "≤": "<=", "≥": ">=", "⊥": "_|_", "±": "+/-", "✗": "x", "…": "..."}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis) -> bytes:
    s = a.settings
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Wie falsch darf das Diagramm sein?"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(f"Welt der Linie mit sechs Größen (S, Z, X, Y, W, C), wahre Wirkung X -> Y = {de(K.EFFECT)}. Gewählte Regel: {s.rule}; Toleranz {de(s.tol)}; {s.n} Beobachtungen, Seed {s.seed}. Alles simuliert."),
                   new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, _s("Das gezeichnete Diagramm"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 4.6, _s("Pfeile: " + (", ".join(D.name(e) for e in sorted(s.edges)) or "keine") + f". Abweichungen vom wahren Diagramm: {a.errors}."), new_x="LMARGIN", new_y="NEXT")
    for kind, edges in a.diff.items():
        for e in edges:
            pdf.multi_cell(0, 4.6, _s(f"  {kind}: {D.name(e)}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, _s("Die fünf Regeln"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    for r in a.rows:
        pdf.multi_cell(0, 4.6, _s(f"{r.rule}: Menge {names_of(r.adjust)}, gültig im wahren Diagramm: {'ja' if r.valid_true else 'nein'}, Bias {signed(r.bias, 3)}, innerhalb der Toleranz: {'ja' if r.harmless else 'nein'}."),
                       new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 4.5, _s(verdict_analysis(a)[1]), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 9)
    verworfen = "ja" if a.rejected else "nein"
    pdf.multi_cell(0, 4.6, _s(f"Prüfung an den Daten: {len(a.markov)} lokale Markov-Aussagen (Fisher-z, Bonferroni), mindestens eine verworfen: {verworfen}. Markov-Klasse des Diagramms: {a.class_size} Mitglieder, das wahre gehört dazu: {'ja' if a.same_class else 'nein'}."),
                   new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Quelle: diagrammfehler-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
