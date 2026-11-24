"""Minimal reportlab wrapper: blocks -> PDF. Text is sanitised to Latin-1 so the built-in Helvetica font renders it."""
import re, os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, Preformatted, KeepTogether

_MAP = {"→": "->", "←": "<-", "≥": ">=", "≤": "<=", "×": "x", "₹": "Rs", "–": "-", "—": "-", "‘": "'", "’": "'",
        "“": '"', "”": '"', "…": "...", "≈": "~", "≠": "!=", "∑": "sum", "√": "sqrt", "σ": "sigma", "θ": "theta", "β": "beta",
        "ε": "eps", "⊙": "(.)", "∈": " in ", "′": "'", "•": "-", " ": " ", "−": "-", "π": "pi", "→": "->", "⋅": "*", "∇": "grad",
        "∥": "||", "∣": "|", "·": "-", "✓": "ok", "✗": "x", "λ": "lambda", "≤": "<="}

ss = getSampleStyleSheet()
BASE = ParagraphStyle("b", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.4, leading=12.6, spaceAfter=4)
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=15, leading=18, spaceBefore=10, spaceAfter=6, textColor=colors.HexColor("#1f3b63"))
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, leading=14, spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#2d5a8e"))
H3 = ParagraphStyle("h3", parent=ss["Heading3"], fontName="Helvetica-Bold", fontSize=10, leading=12, spaceBefore=6, spaceAfter=2)
CELL = ParagraphStyle("c", parent=BASE, fontSize=7.6, leading=9.4, spaceAfter=0)
CELLH = ParagraphStyle("ch", parent=CELL, fontName="Helvetica-Bold", textColor=colors.white)
NOTE = ParagraphStyle("n", parent=BASE, backColor=colors.HexColor("#fff4d6"), borderColor=colors.HexColor("#e0b84c"), borderWidth=0.6, borderPadding=4, spaceBefore=4, spaceAfter=8)
CODE = ParagraphStyle("code", fontName="Courier", fontSize=7.4, leading=9, backColor=colors.HexColor("#f3f3f3"), borderPadding=3, spaceAfter=6)
EQ = ParagraphStyle("eq", parent=CODE, fontName="Courier-Bold", fontSize=8, backColor=colors.HexColor("#eef3fa"), leftIndent=10)


def clean(t):
    t = str(t)
    for k, v in _MAP.items():
        t = t.replace(k, v)
    return t.encode("latin-1", "replace").decode("latin-1")


def fmt(t):
    t = clean(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"`([^`]+)`", r'<font face="Courier">\1</font>', t)
    return t.replace("\n", "<br/>")


def _on_page(title):
    def f(canvas, doc):
        canvas.saveState(); canvas.setFont("Helvetica", 7.5); canvas.setFillColor(colors.grey)
        canvas.drawString(1.6 * cm, 1.0 * cm, clean(title)); canvas.drawRightString(A4[0] - 1.6 * cm, 1.0 * cm, "page %d" % doc.page); canvas.restoreState()
    return f


def make_pdf(path, title, subtitle, blocks, landscape_tables=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.6 * cm, bottomMargin=1.7 * cm, title=clean(title), author="M.Tech Lab CA")
    W = A4[0] - 3.2 * cm
    st = [Spacer(1, 2.2 * cm), Paragraph(fmt(title), ParagraphStyle("t", parent=H1, fontSize=22, leading=26, alignment=0)),
          Spacer(1, 0.3 * cm), Paragraph(fmt(subtitle), ParagraphStyle("s", parent=BASE, fontSize=11, leading=15, textColor=colors.HexColor("#444444"))), Spacer(1, 0.6 * cm)]
    for b in blocks:
        k = b[0]
        if k == "h1": st.append(Paragraph(fmt(b[1]), H1))
        elif k == "h2": st.append(Paragraph(fmt(b[1]), H2))
        elif k == "h3": st.append(Paragraph(fmt(b[1]), H3))
        elif k == "p": st.append(Paragraph(fmt(b[1]), BASE))
        elif k == "note": st.append(Paragraph(fmt(b[1]), NOTE))
        elif k == "bullets":
            for it in b[1]: st.append(Paragraph("&bull; " + fmt(it), ParagraphStyle("bl", parent=BASE, leftIndent=12, firstLineIndent=-8, spaceAfter=2)))
            st.append(Spacer(1, 3))
        elif k == "eq": st.append(Preformatted(clean(b[1]), EQ))
        elif k == "code": st.append(Preformatted(clean(b[1]), CODE))
        elif k == "pagebreak": st.append(PageBreak())
        elif k == "image":
            if os.path.exists(b[1]):
                from reportlab.lib.utils import ImageReader
                iw, ih = ImageReader(b[1]).getSize(); w = min(W, b[2] if len(b) > 2 else W); st.append(Image(b[1], width=w, height=w * ih / iw)); st.append(Spacer(1, 6))
            else:
                st.append(Paragraph(fmt("[chart not available: %s - NOT RUN]" % os.path.basename(b[1])), NOTE))
        elif k == "table":
            hdr, rows = b[1], b[2]
            widths = b[3] if len(b) > 3 and b[3] else [W / len(hdr)] * len(hdr)
            s = sum(widths); widths = [w * W / s for w in widths]
            data = [[Paragraph(fmt(h), CELLH) for h in hdr]] + [[Paragraph(fmt(c), CELL) for c in r] for r in rows]
            t = Table(data, colWidths=widths, repeatRows=1)
            t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2d5a8e")), ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b0b8c4")),
                                   ("VALIGN", (0, 0), (-1, -1), "TOP"), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f8fc")]),
                                   ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3), ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
            st.append(t); st.append(Spacer(1, 8))
        else:
            raise ValueError(k)
    doc.build(st, onFirstPage=_on_page(title), onLaterPages=_on_page(title))
    return path
