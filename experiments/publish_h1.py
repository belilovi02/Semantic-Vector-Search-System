"""Append H1 analysis to the main conclusions file and publish a PDF (or HTML fallback).

Usage: run under the project's venv.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / 'dist' / 'documentation' / 'H1_ANALYSIS.txt'
CONCLUSIONS = ROOT / 'dist' / 'documentation' / 'EXPERIMENTS_CONCLUSIONS.txt'
OUT_PDF = ROOT / 'dist' / 'documentation' / 'H1_ANALYSIS.pdf'
OUT_HTML = ROOT / 'dist' / 'documentation' / 'H1_ANALYSIS.html'

def append_to_conclusions():
    if not ANALYSIS.exists():
        print('Analysis file not found:', ANALYSIS)
        return False
    content = ANALYSIS.read_text(encoding='utf-8')
    header = '\n\n**H1 ANALIZA (automatski dodano)**\n'
    CONCLUSIONS.parent.mkdir(parents=True, exist_ok=True)
    with open(CONCLUSIONS, 'a', encoding='utf-8') as fh:
        fh.write(header)
        fh.write(content)
    print('Appended H1 analysis to', CONCLUSIONS)
    return True

def make_pdf():
    text = ANALYSIS.read_text(encoding='utf-8')
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.pdfgen import canvas
    except Exception as e:
        print('reportlab not available:', e)
        return False

    # try to use a built-in TrueType font if available
    try:
        pdfmetrics.registerFont(TTFont('DejaVuSans', 'DejaVuSans.ttf'))
        font_name = 'DejaVuSans'
    except Exception:
        font_name = 'Helvetica'

    c = canvas.Canvas(str(OUT_PDF), pagesize=A4)
    width, height = A4
    margin = 40
    y = height - margin
    c.setFont(font_name, 10)
    for line in text.splitlines():
        if y < margin:
            c.showPage()
            c.setFont(font_name, 10)
            y = height - margin
        c.drawString(margin, y, line)
        y -= 12
    c.save()
    print('Wrote PDF:', OUT_PDF)
    return True

def make_html():
    text = ANALYSIS.read_text(encoding='utf-8')
    html = '<html><body><pre style="font-family:monospace">' + text + '</pre></body></html>'
    OUT_HTML.write_text(html, encoding='utf-8')
    print('Wrote HTML fallback:', OUT_HTML)
    return True

def main():
    ok = append_to_conclusions()
    if not ok:
        return
    if not make_pdf():
        make_html()

if __name__ == '__main__':
    main()
