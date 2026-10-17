"""Build the one-page submission-details Word document (black and white only).
Run from the repo root:  python scripts/make_submission.py
"""

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE
from docx.shared import Inches, Pt, RGBColor

OUT = "Unmesh_230911030_FGAI_Assignment2.docx"
FONT = "Times New Roman"
BLACK = RGBColor(0, 0, 0)

ROWS = [
    ("Student name", "Unmesh"),
    ("Registration number", "230911030"),
    ("Assignment title", "Foundations of Generative AI (ICT4403) - Assignment 2"),
    ("Title of the developed autoregressive text-generation system",
     "Comparative Word-Level Autoregressive Next-Word Prediction and "
     "Temperature-Controlled Text Generation on Arthur Conan Doyle's "
     "A Study in Scarlet using Simple RNN, Bidirectional RNN, LSTM, and GRU"),
    ("Dataset name", "A Study in Scarlet by Arthur Conan Doyle (Project Gutenberg eBook #244)"),
    ("Dataset link", "https://www.gutenberg.org/cache/epub/244/pg244.txt"),
    ("GitHub repository link", "https://github.com/unmesh1621-ai/fgai-rnn-text-generation"),
]


def style_run(run, size, bold=False, italic=False):
    run.font.name = FONT
    run._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = BLACK


def add_link(paragraph, url, size):
    """Clickable hyperlink rendered in plain black text."""
    r_id = paragraph.part.relate_to(url, RELATIONSHIP_TYPE.HYPERLINK, is_external=True)
    link = OxmlElement("w:hyperlink")
    link.set(qn("r:id"), r_id)
    run = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    fonts = OxmlElement("w:rFonts")
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        fonts.set(qn(attr), FONT)
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "000000")
    sz = OxmlElement("w:sz")
    sz.set(qn("w:val"), str(int(size * 2)))
    rpr.extend([fonts, color, sz])
    run.append(rpr)
    text = OxmlElement("w:t")
    text.text = url
    run.append(text)
    link.append(run)
    paragraph._p.append(link)


def set_borders(table):
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "6")
        el.set(qn("w:color"), "000000")
        borders.append(el)
    table._tbl.tblPr.append(borders)


def main():
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    for side in ("top_margin", "bottom_margin", "left_margin", "right_margin"):
        setattr(sec, side, Inches(1))

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(5)
    style_run(title.add_run("Foundations of Generative AI (ICT4403)"), 18, bold=True)

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.paragraph_format.space_after = Pt(22)
    style_run(sub.add_run("Assignment 2 - Submission Details"), 13)

    table = doc.add_table(rows=0, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_borders(table)
    for label, value in ROWS:
        left, right = table.add_row().cells
        left.width, right.width = Inches(2.25), Inches(4.25)
        for cell in (left, right):
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(5)
            p.paragraph_format.space_after = Pt(5)
        style_run(left.paragraphs[0].add_run(label), 10.5, bold=True)
        if value.startswith("https://"):
            add_link(right.paragraphs[0], value, 10.5)
        elif "A Study in Scarlet using" in value:
            before, after = value.split("A Study in Scarlet")
            p = right.paragraphs[0]
            style_run(p.add_run(before), 10.5)
            style_run(p.add_run("A Study in Scarlet"), 10.5, italic=True)
            style_run(p.add_run(after), 10.5)
        else:
            style_run(right.paragraphs[0].add_run(value), 10.5)

    doc.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
