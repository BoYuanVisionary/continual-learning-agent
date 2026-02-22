#!/usr/bin/env python3
"""Convert approach_summary.md to PDF using fpdf2."""
import os
import re
from fpdf import FPDF

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class MarkdownPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.add_page()
        self.set_auto_page_break(auto=True, margin=15)
        # Use built-in fonts
        self.set_font("Helvetica", size=10)
        self.in_code_block = False

    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.cell(0, 10, "SFT Sample Count vs. Math Performance vs. Forgetting", align="C")
            self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def add_title(self, text, level=1):
        sizes = {1: 18, 2: 14, 3: 12, 4: 11}
        size = sizes.get(level, 10)
        self.ln(4 if level > 1 else 2)
        self.set_font("Helvetica", "B", size)
        # Clean markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        self.multi_cell(0, size * 0.6, text)
        self.ln(2)
        self.set_font("Helvetica", size=10)

    def add_paragraph(self, text):
        # Handle bold
        text = self._clean_text(text)
        self.set_font("Helvetica", size=10)
        self.multi_cell(0, 5, text)
        self.ln(2)

    def add_bullet(self, text, indent=0):
        text = self._clean_text(text)
        self.set_font("Helvetica", size=10)
        x = self.get_x() + indent * 5
        self.set_x(x + 5)
        self.multi_cell(0, 5, f"  - {text}")
        self.ln(1)

    def add_code_block(self, lines):
        self.set_font("Courier", size=8)
        self.set_fill_color(240, 240, 240)
        for line in lines:
            line = line.rstrip()
            if len(line) > 100:
                line = line[:100] + "..."
            self.cell(0, 4, f"  {line}", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)
        self.set_font("Helvetica", size=10)

    def add_table(self, headers, rows):
        self.set_font("Helvetica", "B", 8)
        # Calculate column widths
        n_cols = len(headers)
        page_width = self.w - 2 * self.l_margin
        col_widths = []
        for i, h in enumerate(headers):
            # Give first column more width
            if i == 0:
                col_widths.append(min(page_width * 0.35, 65))
            else:
                remaining = page_width - col_widths[0] if col_widths else page_width
                col_widths.append(remaining / (n_cols - 1))

        # Adjust if too wide
        total = sum(col_widths)
        if total > page_width:
            factor = page_width / total
            col_widths = [w * factor for w in col_widths]

        # Header
        self.set_fill_color(220, 220, 240)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 5, self._clean_text(h.strip()), border=1, fill=True, align="C")
        self.ln()

        # Rows
        self.set_font("Helvetica", size=7.5)
        for row in rows:
            cells = row
            max_h = 5
            for i, cell in enumerate(cells):
                if i < len(col_widths):
                    cell_text = self._clean_text(cell.strip())
                    self.cell(col_widths[i], 5, cell_text, border=1, align="C" if i > 0 else "L")
            self.ln()
        self.ln(2)
        self.set_font("Helvetica", size=10)

    def _clean_text(self, text):
        """Remove markdown formatting."""
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        text = re.sub(r'`(.*?)`', r'\1', text)
        text = text.replace('\\boxed{}', '\\boxed{}')
        # Replace special characters that fpdf can't handle
        text = text.replace('\u2264', '<=')
        text = text.replace('\u2265', '>=')
        text = text.replace('\u2192', '->')
        text = text.replace('\u2248', '~')
        text = text.replace('\u0394', 'Delta')
        text = text.replace('\u2022', '-')
        text = text.replace('\u2013', '-')
        text = text.replace('\u2014', '--')
        text = text.replace('\u2018', "'")
        text = text.replace('\u2019', "'")
        text = text.replace('\u201c', '"')
        text = text.replace('\u201d', '"')
        # Remove any remaining non-latin1 characters
        text = text.encode('latin-1', errors='replace').decode('latin-1')
        return text


def parse_markdown(md_text):
    """Parse markdown into structured elements."""
    lines = md_text.split('\n')
    elements = []
    i = 0
    in_code = False
    code_lines = []
    in_table = False
    table_headers = []
    table_rows = []

    while i < len(lines):
        line = lines[i]

        # Code blocks
        if line.strip().startswith('```'):
            if in_code:
                elements.append(('code', code_lines))
                code_lines = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue

        if in_code:
            code_lines.append(line)
            i += 1
            continue

        # Table
        if '|' in line and line.strip().startswith('|'):
            cells = [c.strip() for c in line.split('|')[1:-1]]
            if all(re.match(r'^[-:]+$', c) for c in cells):
                # Separator line
                i += 1
                continue
            elif not in_table:
                in_table = True
                table_headers = cells
            else:
                table_rows.append(cells)
            i += 1
            # Check if next line is not a table
            if i >= len(lines) or '|' not in lines[i] or not lines[i].strip().startswith('|'):
                elements.append(('table', table_headers, table_rows))
                in_table = False
                table_headers = []
                table_rows = []
            continue

        # Headers
        if line.startswith('#'):
            level = len(line) - len(line.lstrip('#'))
            text = line.lstrip('#').strip()
            elements.append(('heading', level, text))
            i += 1
            continue

        # Bullet points
        if re.match(r'^(\s*)[*\-]\s', line):
            indent = len(line) - len(line.lstrip())
            text = re.sub(r'^\s*[*\-]\s+', '', line)
            elements.append(('bullet', text, indent))
            i += 1
            continue

        # Numbered lists
        if re.match(r'^\s*\d+\.\s', line):
            text = re.sub(r'^\s*\d+\.\s+', '', line)
            elements.append(('bullet', text, 0))
            i += 1
            continue

        # Empty line
        if not line.strip():
            i += 1
            continue

        # Regular paragraph - collect consecutive lines
        para_lines = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].startswith('#') and not lines[i].startswith('```') and not re.match(r'^[*\-]\s', lines[i]) and not re.match(r'^\d+\.\s', lines[i]) and '|' not in lines[i]:
            para_lines.append(lines[i])
            i += 1
        elements.append(('paragraph', ' '.join(para_lines)))

    return elements


def main():
    md_path = os.path.join(SCRIPT_DIR, "results", "approach_summary.md")
    pdf_path = os.path.join(SCRIPT_DIR, "results", "approach_summary.pdf")

    with open(md_path) as f:
        md_text = f.read()

    elements = parse_markdown(md_text)

    pdf = MarkdownPDF()
    pdf.alias_nb_pages()

    for elem in elements:
        if elem[0] == 'heading':
            pdf.add_title(elem[2], elem[1])
        elif elem[0] == 'paragraph':
            pdf.add_paragraph(elem[1])
        elif elem[0] == 'bullet':
            pdf.add_bullet(elem[1], elem[2])
        elif elem[0] == 'code':
            pdf.add_code_block(elem[1])
        elif elem[0] == 'table':
            pdf.add_table(elem[1], elem[2])

    pdf.output(pdf_path)
    print(f"PDF saved to: {pdf_path}")


if __name__ == "__main__":
    main()
