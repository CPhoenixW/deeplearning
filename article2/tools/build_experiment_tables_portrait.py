#!/usr/bin/env python3
"""Build portrait, editable Word tables for Section 4.2.

The output deliberately uses compact grouped cells so all attack families fit
on a portrait A4 page while every attack name remains explicit and editable.
LibreOffice imports the HTML tables as native Word tables; no screenshots are
embedded in the DOCX.
"""

from __future__ import annotations

import html
import io
import subprocess
import zipfile
from pathlib import Path

from lxml import etree


ROOT = Path(__file__).resolve().parents[1]
HTML_PATH = ROOT / "experiment_main_tables_portrait_source.html"
DOCX_PATH = ROOT / "experiment_main_tables_portrait.docx"

DATASETS = [
    ("MNIST", "LeNet"),
    ("Fashion-MNIST", "CNN"),
    ("CIFAR-10", "ResNet-18"),
    ("COVIDx", "CNN"),
    ("AG News", "Transformer"),
]

METHODS = [
    "FedAvg",
    "Trimmed Mean",
    "Multi-Krum",
    "LASA",
    "FedSECA",
    "BNGuard",
    "FedDMC",
    "AE-SVDD (Ours)",
]

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W_NS}


def qn(local: str) -> str:
    return f"{{{W_NS}}}{local}"


def child(parent: etree._Element, name: str) -> etree._Element:
    found = parent.find(f"w:{name}", NS)
    if found is None:
        found = etree.SubElement(parent, qn(name))
    return found


def set_border(parent: etree._Element, side: str, value: str, size: int, color: str) -> None:
    borders = child(parent, "tcBorders")
    element = borders.find(f"w:{side}", NS)
    if element is None:
        element = etree.SubElement(borders, qn(side))
    element.set(qn("val"), value)
    element.set(qn("sz"), str(size))
    element.set(qn("space"), "0")
    element.set(qn("color"), color)


def logical_cells(row: etree._Element) -> list[tuple[etree._Element, int, int]]:
    result: list[tuple[etree._Element, int, int]] = []
    column = 0
    for tc in row.findall("w:tc", NS):
        tc_pr = child(tc, "tcPr")
        span_el = tc_pr.find("w:gridSpan", NS)
        span = int(span_el.get(qn("val"))) if span_el is not None else 1
        result.append((tc, column, span))
        column += span
    return result


def format_run(run: etree._Element, size: int, bold: bool | None = None) -> None:
    run_pr = run.find("w:rPr", NS)
    if run_pr is None:
        run_pr = etree.Element(qn("rPr"))
        run.insert(0, run_pr)
    fonts = child(run_pr, "rFonts")
    for key in ("ascii", "hAnsi", "eastAsia", "cs"):
        fonts.set(qn(key), "Times New Roman")
    child(run_pr, "sz").set(qn("val"), str(size))
    child(run_pr, "szCs").set(qn("val"), str(size))
    if bold is not None:
        child(run_pr, "b").set(qn("val"), "true" if bold else "false")
        child(run_pr, "bCs").set(qn("val"), "true" if bold else "false")


def format_paragraph(paragraph: etree._Element, align: str, line: int) -> None:
    p_pr = paragraph.find("w:pPr", NS)
    if p_pr is None:
        p_pr = etree.Element(qn("pPr"))
        paragraph.insert(0, p_pr)
    child(p_pr, "jc").set(qn("val"), align)
    spacing = child(p_pr, "spacing")
    spacing.set(qn("before"), "0")
    spacing.set(qn("after"), "0")
    spacing.set(qn("line"), str(line))
    spacing.set(qn("lineRule"), "exact")


def attack_values(dataset: str, detection: bool = False) -> list[str]:
    """Return the placeholders used by the utility or detection table."""
    if detection:
        available = "— / — / —"
        unsupported = "N/A / N/A / N/A"
        return [
            available,  # LF
            available,  # GN
            available,  # SF
            available,  # Min-Max
            unsupported if dataset == "AG News" else available,  # BD
            unsupported if dataset == "AG News" else available,  # DBA
        ]
    backdoor = "N/A" if dataset == "AG News" else "—"
    mix = "N/A / N/A" if dataset == "AG News" else "— / —"
    return ["—", "—", "—", "—", "—", backdoor, backdoor, mix, mix]


def table1_rows() -> str:
    rows = []
    for dataset, model in DATASETS:
        for method_index, method in enumerate(METHODS):
            classes = []
            if method_index == len(METHODS) - 1:
                classes = ["ours", "dataset-end"]
            row = [f'<tr class="{" ".join(classes)}">']
            if method_index == 0:
                row.append(
                    '<td class="dataset" rowspan="8">'
                    f"{html.escape(dataset)}<br><span class=\"model\">({html.escape(model)})</span></td>"
                )
            row.append(f'<td class="method">{html.escape(method)}</td>')
            row.append('<td class="metric">—</td>')
            for value in attack_values(dataset):
                row.append(f'<td class="metric">{value}</td>')
            row.append("</tr>")
            rows.append("".join(row))
    return "\n".join(rows)


def table2_rows() -> str:
    rows = []
    for dataset, model in DATASETS:
        for method_index, method in enumerate(METHODS):
            classes = []
            if method_index == len(METHODS) - 1:
                classes = ["ours", "dataset-end"]
            row = [f'<tr class="{" ".join(classes)}">']
            if method_index == 0:
                row.append(
                    '<td class="dataset" rowspan="8">'
                    f"{html.escape(dataset)}<br><span class=\"model\">({html.escape(model)})</span></td>"
                )
            row.append(f'<td class="method">{html.escape(method)}</td>')
            for value in attack_values(dataset, detection=True):
                row.append(f'<td class="metric triplet">{value}</td>')
            row.append("</tr>")
            rows.append("".join(row))
    return "\n".join(rows)


def ablation_rows() -> str:
    rows = []
    configurations = ("P1-only", "P2-only", "Full")
    for ratio in ("10%", "20%", "30%", "40%"):
        for configuration in configurations:
            classes = "full" if configuration == "Full" else ""
            rows.append(
                f'<tr class="{classes}">'
                f'<td class="ratio">{ratio}</td>'
                f'<td class="method">{configuration}</td>'
                '<td class="metric triplet">— / —</td>'
                '<td class="metric triplet">— / —</td>'
                '<td class="metric triplet">— / —</td>'
                '<td class="metric triplet">— / —</td>'
                '</tr>'
            )
    return "\n".join(rows)


def topk_validation_rows() -> str:
    rows = []
    for ratio in ("10%", "20%", "30%", "40%"):
        for attack in ("LF", "GN", "SF", "BD"):
            rows.append(
                "<tr>"
                f'<td class="ratio">{ratio}</td>'
                f'<td class="method">{attack}</td>'
                '<td class="metric">—</td>'
                '<td class="metric">—</td>'
                '<td class="metric">—</td>'
                '<td class="metric">—</td>'
                '<td class="metric">—</td>'
                '<td class="metric">—</td>'
                "</tr>"
            )
    return "\n".join(rows)


def build_html() -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Section 4.2 Tables</title>
<style>
  @page {{ size: A4 portrait; margin: 0.52in 0.52in 0.45in 0.52in; }}
  html, body {{ margin: 0; padding: 0; }}
  body {{ font-family: "Times New Roman", "Liberation Serif", serif; color: #000; font-size: 8pt; }}
  .caption {{ margin: 0 0 5pt; text-align: center; font-size: 10pt; line-height: 1.08; }}
  .caption .label {{ font-weight: bold; }}
  .note {{ margin: 4pt 0 0; font-size: 7.4pt; line-height: 1.12; text-align: left; }}
  .note .label {{ font-weight: bold; }}
  .page-break {{ page-break-before: always; break-before: page; }}
  table {{ width: 100%; border-collapse: collapse; table-layout: fixed; border-top: 1.4pt solid #000; border-bottom: 1.4pt solid #000; }}
  th, td {{ border: 0; padding: 1.25pt 1pt; text-align: center; vertical-align: middle; line-height: 1.02; }}
  thead th {{ font-weight: bold; white-space: nowrap; font-size: 7.2pt; }}
  thead tr:last-child th {{ border-bottom: 0.85pt solid #000; }}
  th.group {{ border-bottom: 0.45pt solid #777; }}
  .sep-left {{ border-left: 0.45pt solid #777; }}
  td.dataset {{ font-size: 9pt; font-weight: bold; }}
  td.dataset .model {{ font-weight: normal; }}
  td.method {{ text-align: left; padding-left: 4pt; white-space: nowrap; }}
  td.metric {{ font-variant-numeric: tabular-nums; }}
  td.triplet {{ white-space: nowrap; font-size: 7.2pt; }}
  tr.ours td {{ background: #d9d9d9; font-weight: bold; }}
  tr.ours td.dataset {{ background: #fff; }}
  tr.dataset-end td {{ border-bottom: 1.4pt double #444; }}
  .table2 th, .table2 td {{ padding-top: 1.25pt; padding-bottom: 1.25pt; }}
  .table3 th, .table3 td {{ padding-top: 2pt; padding-bottom: 2pt; }}
  .table3 thead tr:first-child th.ablation-attack {{ border-bottom: 0.85pt solid #000; }}
  .table3 thead tr:nth-child(2) th.ablation-dar {{ border-bottom: 0.85pt solid #000; }}
  .table4 th, .table4 td {{ padding-top: 2pt; padding-bottom: 2pt; }}
  td.ratio {{ font-weight: bold; }}
  tr.full td {{ background: #d9d9d9; font-weight: bold; }}
</style>
</head>
<body>
  <p class="caption"><span class="label">Table 1.</span> Global model utility and attack suppression (mean ± std).</p>
  <table class="table1">
    <colgroup><col style="width: 11.5%"><col style="width: 17.3%"><col style="width: 6.4%"><col style="width: 7.2%"><col style="width: 7.2%"><col style="width: 7.2%"><col style="width: 7.2%"><col style="width: 7.2%"><col style="width: 7.2%"><col style="width: 7.2%"><col style="width: 7.2%"><col style="width: 7.2%"></colgroup>
    <thead>
      <tr>
        <th rowspan="3">Dataset (Model)</th>
        <th rowspan="3">Defense Method</th>
        <th rowspan="2" class="sep-left">No Attack</th>
        <th class="group sep-left" colspan="1">Data</th>
        <th class="group sep-left" colspan="4">Byzantine</th>
        <th class="group sep-left" colspan="2">Backdoor</th>
        <th class="group sep-left" colspan="2">Mix</th>
      </tr>
      <tr>
        <th class="sep-left">LF</th>
        <th class="sep-left">GN</th>
        <th>SF</th>
        <th>LIE</th>
        <th>Min-Max</th>
        <th class="sep-left">BD</th>
        <th>DBA</th>
        <th class="sep-left">M1</th>
        <th>M2</th>
      </tr>
      <tr>
        <th class="sep-left">ACC</th>
        <th class="sep-left">ACC</th>
        <th class="sep-left">ACC</th>
        <th>ACC</th>
        <th>ACC</th>
        <th>ACC</th>
        <th class="sep-left">ASR</th>
        <th>ASR</th>
        <th class="sep-left">ACC/ASR</th>
        <th>ACC/ASR</th>
      </tr>
    </thead>
    <tbody>{table1_rows()}</tbody>
  </table>
  <p class="note"><span class="label">Note.</span> Each value is the mean ± sample standard deviation over seeds 42–44, using the final-10-round mean. LF: label flipping; GN: Gaussian poisoning; SF: sign flipping; LIE: little-is-enough attack; Min-Max: Min-Max Byzantine attack; BD: backdoor attack; DBA: distributed backdoor attack. M1 and M2 report ACC / ASR in that order. “—” denotes a pending result; N/A denotes an unsupported condition. AE-SVDD (Ours) is shaded for identification.</p>

  <div class="page-break"></div>
  <p class="caption"><span class="label">Table 2.</span> Malicious client detection under different attacks (mean ± std).</p>
  <table class="table2">
    <colgroup><col style="width: 11.5%"><col style="width: 17.3%"><col style="width: 11.87%"><col style="width: 11.87%"><col style="width: 11.87%"><col style="width: 11.87%"><col style="width: 11.86%"><col style="width: 11.86%"></colgroup>
    <thead>
      <tr>
        <th rowspan="3">Dataset (Model)</th>
        <th rowspan="3">Detector</th>
        <th class="group sep-left" colspan="1">Data</th>
        <th class="group sep-left" colspan="3">Byzantine</th>
        <th class="group sep-left" colspan="2">Backdoor</th>
      </tr>
      <tr>
        <th class="sep-left">LF</th>
        <th class="sep-left">GN</th>
        <th>SF</th>
        <th>Min-Max</th>
        <th class="sep-left">BD</th>
        <th>DBA</th>
      </tr>
      <tr><th class="sep-left">DAR/DPR/RR</th><th class="sep-left">DAR/DPR/RR</th><th>DAR/DPR/RR</th><th>DAR/DPR/RR</th><th class="sep-left">DAR/DPR/RR</th><th>DAR/DPR/RR</th></tr>
    </thead>
    <tbody>{table2_rows()}</tbody>
  </table>
  <p class="note"><span class="label">Note.</span> Values in every attack column follow the fixed order DAR / DPR / RR. DAR: detection accuracy rate; DPR: detection precision rate; RR: malicious-client recall rate. Each metric is reported as mean ± sample standard deviation over seeds 42–44. “—” denotes a pending result; N/A denotes an unsupported condition. AE-SVDD (Ours) is shaded for identification.</p>

  <div class="page-break"></div>
  <p class="caption"><span class="label">Table 3.</span> Two-stage ranking-schedule ablation on Fashion-MNIST (mean ± std).</p>
  <table class="table3">
    <colgroup><col style="width: 12%"><col style="width: 22%"><col style="width: 16.5%"><col style="width: 16.5%"><col style="width: 16.5%"><col style="width: 16.5%"></colgroup>
    <thead>
      <tr>
        <th rowspan="3">Malicious ratio</th>
        <th rowspan="3">Configuration</th>
        <th class="ablation-attack">LF</th>
        <th class="ablation-attack">GN</th>
        <th class="ablation-attack">SF</th>
        <th class="ablation-attack">BD</th>
      </tr>
      <tr>
        <th class="ablation-dar">DAR</th>
        <th class="ablation-dar">DAR</th>
        <th class="ablation-dar">DAR</th>
        <th class="ablation-dar">DAR</th>
      </tr>
      <tr>
        <th>ACC</th>
        <th>ACC</th>
        <th>ACC</th>
        <th>ASR</th>
      </tr>
    </thead>
    <tbody>{ablation_rows()}</tbody>
  </table>
  <p class="note"><span class="label">Note.</span> The ablation is run only on Fashion-MNIST with malicious-client ratios of 10%, 20%, 30%, and 40%, using LF, GN, SF, and BD. Each cell reports DAR / ACC for LF, GN, and SF, and DAR / ASR for BD. Every value is the mean ± sample standard deviation over seeds 42–44; “—” denotes a pending result.</p>

  <div class="page-break"></div>
  <p class="caption"><span class="label">Table 4.</span> Candidate validation accuracy for validation-driven Top-K selection on Fashion-MNIST (mean ± std).</p>
  <table class="table4">
    <colgroup><col style="width: 12%"><col style="width: 14%"><col style="width: 12.33%"><col style="width: 12.33%"><col style="width: 12.33%"><col style="width: 12.33%"><col style="width: 12.33%"><col style="width: 12.35%"></colgroup>
    <thead>
      <tr>
        <th>Malicious ratio</th>
        <th>Attack</th>
        <th>ρ=0%</th>
        <th>ρ=10%</th>
        <th>ρ=20%</th>
        <th>ρ=30%</th>
        <th>ρ=40%</th>
        <th>Selected ρ</th>
      </tr>
    </thead>
    <tbody>{topk_validation_rows()}</tbody>
  </table>
  <p class="note"><span class="label">Note.</span> Each candidate column reports clean validation accuracy for the corresponding rejection ratio ρ. The server selects the candidate with the highest validation accuracy; exact ties are resolved in favor of the larger ρ. Every value is the mean ± sample standard deviation over seeds 42–44; “—” denotes a pending result.</p>
</body>
</html>
"""


def set_page_portrait(section: etree._Element) -> None:
    pg_sz = child(section, "pgSz")
    pg_sz.set(qn("w"), "11906")
    pg_sz.set(qn("h"), "16838")
    pg_sz.attrib.pop(qn("orient"), None)
    pg_mar = child(section, "pgMar")
    for side in ("left", "right", "top", "bottom"):
        pg_mar.set(qn(side), "720")
    pg_mar.set(qn("header"), "300")
    pg_mar.set(qn("footer"), "300")
    pg_mar.set(qn("gutter"), "0")


def set_table_style(
    table: etree._Element,
    widths: list[int],
    body_start: int,
    group_end_rows: set[int],
    separators: set[int],
) -> None:
    usable_width = sum(widths)
    tbl_pr = child(table, "tblPr")
    tbl_w = child(tbl_pr, "tblW")
    tbl_w.set(qn("w"), str(usable_width))
    tbl_w.set(qn("type"), "dxa")
    child(tbl_pr, "jc").set(qn("val"), "center")
    child(tbl_pr, "tblLayout").set(qn("type"), "fixed")
    cell_mar = child(tbl_pr, "tblCellMar")
    for side, value in (("top", "0"), ("bottom", "0"), ("left", "42"), ("right", "42")):
        margin = cell_mar.find(f"w:{side}", NS)
        if margin is None:
            margin = etree.SubElement(cell_mar, qn(side))
        margin.set(qn("w"), value)
        margin.set(qn("type"), "dxa")
    tbl_borders = child(tbl_pr, "tblBorders")
    for side, value, size in (("top", "single", 12), ("bottom", "single", 12), ("left", "nil", 0), ("right", "nil", 0), ("insideH", "nil", 0), ("insideV", "nil", 0)):
        border = tbl_borders.find(f"w:{side}", NS)
        if border is None:
            border = etree.SubElement(tbl_borders, qn(side))
        border.set(qn("val"), value)
        border.set(qn("sz"), str(size))
        border.set(qn("space"), "0")
        border.set(qn("color"), "000000")

    grid = table.find("w:tblGrid", NS)
    if grid is None:
        grid = etree.Element(qn("tblGrid"))
        table.insert(1, grid)
    for old in list(grid):
        grid.remove(old)
    for width in widths:
        col = etree.SubElement(grid, qn("gridCol"))
        col.set(qn("w"), str(width))

    rows = table.findall("w:tr", NS)
    for row_index, row in enumerate(rows):
        tr_pr = child(row, "trPr")
        child(tr_pr, "cantSplit").set(qn("val"), "true")
        height = child(tr_pr, "trHeight")
        height.set(qn("val"), "360" if row_index == 0 else ("250" if row_index < body_start else "205"))
        height.set(qn("hRule"), "exact")
        if row_index < body_start:
            child(tr_pr, "tblHeader").set(qn("val"), "true")
        for tc, column, span in logical_cells(row):
            tc_pr = child(tc, "tcPr")
            tc_w = child(tc_pr, "tcW")
            tc_w.set(qn("w"), str(sum(widths[column:column + span])))
            tc_w.set(qn("type"), "dxa")
            child(tc_pr, "vAlign").set(qn("val"), "center")
            borders = child(tc_pr, "tcBorders")
            for old in list(borders):
                borders.remove(old)
            for side in ("top", "left", "bottom", "right"):
                set_border(tc_pr, side, "nil", 0, "000000")
            if column in separators:
                set_border(tc_pr, "left", "single", 5, "777777")
            if row_index == 0:
                set_border(tc_pr, "top", "single", 12, "000000")
            if row_index == body_start - 1:
                set_border(tc_pr, "bottom", "single", 8, "000000")
            if row_index in group_end_rows:
                set_border(tc_pr, "bottom", "double", 8, "444444")
            if row_index == len(rows) - 1:
                set_border(tc_pr, "bottom", "single", 12, "000000")
            text = "".join(tc.xpath(".//w:t/text()", namespaces=NS))
            ours = row_index in group_end_rows and column >= 1
            shd = tc_pr.find("w:shd", NS)
            if ours:
                if shd is None:
                    shd = etree.SubElement(tc_pr, qn("shd"))
                shd.set(qn("val"), "clear")
                shd.set(qn("color"), "auto")
                shd.set(qn("fill"), "D9D9D9")
            elif shd is not None:
                tc_pr.remove(shd)
            for paragraph in tc.findall("w:p", NS):
                align = "left" if column == 1 else "center"
                format_paragraph(paragraph, align, 165)
                for run in paragraph.findall("w:r", NS):
                    format_run(run, 14, True if (row_index < body_start or ours) else False)


def set_header_row_bottom_border(table: etree._Element, row_index: int, first_column: int) -> None:
    rows = table.findall("w:tr", NS)
    if row_index >= len(rows):
        return
    for tc, column, _span in logical_cells(rows[row_index]):
        if column >= first_column:
            set_border(child(tc, "tcPr"), "bottom", "single", 8, "000000")


def postprocess_docx(path: Path) -> None:
    with zipfile.ZipFile(path, "r") as source:
        entries = {name: source.read(name) for name in source.namelist()}
    root = etree.fromstring(entries["word/document.xml"])
    body = root.find("w:body", NS)
    if body is None:
        raise RuntimeError("DOCX has no body")
    section = body.find("w:sectPr", NS)
    if section is None:
        raise RuntimeError("DOCX has no section")
    set_page_portrait(section)
    tables = body.findall("w:tbl", NS)
    if len(tables) != 4:
        raise RuntimeError(f"Expected four tables, found {len(tables)}")
    set_table_style(
        tables[0], [1200, 1800, 1000] + [710] * 9, 3, {10, 18, 26, 34, 42}, {2, 3, 4, 8, 10}
    )
    set_table_style(
        tables[1], [1200, 1800] + [1233] * 4 + [1234] * 2, 3, {10, 18, 26, 34, 42}, {2, 3, 6}
    )
    set_table_style(
        tables[2], [1200, 2200] + [1740] * 4, 3, {5, 8, 11, 14}, {2, 3, 4, 5}
    )
    set_header_row_bottom_border(tables[2], 0, 2)
    set_header_row_bottom_border(tables[2], 1, 2)
    set_table_style(
        tables[3], [1250, 1450] + [1290] * 6, 1, {4, 8, 12}, {2}
    )
    set_header_row_bottom_border(tables[3], 0, 0)
    paragraphs = body.findall("w:p", NS)
    caption_indices = (0, 2, 4, 6)
    for index, paragraph in enumerate(paragraphs):
        format_paragraph(paragraph, "center" if index in caption_indices else "left", 220 if index in caption_indices else 170)
        for run in paragraph.findall("w:r", NS):
            format_run(run, 19 if index in caption_indices else 13, None)
    # LibreOffice's HTML import may discard CSS page breaks. Force Table 2's
    # caption onto a new portrait page in the Word document itself.
    table2_caption_pr = child(paragraphs[2], "pPr")
    child(table2_caption_pr, "pageBreakBefore").set(qn("val"), "true")
    table3_caption_pr = child(paragraphs[4], "pPr")
    child(table3_caption_pr, "pageBreakBefore").set(qn("val"), "true")
    table4_caption_pr = child(paragraphs[6], "pPr")
    child(table4_caption_pr, "pageBreakBefore").set(qn("val"), "true")
    entries["word/document.xml"] = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as target:
        for name, payload in entries.items():
            target.writestr(name, payload)
    path.write_bytes(output.getvalue())


def main() -> None:
    HTML_PATH.write_text(build_html(), encoding="utf-8")
    subprocess.run(["libreoffice", "--headless", "--convert-to", "docx:Office Open XML Text", "--outdir", str(ROOT), str(HTML_PATH)], check=True)
    converted = ROOT / f"{HTML_PATH.stem}.docx"
    if DOCX_PATH.exists():
        DOCX_PATH.unlink()
    converted.replace(DOCX_PATH)
    postprocess_docx(DOCX_PATH)
    print(DOCX_PATH)


if __name__ == "__main__":
    main()
