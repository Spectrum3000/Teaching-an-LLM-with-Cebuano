from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from openpyxl.chart import BarChart, LineChart, PieChart, Reference
from openpyxl.chart.label import DataLabelList
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo

from common import project_root


NAVY = "17324D"
TEAL = "0F766E"
ORANGE = "D97706"
BLUE = "2563EB"
PURPLE = "7C3AED"
LIGHT_BLUE = "DBEAFE"
LIGHT_TEAL = "CCFBF1"
LIGHT_ORANGE = "FEF3C7"
LIGHT_GRAY = "F1F5F9"
DARK = "1E293B"
WHITE = "FFFFFF"
THIN_GRAY = Side(style="thin", color="CBD5E1")


def load_audit(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def title(ws: Any, text: str, subtitle: str | None = None) -> None:
    ws.merge_cells("A1:H1")
    cell = ws["A1"]
    cell.value = text
    cell.font = Font(size=18, bold=True, color=WHITE)
    cell.fill = PatternFill("solid", fgColor=NAVY)
    cell.alignment = Alignment(horizontal="left")
    if subtitle:
        ws.merge_cells("A2:H2")
        ws["A2"] = subtitle
        ws["A2"].font = Font(italic=True, color="475569")
        ws["A2"].alignment = Alignment(wrap_text=True)


def style_header(row: Any, fill: str = TEAL) -> None:
    cells = [cell for item in row for cell in (item if isinstance(item, tuple) else (item,))]
    for cell in cells:
        cell.font = Font(bold=True, color=WHITE)
        cell.fill = PatternFill("solid", fgColor=fill)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = Border(bottom=THIN_GRAY)


def style_body(ws: Any, start_row: int, end_row: int, end_col: int) -> None:
    for row in ws.iter_rows(min_row=start_row, max_row=end_row, min_col=1, max_col=end_col):
        for cell in row:
            cell.border = Border(bottom=THIN_GRAY)
            cell.alignment = Alignment(vertical="top", wrap_text=True)


def widths(ws: Any, values: dict[str, int]) -> None:
    for column, width in values.items():
        ws.column_dimensions[column].width = width


def add_table(ws: Any, ref: str, name: str) -> None:
    table = Table(displayName=name, ref=ref)
    table.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showRowStripes=True, showColumnStripes=False)
    ws.add_table(table)


def build_workbook(audit: dict[str, Any], output: Path) -> None:
    wb = Workbook()
    guide = wb.active
    guide.title = "Guide"
    corpus = wb.create_sheet("Corpus Data")
    architecture = wb.create_sheet("Architecture")
    weights = wb.create_sheet("Weight Calculations")
    forward = wb.create_sheet("Forward Pass")
    training = wb.create_sheet("Training Metrics")
    decoding = wb.create_sheet("Decoding Metrics")
    graphs = wb.create_sheet("Graphs")
    sources = wb.create_sheet("Sources")

    for ws in wb.worksheets:
        ws.sheet_view.showGridLines = False
        ws.freeze_panes = "A4"

    # Guide
    title(guide, "DistilGPT2 Cebuano Project: Calculation Workbook", "Formula-driven workbook. Blue cells are inputs, green cells are calculated values, and orange cells are measured results to fill after running experiments.")
    guide_rows = [
        (4, "Purpose", "Trace the numbers in the project from downloaded datasets through model weights, LoRA adapters, forward-pass tensors, training, decoding, and KV-cache memory."),
        (5, "How to read formulas", "Select any green cell to see the Excel formula. Calculated values are intentionally exposed rather than hard-coded."),
        (6, "Measured versus derived", "Corpus values come from outputs/corpus_audit.json. Architecture values are derived from the DistilGPT2 configuration. Training and latency cells remain blank until real runs produce ledgers."),
        (7, "Weight limitation", "The workbook calculates parameter counts and storage footprints. It does not invent the learned floating-point values inside the model's weight tensors."),
        (9, "Run order", "1) prepare_corpus.py  2) step0_baseline.py  3) step1_lora.py  4) step2_decode.py  5) step3_dpo.py  6) make_plots.py"),
        (10, "Important formulas", "Perplexity = EXP(loss). Trainable fraction = trainable parameters / total parameters. KV-cache speedup = uncached seconds / cached seconds. LoRA parameters = r x (input dimension + output dimension) per adapted matrix."),
    ]
    guide["A3"] = "Section"
    guide["B3"] = "Explanation"
    style_header(guide["3:3"])
    for row, label, value in guide_rows:
        guide.cell(row, 1, label).font = Font(bold=True, color=DARK)
        guide.cell(row, 2, value)
        guide.merge_cells(start_row=row, start_column=2, end_row=row, end_column=8)
    style_body(guide, 4, 10, 8)
    widths(guide, {"A": 24, "B": 28, "C": 18, "D": 18, "E": 18, "F": 18, "G": 18, "H": 18})
    guide.row_dimensions[6].height = 42
    guide.row_dimensions[7].height = 42

    # Corpus Data
    title(corpus, "Measured Dataset and Corpus Calculations", "Values in this sheet are imported from the live Hugging Face dataset run recorded in outputs/corpus_audit.json.")
    corpus_headers = ["Source", "Rows observed", "Usable units", "Characters / bytes", "Share of combined units", "Calculation / interpretation"]
    for col, value in enumerate(corpus_headers, 1):
        corpus.cell(3, col, value)
    style_header(corpus["3:3"])
    ner = audit["sources"]["josephimperial/CebuaNER"]
    readability = audit["sources"]["filbench/cebuano-readability"]
    corpus.append(["josephimperial/CebuaNER", ner["rows_seen"], ner["documents_reconstructed"], "", "", "188,291 CoNLL-style rows; token labels discarded; blank lines delimit 4,258 text units."])
    corpus.append(["filbench/cebuano-readability", readability["rows_seen"], readability["passages_kept"], "", "", "350 passages from the live test split; the text field is retained."])
    corpus.append(["Combined output", "", "=SUM(C4:C5)", audit["output"]["characters"], "=C6/C6", "Combined documents written to data/cebuano_corpus.txt."])
    corpus.append(["", "", "", audit["output"]["bytes_utf8"], "", "UTF-8 byte size of the generated corpus file."])
    for row in range(4, 8):
        corpus.cell(row, 5).number_format = "0.00%"
    style_body(corpus, 4, 7, 6)
    add_table(corpus, "A3:F6", "CorpusSummary")
    corpus["A10"] = "Additional audit value"
    corpus["B10"] = "Measured number"
    style_header(corpus["A10:B10"], ORANGE)
    audit_rows = [
        ("CebuaNER -DOCSTART- markers", ner["docstart_boundaries"]),
        ("CebuaNER blank boundaries", ner["blank_boundaries"]),
        ("Approximate whitespace words", audit["output"]["words_approx"]),
        ("Corpus characters", audit["output"]["characters"]),
        ("Corpus UTF-8 bytes", audit["output"]["bytes_utf8"]),
    ]
    for row, values in enumerate(audit_rows, 11):
        corpus.cell(row, 1, values[0])
        corpus.cell(row, 2, values[1])
    style_body(corpus, 11, 15, 2)
    widths(corpus, {"A": 34, "B": 18, "C": 18, "D": 22, "E": 24, "F": 72})

    # Architecture
    title(architecture, "DistilGPT2 Configuration Inputs", "These are the architectural inputs used by the downstream weight and tensor calculations.")
    arch_headers = ["Parameter", "Value", "Unit", "Why it matters"]
    for col, value in enumerate(arch_headers, 1):
        architecture.cell(3, col, value)
    style_header(architecture["3:3"], BLUE)
    arch_values = [
        ("Vocabulary size", 50257, "tokens", "GPT-2 tokenizer vocabulary."),
        ("Context length", 1024, "positions", "Maximum learned positional embedding rows."),
        ("Embedding width", 768, "features", "Hidden size and token-vector width."),
        ("Transformer layers", 6, "layers", "DistilGPT2 has half of GPT-2 small's 12 layers."),
        ("Attention heads", 12, "heads", "Number of attention heads per layer."),
        ("Head dimension", "=B6/B8", "features/head", "Embedding width divided by head count."),
        ("FFN multiplier", 4, "x hidden", "GPT-2 feed-forward width multiplier."),
        ("FFN width", "=B6*B10", "features", "Intermediate feed-forward width."),
        ("Bytes per FP32 weight", 4, "bytes", "Used for a transparent storage estimate."),
        ("LoRA rank", 8, "rank", "Low-rank adapter dimension."),
        ("LoRA alpha", 16, "scale", "Adapter scaling hyperparameter."),
    ]
    for row, values in enumerate(arch_values, 4):
        for col, value in enumerate(values, 1):
            architecture.cell(row, col, value)
    for row in (9, 12):
        architecture.cell(row, 2).fill = PatternFill("solid", fgColor=LIGHT_TEAL)
    for row in range(4, 15):
        architecture.cell(row, 2).fill = PatternFill("solid", fgColor=LIGHT_BLUE if row not in (9, 12) else LIGHT_TEAL)
    style_body(architecture, 4, 14, 4)
    widths(architecture, {"A": 28, "B": 16, "C": 18, "D": 62})

    # Weight Calculations
    title(weights, "Model Weight and LoRA Parameter Calculations", "Every row exposes tensor dimensions, the parameter formula, parameter count, and FP32 storage estimate.")
    headers = ["Component", "Tensor / role", "Rows", "Columns", "Copies", "Formula", "Parameters", "FP32 MB", "Trainable?", "Explanation"]
    for col, value in enumerate(headers, 1):
        weights.cell(3, col, value)
    style_header(weights["3:3"], TEAL)
    rows = [
        ("Token embedding", "vocab x embed", "=Architecture!B4", "=Architecture!B6", 1, "=C4*D4*E4", "=C4*D4*E4", "=G4*Architecture!B12/1048576", "Frozen", "Token lookup table; tied to the language-model output head."),
        ("Position embedding", "context x embed", "=Architecture!B5", "=Architecture!B6", 1, "=C5*D5*E5", "=C5*D5*E5", "=G5*Architecture!B12/1048576", "Frozen", "Learned absolute position vectors."),
        ("Attention QKV", "embed x 3embed", "=Architecture!B6", "=3*Architecture!B6", "=Architecture!B7", "=C6*D6*E6", "=C6*D6*E6", "=G6*Architecture!B12/1048576", "Frozen", "Combined query, key, and value projection in c_attn."),
        ("Attention output", "embed x embed", "=Architecture!B6", "=Architecture!B6", "=Architecture!B7", "=C7*D7*E7", "=C7*D7*E7", "=G7*Architecture!B12/1048576", "Frozen", "Projection after attention heads are concatenated."),
        ("Feed-forward in", "embed x ffn", "=Architecture!B6", "=Architecture!B11", "=Architecture!B7", "=C8*D8*E8", "=C8*D8*E8", "=G8*Architecture!B12/1048576", "Frozen", "First dense projection expands hidden width."),
        ("Feed-forward out", "ffn x embed", "=Architecture!B11", "=Architecture!B6", "=Architecture!B7", "=C9*D9*E9", "=C9*D9*E9", "=G9*Architecture!B12/1048576", "Frozen", "Second dense projection returns to hidden width."),
        ("Projection biases", "QKV + output", "=3*Architecture!B6+Architecture!B6", 1, "=Architecture!B7", "=C10*D10*E10", "=C10*D10*E10", "=G10*Architecture!B12/1048576", "Frozen", "Bias terms for attention projections."),
        ("FFN biases", "in + out", "=Architecture!B11+Architecture!B6", 1, "=Architecture!B7", "=C11*D11*E11", "=C11*D11*E11", "=G11*Architecture!B12/1048576", "Frozen", "Bias terms for both feed-forward projections."),
        ("LayerNorm scales/biases", "2 norms x 2 vectors", "=2*2*Architecture!B6", 1, "=Architecture!B7", "=C12*D12*E12", "=C12*D12*E12", "=G12*Architecture!B12/1048576", "Frozen", "Two parameters per hidden feature for two norms per layer."),
        ("Final LayerNorm", "scale + bias", "=2*Architecture!B6", 1, 1, "=C13*D13*E13", "=C13*D13*E13", "=G13*Architecture!B12/1048576", "Frozen", "Final normalization before the tied output projection."),
        ("LoRA QKV A", "rank x embed", "=Architecture!B13", "=Architecture!B6", "=Architecture!B7", "=C14*D14*E14", "=C14*D14*E14", "=G14*Architecture!B12/1048576", "Trainable", "Low-rank down projection for each adapted attention QKV module."),
        ("LoRA QKV B", "3embed x rank", "=3*Architecture!B6", "=Architecture!B13", "=Architecture!B7", "=C15*D15*E15", "=C15*D15*E15", "=G15*Architecture!B12/1048576", "Trainable", "Low-rank up projection for each adapted attention QKV module."),
    ]
    for row, values in enumerate(rows, 4):
        for col, value in enumerate(values, 1):
            weights.cell(row, col, value)
    total_row = 18
    weights.cell(total_row, 1, "Architecture total, excluding LoRA")
    weights.cell(total_row, 7, "=SUM(G4:G13)")
    weights.cell(total_row, 8, "=SUM(H4:H13)")
    weights.cell(total_row, 9, "Frozen base")
    weights.cell(total_row + 1, 1, "LoRA trainable total")
    weights.cell(total_row + 1, 7, "=SUM(G14:G15)")
    weights.cell(total_row + 1, 8, "=SUM(H14:H15)")
    weights.cell(total_row + 1, 9, "Trainable")
    weights.cell(total_row + 2, 1, "Combined parameter count")
    weights.cell(total_row + 2, 7, "=G18+G19")
    weights.cell(total_row + 2, 8, "=H18+H19")
    weights.cell(total_row + 2, 9, "Base + LoRA")
    weights.cell(total_row + 3, 1, "LoRA fraction of combined parameters")
    weights.cell(total_row + 3, 7, "=G19/G20")
    weights.cell(total_row + 3, 9, "Formula")
    for row in range(18, 22):
        for cell in weights[row]:
            cell.fill = PatternFill("solid", fgColor=LIGHT_ORANGE if row == 19 else LIGHT_TEAL)
            cell.font = Font(bold=True, color=DARK)
    style_body(weights, 4, 16, 10)
    for row in range(4, 22):
        weights.cell(row, 7).number_format = '#,##0'
        weights.cell(row, 8).number_format = '0.000'
    weights[21][6].number_format = "0.0000%"
    widths(weights, {"A": 25, "B": 24, "C": 15, "D": 15, "E": 12, "F": 20, "G": 18, "H": 15, "I": 15, "J": 62})
    weights.freeze_panes = "A4"

    # Forward pass
    title(forward, "Forward-Pass Tensor Shapes and Operation Counts", "Example uses batch=1 and sequence length=128. Change the blue inputs to see dependent formulas update.")
    forward["A3"] = "Input"
    forward["B3"] = "Value"
    forward["C3"] = "Unit"
    forward["D3"] = "Meaning"
    style_header(forward["A3:D3"], BLUE)
    forward_rows = [
        ("Batch size", 1, "examples", "Number of sequences processed together."),
        ("Sequence length", 128, "tokens", "Tokens in the example input block."),
        ("Embedding width", "=Architecture!B6", "features", "Hidden dimension."),
        ("Attention heads", "=Architecture!B8", "heads", "Parallel attention heads."),
        ("Head dimension", "=Architecture!B9", "features/head", "Embedding width / heads."),
        ("Vocabulary size", "=Architecture!B4", "logits", "Output classes at every position."),
    ]
    for row, values in enumerate(forward_rows, 4):
        for col, value in enumerate(values, 1):
            forward.cell(row, col, value)
    style_body(forward, 4, 9, 4)
    forward["F3"] = "Tensor / operation"
    forward["G3"] = "Shape or calculation"
    forward["H3"] = "Elements / operations"
    forward["I3"] = "Explanation"
    style_header(forward["F3:I3"], TEAL)
    tensor_rows = [
        ("Input IDs", "=B4 x B5", "=B4*B5", "Integer token IDs."),
        ("Hidden states", "=B4 x B5 x B6", "=B4*B5*B6", "Token embeddings plus positional embeddings."),
        ("QKV projection output", "=B4 x B5 x (3*B6)", "=B4*B5*3*B6", "One matrix produces Q, K, and V."),
        ("Q/K/V per head", "=B4 x B7 x B5 x B8", "=B4*B7*B5*B8", "Reshaped multi-head tensors."),
        ("Attention score matrix", "=B4 x B7 x B5 x B5", "=B4*B7*B5*B5", "Each query position scores every key position."),
        ("FFN intermediate", "=B4 x B5 x Architecture!B11", "=B4*B5*Architecture!B11", "Expanded hidden representation."),
        ("Output logits", "=B4 x B5 x B9", "=B4*B5*B9", "Vocabulary score vector for each position."),
        ("Cross-entropy targets", "=B4 x B5", "=B4*B5", "One next-token target per position."),
    ]
    for row, values in enumerate(tensor_rows, 4):
        for col, value in enumerate(values, 6):
            forward.cell(row, col, value)
    style_body(forward, 4, 11, 9)
    widths(forward, {"A": 22, "B": 15, "C": 18, "D": 45, "E": 4, "F": 28, "G": 28, "H": 24, "I": 55})

    # Training metrics
    title(training, "Training Metrics and Perplexity Calculator", "Orange cells are intentionally open for actual values from outputs/step1_lora.json; green formulas calculate the derived quantities.")
    train_headers = ["Step", "Logged loss", "Perplexity = EXP(loss)", "Learning rate", "Elapsed seconds", "Source / note"]
    for col, value in enumerate(train_headers, 1):
        training.cell(3, col, value)
    style_header(training["3:3"], ORANGE)
    for row in range(4, 24):
        training.cell(row, 1, row - 3)
        training.cell(row, 2).fill = PatternFill("solid", fgColor=LIGHT_ORANGE)
        training.cell(row, 3, f'=IF(B{row}="","",EXP(B{row}))')
        training.cell(row, 4).fill = PatternFill("solid", fgColor=LIGHT_ORANGE)
        training.cell(row, 5).fill = PatternFill("solid", fgColor=LIGHT_ORANGE)
        training.cell(row, 6, "Paste log_history values here after the LoRA run.")
        training.cell(row, 3).fill = PatternFill("solid", fgColor=LIGHT_TEAL)
    style_body(training, 4, 23, 6)
    for row in range(4, 24):
        training.cell(row, 2).number_format = "0.000000"
        training.cell(row, 3).number_format = "0.0000"
    widths(training, {"A": 12, "B": 18, "C": 24, "D": 18, "E": 18, "F": 58})

    # Decoding metrics
    title(decoding, "Decoding and KV-Cache Benchmark Sheet", "Paste real values from outputs/step2_decode.json. The speed and throughput formulas are already wired.")
    decode_headers = ["Prompt", "Strategy", "Cache", "Generated tokens", "Seconds", "Tokens/second", "Speedup", "Interpretation"]
    for col, value in enumerate(decode_headers, 1):
        decoding.cell(3, col, value)
    style_header(decoding["3:3"], ORANGE)
    prompts = ["Kumusta ka?", "Ang Cebuano nga pinulongan", "Karon sa buntag"]
    strategies = ["greedy", "top_k", "top_p", "beam"]
    row = 4
    for prompt in prompts:
        for strategy in strategies:
            decoding.cell(row, 1, prompt)
            decoding.cell(row, 2, strategy)
            decoding.cell(row, 3, True)
            decoding.cell(row, 4).fill = PatternFill("solid", fgColor=LIGHT_ORANGE)
            decoding.cell(row, 5).fill = PatternFill("solid", fgColor=LIGHT_ORANGE)
            decoding.cell(row, 6, f'=IFERROR(D{row}/E{row},"")')
            decoding.cell(row, 8, "Paste generated_tokens and seconds from the ledger.")
            row += 1
    for prompt in prompts:
        decoding.cell(row, 1, prompt)
        decoding.cell(row, 2, "cache_ablation")
        decoding.cell(row, 3, "cached")
        decoding.cell(row, 4).fill = PatternFill("solid", fgColor=LIGHT_ORANGE)
        decoding.cell(row, 5).fill = PatternFill("solid", fgColor=LIGHT_ORANGE)
        decoding.cell(row, 7).fill = PatternFill("solid", fgColor=LIGHT_TEAL)
        decoding.cell(row, 8, "Enter uncached seconds in the next row; speedup = uncached / cached.")
        row += 1
        decoding.cell(row, 1, prompt)
        decoding.cell(row, 2, "cache_ablation")
        decoding.cell(row, 3, "uncached")
        decoding.cell(row, 4).fill = PatternFill("solid", fgColor=LIGHT_ORANGE)
        decoding.cell(row, 5).fill = PatternFill("solid", fgColor=LIGHT_ORANGE)
        decoding.cell(row, 7, f'=IFERROR(E{row}/E{row-1},"")')
        decoding.cell(row, 8, "Formula row: uncached seconds divided by cached seconds.")
        row += 1
    style_body(decoding, 4, row - 1, 8)
    for current in range(4, row):
        decoding.cell(current, 6).number_format = "0.000"
        decoding.cell(current, 7).number_format = "0.000x"
    widths(decoding, {"A": 30, "B": 18, "C": 14, "D": 18, "E": 14, "F": 18, "G": 14, "H": 60})

    # Graphs
    title(graphs, "Graphs: Weights, Data Composition, and Runtime Formulas", "Charts are linked to workbook cells. Architecture charts work immediately; measured training/latency charts become informative after ledger values are entered.")
    graphs["A3"] = "Chart"
    graphs["B3"] = "What it demonstrates"
    style_header(graphs["A3:B3"], TEAL)
    graph_notes = [
        ("Parameter distribution", "The base model's major embedding, attention, and feed-forward parameter groups."),
        ("Frozen versus LoRA", "Why LoRA trains a small fraction while keeping the base weights frozen."),
        ("Corpus units", "The measured contribution of CebuaNER and readability passages."),
        ("KV cache memory", "How cache memory scales linearly with sequence length."),
        ("Training curve", "Loss and derived perplexity after actual training values are entered."),
    ]
    for row, values in enumerate(graph_notes, 4):
        graphs.cell(row, 1, values[0])
        graphs.cell(row, 2, values[1])
    style_body(graphs, 4, 8, 2)
    widths(graphs, {"A": 28, "B": 72, "C": 16, "D": 16, "E": 16, "F": 16, "G": 16, "H": 16})

    # Helper source data for charts on right side.
    graphs["J3"] = "Component"
    graphs["K3"] = "Parameters"
    style_header(graphs["J3:K3"], BLUE)
    for index, source_row in enumerate(range(4, 10), 4):
        graphs.cell(index, 10, f"='Weight Calculations'!A{source_row}")
        graphs.cell(index, 11, f"='Weight Calculations'!G{source_row}")
    pie = PieChart()
    pie.title = "Base parameter distribution"
    pie.add_data(Reference(graphs, min_col=11, min_row=3, max_row=9), titles_from_data=True)
    pie.set_categories(Reference(graphs, min_col=10, min_row=4, max_row=9))
    pie.height = 8
    pie.width = 12
    pie.dataLabels = DataLabelList()
    pie.dataLabels.showPercent = True
    graphs.add_chart(pie, "A11")

    graphs["J12"] = "Type"
    graphs["K12"] = "Parameters"
    style_header(graphs["J12:K12"], PURPLE)
    graphs["J13"] = "Frozen base"
    graphs["K13"] = "='Weight Calculations'!G18"
    graphs["J14"] = "Trainable LoRA"
    graphs["K14"] = "='Weight Calculations'!G19"
    bar = BarChart()
    bar.type = "bar"
    bar.style = 10
    bar.title = "Frozen base versus trainable LoRA"
    bar.y_axis.title = "Parameter type"
    bar.x_axis.title = "Parameters"
    bar.add_data(Reference(graphs, min_col=11, min_row=12, max_row=14), titles_from_data=True)
    bar.set_categories(Reference(graphs, min_col=10, min_row=13, max_row=14))
    bar.height = 7
    bar.width = 12
    graphs.add_chart(bar, "J16")

    graphs["M3"] = "Corpus source"
    graphs["N3"] = "Usable units"
    style_header(graphs["M3:N3"], ORANGE)
    graphs["M4"] = "CebuaNER"
    graphs["N4"] = "='Corpus Data'!C4"
    graphs["M5"] = "Readability"
    graphs["N5"] = "='Corpus Data'!C5"
    corpus_chart = BarChart()
    corpus_chart.title = "Usable corpus units"
    corpus_chart.y_axis.title = "Documents / passages"
    corpus_chart.x_axis.title = "Source"
    corpus_chart.add_data(Reference(graphs, min_col=14, min_row=3, max_row=5), titles_from_data=True)
    corpus_chart.set_categories(Reference(graphs, min_col=13, min_row=4, max_row=5))
    corpus_chart.height = 7
    corpus_chart.width = 12
    graphs.add_chart(corpus_chart, "A29")

    graphs["M9"] = "Sequence length"
    graphs["N9"] = "KV cache bytes"
    style_header(graphs["M9:N9"], TEAL)
    for row, seq in enumerate((1, 16, 128, 512, 1024), 10):
        graphs.cell(row, 13, seq)
        graphs.cell(row, 14, f"=M{row}*2*Architecture!B7*Architecture!B6*Architecture!B12")
    cache_chart = LineChart()
    cache_chart.title = "KV-cache memory versus sequence length"
    cache_chart.y_axis.title = "Bytes (FP32)"
    cache_chart.x_axis.title = "Sequence length"
    cache_chart.add_data(Reference(graphs, min_col=14, min_row=9, max_row=14), titles_from_data=True)
    cache_chart.set_categories(Reference(graphs, min_col=13, min_row=10, max_row=14))
    cache_chart.height = 7
    cache_chart.width = 12
    graphs.add_chart(cache_chart, "J29")

    # Sources
    title(sources, "Sources, Assumptions, and Reproducibility Notes", "Use this sheet to distinguish observed dataset facts, model configuration facts, and assumptions used only for calculations.")
    source_headers = ["Category", "Source / value", "How used"]
    for col, value in enumerate(source_headers, 1):
        sources.cell(3, col, value)
    style_header(sources["3:3"], TEAL)
    source_rows = [
        ("Dataset", "josephimperial/CebuaNER", "Loaded with load_dataset; live schema and counts recorded in corpus_audit.json."),
        ("Dataset", "filbench/cebuano-readability", "Loaded with load_dataset; live schema and counts recorded in corpus_audit.json."),
        ("Model", "distilgpt2 configuration", "Vocab 50,257; context 1,024; width 768; 6 layers; 12 heads."),
        ("Architecture", "GPT-2 family tied output head", "The output projection reuses token embeddings, so it is not double-counted in the base total."),
        ("LoRA", "target module c_attn", "Adapter parameter count uses rank 8 on the combined QKV projection in each of 6 layers."),
        ("Storage", "4 bytes per FP32 parameter", "A simple memory estimate; actual checkpoints may use other dtypes and serialization overhead."),
        ("Measured output", "outputs/corpus_audit.json", "Generated from the live dataset download on this machine."),
        ("Not measured yet", "Training and decoding sheets", "Orange cells must be filled from real step1 and step2 JSON ledgers; no invented performance values are included."),
    ]
    for row, values in enumerate(source_rows, 4):
        for col, value in enumerate(values, 1):
            sources.cell(row, col, value)
    style_body(sources, 4, 11, 3)
    widths(sources, {"A": 18, "B": 44, "C": 90})

    # Workbook-wide formatting.
    for ws in wb.worksheets:
        ws.sheet_properties.pageSetUpPr.fitToPage = True
        ws.page_setup.fitToWidth = 1
        ws.page_setup.fitToHeight = 0
        for row in ws.iter_rows():
            for cell in row:
                if cell.value is not None and cell.alignment == Alignment():
                    cell.alignment = Alignment(vertical="top", wrap_text=True)
        ws.sheet_view.zoomScale = 90
    wb.calculation.fullCalcOnLoad = True
    wb.calculation.forceFullCalc = True
    output.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the detailed Excel calculation workbook.")
    parser.add_argument("--audit", type=Path, default=project_root() / "outputs" / "corpus_audit.json")
    parser.add_argument("--output", type=Path, default=project_root() / "outputs" / "cebuano_project_calculations.xlsx")
    args = parser.parse_args()
    if not args.audit.exists():
        raise FileNotFoundError(f"Run prepare_corpus.py first: {args.audit}")
    build_workbook(load_audit(args.audit), args.output)
    print(f"Workbook written to {args.output}")


if __name__ == "__main__":
    main()
