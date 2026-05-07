#!/usr/bin/env python3
"""Build a compact VNWoodKnot qualitative comparison figure from retained val-batch renders.

This script uses the main reported seed-42 runs:
  - T0: target-only training
  - T1: source-initialized fine-tuning

Local artifacts do not retain per-image held-out-test prediction dumps or checkpoints for
these runs, but they do retain validation-side `val_batch*.jpg` renderings produced by
Ultralytics during training. To keep the figure reproducible and balanced, the script
selects a fixed set of panel references from those retained renderings and exports:

  1. A compact 3-column qualitative figure (GT / T0 / T1)
  2. A JSON + CSV manifest describing the selected rows
  3. A short note explaining the scope and interpretation
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"

T0_RUN = "t0_yolov8s_vnwoodknot_target_only_e50"
T1_RUN = "t1_y0_3600e200_to_vnwoodknot_e50"

T0_DIR = PROJECT_ROOT / "outputs" / "yolo" / T0_RUN
T1_DIR = PROJECT_ROOT / "outputs" / "yolo" / T1_RUN

FONT_PATH = Path("/System/Library/Fonts/Helvetica.ttc")


@dataclass(frozen=True)
class SelectedCase:
    row_index: int
    batch_index: int
    row: int
    col: int
    case_type: str
    outcome_label: str
    reason_for_selection: str
    filename_hint: str | None = None

    @property
    def panel_ref(self) -> str:
        return f"val_batch{self.batch_index}[r{self.row + 1},c{self.col + 1}]"


SELECTED_CASES: tuple[SelectedCase, ...] = (
    SelectedCase(
        row_index=1,
        batch_index=0,
        row=0,
        col=0,
        case_type="T1 localization improvement",
        outcome_label="T1 better",
        reason_for_selection=(
            "T0 misses the live-knot region, whereas T1 recovers it with a coherent box. "
            "This row illustrates the case-level recall/localization improvement that source "
            "initialization can sometimes provide."
        ),
    ),
    SelectedCase(
        row_index=2,
        batch_index=0,
        row=1,
        col=1,
        case_type="T1 cleaner prediction",
        outcome_label="T1 better",
        reason_for_selection=(
            "Both models detect the live knot, but T0 returns several overlapping boxes while "
            "T1 keeps a single cleaner detection. This row represents the 'same target, cleaner "
            "optimization outcome' pattern seen in some validation examples."
        ),
    ),
    SelectedCase(
        row_index=3,
        batch_index=1,
        row=0,
        col=2,
        case_type="Competitive T0 / seam confusion",
        outcome_label="T0 similar/better",
        reason_for_selection=(
            "This small dead-knot case lies next to a strong seam. Both models respond to the "
            "true target, but T1 also expands into a larger false positive region. The example "
            "keeps the figure balanced by showing a setting where source initialization is not "
            "visually stronger."
        ),
    ),
    SelectedCase(
        row_index=4,
        batch_index=0,
        row=0,
        col=1,
        case_type="Low-contrast difficult case",
        outcome_label="Both difficult",
        reason_for_selection=(
            "A weak knot near the image boundary remains difficult. T0 misses the case, while "
            "T1 only partially captures it with imperfect localization. This row reflects the "
            "fact that adaptation helps some cases but does not remove the domain difficulty."
        ),
    ),
)


def _ensure_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def _load_image(path: Path) -> Image.Image:
    _ensure_file(path)
    return Image.open(path).convert("RGB")


def _crop_cell(image: Image.Image, *, row: int, col: int, cell_w: int, cell_h: int) -> Image.Image:
    x = col * cell_w
    y = row * cell_h
    return image.crop((x, y, x + cell_w, y + cell_h))


def _pad_cell(image: Image.Image, *, margin: int = 10) -> Image.Image:
    padded = Image.new("RGB", (image.width + margin * 2, image.height + margin * 2), "white")
    padded.paste(image, (margin, margin))
    return padded


def _stack_row(gt_image: Image.Image, t0_image: Image.Image, t1_image: Image.Image, *, gap: int = 12, margin: int = 10) -> Image.Image:
    cells = [_pad_cell(image, margin=margin) for image in (gt_image, t0_image, t1_image)]
    width = sum(cell.width for cell in cells) + gap * (len(cells) - 1)
    height = max(cell.height for cell in cells)
    row_image = Image.new("RGB", (width, height), "white")
    x_offset = 0
    for cell in cells:
        row_image.paste(cell, (x_offset, 0))
        x_offset += cell.width + gap
    return row_image


def _stack_rows(rows: list[Image.Image], *, gap: int = 12) -> Image.Image:
    width = max(row.width for row in rows)
    height = sum(row.height for row in rows) + gap * (len(rows) - 1)
    stacked = Image.new("RGB", (width, height), "white")
    y_offset = 0
    for row in rows:
        stacked.paste(row, (0, y_offset))
        y_offset += row.height + gap
    return stacked


def _add_headers(image: Image.Image, *, header_height: int = 56, gap: int = 12, margin: int = 10) -> Image.Image:
    cell_width = 480 + margin * 2
    col_width = cell_width
    x_offsets = [
        0,
        col_width + gap,
        col_width * 2 + gap * 2,
    ]
    canvas = Image.new("RGB", (image.width, image.height + header_height), "white")
    canvas.paste(image, (0, header_height))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(str(FONT_PATH), 28)
    labels = ["GT", "T0", "T1"]
    for label, x_offset in zip(labels, x_offsets):
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = x_offset + (col_width - text_width) // 2
        draw.text((text_x, 14), label, font=font, fill="black")
    return canvas


def build_figure() -> dict[str, Path]:
    OUTPUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)

    _ensure_file(T0_DIR / "val_batch0_labels.jpg")
    _ensure_file(T0_DIR / "val_batch0_pred.jpg")
    _ensure_file(T1_DIR / "val_batch0_pred.jpg")

    gt_batch_cache: dict[int, Image.Image] = {}
    t0_batch_cache: dict[int, Image.Image] = {}
    t1_batch_cache: dict[int, Image.Image] = {}
    cell_width = 0
    cell_height = 0
    row_images: list[Image.Image] = []
    for case in SELECTED_CASES:
        gt_panel = T0_DIR / f"val_batch{case.batch_index}_labels.jpg"
        t0_panel = T0_DIR / f"val_batch{case.batch_index}_pred.jpg"
        t1_panel = T1_DIR / f"val_batch{case.batch_index}_pred.jpg"
        for panel in (gt_panel, t0_panel, t1_panel):
            _ensure_file(panel)

        if case.batch_index not in gt_batch_cache:
            gt_batch_cache[case.batch_index] = _load_image(gt_panel)
            t0_batch_cache[case.batch_index] = _load_image(t0_panel)
            t1_batch_cache[case.batch_index] = _load_image(t1_panel)
            batch_width, batch_height = gt_batch_cache[case.batch_index].size
            cell_width = batch_width // 4
            cell_height = batch_height // 4

        gt_crop = _crop_cell(gt_batch_cache[case.batch_index], row=case.row, col=case.col, cell_w=cell_width, cell_h=cell_height)
        t0_crop = _crop_cell(t0_batch_cache[case.batch_index], row=case.row, col=case.col, cell_w=cell_width, cell_h=cell_height)
        t1_crop = _crop_cell(t1_batch_cache[case.batch_index], row=case.row, col=case.col, cell_w=cell_width, cell_h=cell_height)
        row_images.append(_stack_row(gt_crop, t0_crop, t1_crop))

    final_png = OUTPUT_FIG_DIR / "vnwoodknot_transfer_qualitative_validation_seed42.png"
    final_pdf = OUTPUT_FIG_DIR / "vnwoodknot_transfer_qualitative_validation_seed42.pdf"
    stacked = _stack_rows(row_images)
    figure = _add_headers(stacked)
    figure.save(final_png)
    figure.save(final_pdf, "PDF", resolution=300.0)

    return {
        "png": final_png,
        "pdf": final_pdf,
    }


def write_manifest() -> dict[str, Path]:
    manifest_json = OUTPUT_TABLE_DIR / "vnwoodknot_transfer_qualitative_manifest.json"
    manifest_csv = OUTPUT_TABLE_DIR / "vnwoodknot_transfer_qualitative_manifest.csv"
    note_txt = OUTPUT_TABLE_DIR / "vnwoodknot_transfer_qualitative_note.txt"

    rows = []
    for case in SELECTED_CASES:
        rows.append(
            {
                "row_index": case.row_index,
                "panel_ref": case.panel_ref,
                "split": "validation",
                "source_run_t0": T0_RUN,
                "source_run_t1": T1_RUN,
                "case_type": case.case_type,
                "outcome_label": case.outcome_label,
                "filename_hint": case.filename_hint,
                "image_id": None,
                "reason_for_selection": case.reason_for_selection,
            }
        )

    manifest_payload = {
        "figure_scope": "VNWoodKnot target-domain qualitative comparison",
        "runs_used": {
            "t0": T0_RUN,
            "t1": T1_RUN,
        },
        "selection_basis": (
            "Local retained artifacts only preserve validation-side val_batch panels for the main seed-42 runs. "
            "Held-out-test per-image predictions are not retained locally, so the qualitative panel is drawn from "
            "validation examples while the mixed held-out-test conclusion remains supported by the repeated-seed "
            "quantitative tables."
        ),
        "rows": rows,
    }
    manifest_json.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

    with manifest_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    note_txt.write_text(
        (
            "The figure compares retained validation-side VNWoodKnot examples from the main seed-42 T0 and T1 runs. "
            "Two rows show cases where source initialization improves case-level behavior, one row shows a seam-driven "
            "example where T0 is competitive or cleaner, and one row remains difficult for both models. This supports "
            "the paper's balanced interpretation: source initialization can help optimization and some individual cases, "
            "but it does not imply a uniformly stronger held-out-test detector under the current 50-epoch adaptation budget."
        ),
        encoding="utf-8",
    )

    return {
        "json": manifest_json,
        "csv": manifest_csv,
        "note": note_txt,
    }


def main() -> None:
    if not FONT_PATH.exists():
        raise FileNotFoundError(f"Font file does not exist: {FONT_PATH}")

    figure_paths = build_figure()
    manifest_paths = write_manifest()

    summary = {
        "figure_png": str(figure_paths["png"]),
        "figure_pdf": str(figure_paths["pdf"]) if figure_paths["pdf"].exists() else None,
        "manifest_json": str(manifest_paths["json"]),
        "manifest_csv": str(manifest_paths["csv"]),
        "note_txt": str(manifest_paths["note"]),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
