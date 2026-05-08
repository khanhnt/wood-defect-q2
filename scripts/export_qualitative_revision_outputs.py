#!/usr/bin/env python3
"""Export both qualitative revision figure groups with a publication-ready folder layout."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_DOMAIN_SCRIPT = PROJECT_ROOT / "scripts" / "build_in_domain_qualitative_figure.py"
VN_SCRIPT = PROJECT_ROOT / "scripts" / "build_vnwoodknot_qualitative_from_predictions.py"


def _fresh_dir(path: Path) -> Path:
    if not path.exists() or not any(path.iterdir()):
        path.mkdir(parents=True, exist_ok=True)
        return path
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    fresh = path.parent / f"{path.name}_{timestamp}"
    fresh.mkdir(parents=True, exist_ok=True)
    return fresh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=str, required=True, help="Root directory for qualitative_revision_outputs.")

    parser.add_argument("--in-manifest", type=str, required=True)
    parser.add_argument("--in-image-root-dir", type=str, default="")
    parser.add_argument("--in-split", type=str, default="test")
    parser.add_argument("--in-rows", type=int, default=4)
    parser.add_argument("--in-panel-width", type=int, default=360)
    parser.add_argument("--in-panel-height", type=int, default=220)
    parser.add_argument("--in-show-scores", action="store_true")
    parser.add_argument("--baseline-run-name", type=str, required=True)
    parser.add_argument("--baseline-header", type=str, default="Faster R-CNN")
    parser.add_argument("--baseline-predictions", type=str, required=True)
    parser.add_argument("--baseline-checkpoint", type=str, default="")
    parser.add_argument("--yolo-run-name", type=str, required=True)
    parser.add_argument("--yolo-header", type=str, default="YOLOv8s")
    parser.add_argument("--yolo-predictions", type=str, required=True)
    parser.add_argument("--yolo-checkpoint", type=str, default="")
    parser.add_argument("--variant-primary-run-name", type=str, required=True)
    parser.add_argument("--variant-primary-header", type=str, default="YOLO variant")
    parser.add_argument("--variant-primary-predictions", type=str, required=True)
    parser.add_argument("--variant-primary-checkpoint", type=str, default="")
    parser.add_argument("--variant-fallback-run-name", type=str, default="")
    parser.add_argument("--variant-fallback-header", type=str, default="YOLO P2")
    parser.add_argument("--variant-fallback-predictions", type=str, default="")
    parser.add_argument("--variant-fallback-checkpoint", type=str, default="")

    parser.add_argument("--vn-manifest", type=str, required=True)
    parser.add_argument("--vn-image-root-dir", type=str, default="")
    parser.add_argument("--vn-split", type=str, default="test")
    parser.add_argument("--vn-rows", type=int, default=4)
    parser.add_argument("--vn-panel-width", type=int, default=360)
    parser.add_argument("--vn-panel-height", type=int, default=240)
    parser.add_argument("--vn-show-scores", action="store_true")
    parser.add_argument("--vn-replace-row3-with-moderate-t1", action="store_true")
    parser.add_argument("--t0-run-name", type=str, required=True)
    parser.add_argument("--t0-header", type=str, default="T0")
    parser.add_argument("--t0-predictions", type=str, required=True)
    parser.add_argument("--t0-checkpoint", type=str, default="")
    parser.add_argument("--t1-run-name", type=str, required=True)
    parser.add_argument("--t1-header", type=str, default="T1")
    parser.add_argument("--t1-predictions", type=str, required=True)
    parser.add_argument("--t1-checkpoint", type=str, default="")
    return parser.parse_args()


def _pick_variant(args: argparse.Namespace) -> dict[str, str]:
    primary_predictions = Path(args.variant_primary_predictions)
    if primary_predictions.exists():
        return {
            "run_name": args.variant_primary_run_name,
            "header": args.variant_primary_header,
            "predictions": str(primary_predictions),
            "checkpoint": args.variant_primary_checkpoint,
            "selection_note": "Used preferred variant candidate because its predictions artifact exists.",
        }

    if args.variant_fallback_predictions and Path(args.variant_fallback_predictions).exists():
        return {
            "run_name": args.variant_fallback_run_name,
            "header": args.variant_fallback_header,
            "predictions": args.variant_fallback_predictions,
            "checkpoint": args.variant_fallback_checkpoint,
            "selection_note": "Fell back to the secondary variant because the preferred artifact was unavailable.",
        }

    raise FileNotFoundError(
        "Could not find a valid YOLO variant predictions artifact. "
        f"Checked primary={args.variant_primary_predictions!r} and fallback={args.variant_fallback_predictions!r}."
    )


def _run_json_command(command: list[str]) -> dict[str, object]:
    result = subprocess.run(command, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)
    if result.stderr.strip():
        print(result.stderr.strip(), file=sys.stderr)
    stdout = result.stdout.strip()
    if stdout:
        print(stdout)
    return json.loads(stdout)


def main() -> None:
    args = parse_args()
    output_root = _fresh_dir(Path(args.output_root))
    top_scripts_dir = output_root / "scripts"
    top_scripts_dir.mkdir(parents=True, exist_ok=True)
    script_copy_path = top_scripts_dir / Path(__file__).name
    shutil.copy2(Path(__file__), script_copy_path)
    reproduce_path = top_scripts_dir / "reproduce.sh"
    reproduce_command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    reproduce_path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n" + " ".join(json.dumps(token) for token in reproduce_command) + "\n",
        encoding="utf-8",
    )
    reproduce_path.chmod(0o755)

    variant = _pick_variant(args)
    in_output = output_root / "in_domain"
    vn_output = output_root / "vnwoodknot"

    in_command = [
        sys.executable,
        str(IN_DOMAIN_SCRIPT),
        "--manifest",
        args.in_manifest,
        "--output-dir",
        str(in_output),
        "--split",
        args.in_split,
        "--rows",
        str(args.in_rows),
        "--panel-width",
        str(args.in_panel_width),
        "--panel-height",
        str(args.in_panel_height),
        "--baseline-run-name",
        args.baseline_run_name,
        "--baseline-header",
        args.baseline_header,
        "--baseline-predictions",
        args.baseline_predictions,
        "--baseline-checkpoint",
        args.baseline_checkpoint,
        "--yolo-run-name",
        args.yolo_run_name,
        "--yolo-header",
        args.yolo_header,
        "--yolo-predictions",
        args.yolo_predictions,
        "--yolo-checkpoint",
        args.yolo_checkpoint,
        "--variant-run-name",
        variant["run_name"],
        "--variant-header",
        variant["header"],
        "--variant-predictions",
        variant["predictions"],
        "--variant-checkpoint",
        variant["checkpoint"],
    ]
    if args.in_image_root_dir:
        in_command.extend(["--image-root-dir", args.in_image_root_dir])
    if args.in_show_scores:
        in_command.append("--show-scores")

    vn_command = [
        sys.executable,
        str(VN_SCRIPT),
        "--manifest",
        args.vn_manifest,
        "--output-dir",
        str(vn_output),
        "--split",
        args.vn_split,
        "--rows",
        str(args.vn_rows),
        "--panel-width",
        str(args.vn_panel_width),
        "--panel-height",
        str(args.vn_panel_height),
        "--t0-run-name",
        args.t0_run_name,
        "--t0-header",
        args.t0_header,
        "--t0-predictions",
        args.t0_predictions,
        "--t0-checkpoint",
        args.t0_checkpoint,
        "--t1-run-name",
        args.t1_run_name,
        "--t1-header",
        args.t1_header,
        "--t1-predictions",
        args.t1_predictions,
        "--t1-checkpoint",
        args.t1_checkpoint,
    ]
    if args.vn_image_root_dir:
        vn_command.extend(["--image-root-dir", args.vn_image_root_dir])
    if args.vn_show_scores:
        vn_command.append("--show-scores")
    if args.vn_replace_row3_with_moderate_t1:
        vn_command.append("--replace-row3-with-moderate-t1")

    in_summary = _run_json_command(in_command)
    vn_summary = _run_json_command(vn_command)

    summary_note = output_root / "revision_summary.txt"
    summary_note.write_text(
        "\n".join(
            [
                "Qualitative revision export summary",
                "",
                "In-domain figure",
                f"- Output root: {in_summary['output_root']}",
                f"- Baseline run: {args.baseline_run_name}",
                f"- YOLOv8s run: {args.yolo_run_name}",
                f"- YOLO variant run used: {variant['run_name']}",
                f"- Variant selection note: {variant['selection_note']}",
                "",
                "VNWoodKnot figure",
                f"- Output root: {vn_summary['output_root']}",
                f"- T0 run: {args.t0_run_name}",
                f"- T1 run: {args.t1_run_name}",
                f"- Row 3 replacement enabled: {args.vn_replace_row3_with_moderate_t1}",
                "",
                "Selection intent",
                "- In-domain: balanced readable rows spanning easy correct, texture FP, small/weak defect, and disagreement cases.",
                "- VNWoodKnot: two T1-better rows, one T0-similar/better row, and one difficult row.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "in_domain": in_summary,
                "vnwoodknot": vn_summary,
                "summary_note": str(summary_note),
                "variant_used": variant,
                "script_copy": str(script_copy_path),
                "reproduce_sh": str(reproduce_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
