"""Replace hardware identifiers in short Note 2 figures, without refitting."""
import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pypdf import PdfReader, PdfWriter

from prepare_note1_labels import remove_labels

ROOT = Path(__file__).resolve().parent
NAMES = ("slope_ch150", "comparison_ch137", "comparison_ch154",
         "original_figure5", "awc_summary_selected")


def prepare():
    output = ROOT / "assets/note2_public_labels"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for name in NAMES:
        source = ROOT.parent / "note2_dual_threshold/figures" / f"{name}.pdf"
        if name == "original_figure5":
            source = ROOT / "assets/note2_figure5/figure5.pdf"
        elif name == "awc_summary_selected":
            source = ROOT / "assets/note2_internal_backups/awc_summary_selected.pdf"
        page = PdfReader(source).pages[0]
        width, height = float(page.mediabox.width), float(page.mediabox.height)
        fig = plt.figure(figsize=(width / 72, height / 72))
        fig.patch.set_alpha(0)
        if name == "slope_ch150":
            replacements = {"Run 4076 | Channel 150": "Dual-threshold sample | Bar 8, right SiPM"}
        elif name.startswith("comparison_"):
            channel = "137" if name.endswith("137") else "154"
            side = "Bar 8, left SiPM" if channel == "137" else "Bar 7, right SiPM"
            replacements = {
                f"Channel {channel} | Joint measured support only": f"{side} | Common measured support",
                "MCP quadratic (run 4399)": "MCP-reference quadratic",
                "AWC/S(Q) + offset (run 4076)": r"$\mathrm{AWC}/\overline{S}(Q)$ + offset",
            }
        elif name == "awc_summary_selected":
            replacements = {str(ch): "" for ch in (159, 156, 155, 154, 151, 150, 137, 135, 134, 133)}
            # Point identifiers are 9 pt; axis ticks (including 150) are 12 pt.
            remove_labels(page, replacements, font_size=9)
        else:
            replacements = {}
            # Cover only the raster title above the axes, retaining original image bytes.
            box = (.25, .897, .55, .035)
            fig.add_artist(Rectangle(box[:2], box[2], box[3], transform=fig.transFigure,
                                     color="white", linewidth=0))
            fig.text(.526, .91, "Bar 8, left SiPM: residual after walk correction",
                     ha="center", va="center", fontsize=11)
        if name not in ("awc_summary_selected", "original_figure5"):
            for old, x, y, size in remove_labels(page, replacements):
                title = old.startswith(("Run ", "Channel "))
                fig.text(.5 if title else x / width, y / height, replacements[old],
                         ha="center" if title else "left", va="baseline", fontsize=size)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="pdf", transparent=True)
        plt.close(fig)
        buffer.seek(0)
        page.merge_page(PdfReader(buffer).pages[0])
        target = output / f"{name}.pdf"
        writer = PdfWriter()
        writer.add_page(page)
        writer.write(target)
        if name == "original_figure5":
            before_images = [img.data for img in PdfReader(source).pages[0].images]
            after_images = [img.data for img in PdfReader(target).pages[0].images]
            assert before_images == after_images
        records.append({"figure": target.name, "source": str(source.relative_to(ROOT.parent)),
                        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                        "text_replacements": replacements,
                        "raster_title_mask": box if name == "original_figure5" else None})
    manifest = {"no_refitting": True, "numerical_results_unchanged": True,
                "mapping_source": "TimeCalibration/channel_mapping.py",
                "reference_module": {"bar7_left": 133, "bar7_right": 154,
                                     "bar8_left": 137, "bar8_right": 150},
                "figures": records}
    (output / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    prepare()
