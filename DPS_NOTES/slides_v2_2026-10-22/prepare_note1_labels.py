"""Relabel short-deck figures without changing data, fits, or full-note assets."""
import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "assets/note1_public_labels"
THRESHOLDS = ROOT.parent.parent / "Q_Walk/results/2026-10-22_bar_pair_by_threshold"
SOURCES = {
    "setup": ROOT.parent / "note1_mcp_reference/figures/setup.pdf",
    "event_matching": ROOT.parent / "note1_mcp_reference/figures/event_matching.pdf",
    "threshold_walk_profiles": THRESHOLDS / "vth1_3_vth2_60/upstream_walk_profiles.pdf",
    "threshold_histograms": THRESHOLDS / "before_after_all_thresholds.pdf",
}
NAMES = tuple(SOURCES)


def remove_labels(page, replacements, font_size=None):
    """Remove complete text objects only; retain all plot drawing operations."""
    positions = []

    def visit(text, cm, tm, font, size):
        # This source PDF stores the rotated label's ff ligature separately.
        if text.strip() == "Time di" and "Time difference [ns]" in replacements:
            text = "Time difference [ns]"
        if text.strip() == "AWC/S(Q) + o" and "AWC/S(Q) + offset (run 4076)" in replacements:
            text = "AWC/S(Q) + offset (run 4076)"
        if text.strip() in replacements and (font_size is None or size == font_size):
            x = tm[4] * cm[0] + tm[5] * cm[2] + cm[4]
            y = tm[4] * cm[1] + tm[5] * cm[3] + cm[5]
            positions.append((text.strip(), x, y, size))

    page.extract_text(visitor_text=visit)
    content = page.get_contents()
    remove = []
    start = None
    size = None
    for index, (args, operator) in enumerate(content.operations):
        if operator == b"BT":
            start, text = index, ""
        elif operator == b"Tf":
            size = float(args[1])
        elif start is not None and operator == b"Tj":
            text += str(args[0])
        elif start is not None and operator == b"TJ":
            text += "".join(str(item) for item in args[0] if isinstance(item, str))
        elif operator == b"ET" and start is not None:
            if text.strip() == "Time dierence [ns]":
                text = "Time difference [ns]"
            if text.startswith("AWC/S(Q)"):
                text = text.replace("\x01", "ff")
            if text.strip() in replacements and (font_size is None or size == font_size):
                remove.append((start, index + 1))
            start = None
    assert len(remove) == len(positions)
    assert set(replacements) == {position[0] for position in positions}
    for start, end in reversed(remove):
        del content.operations[start:end]
    page.replace_contents(content)
    return positions


def prepare():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = []
    for name, source in SOURCES.items():
        page = PdfReader(source).pages[0]
        width, height = float(page.mediabox.width), float(page.mediabox.height)
        fig = plt.figure(figsize=(width / 72, height / 72))
        fig.patch.set_alpha(0)
        changes = []
        if name.startswith("threshold_"):
            if name == "threshold_walk_profiles":
                replacements = {
                    "Channel 137": "Channel left",
                    "Channel 150": "Channel right",
                    "Time difference [ns]": "bar-end time minus clock-aligned MCP time [ns]",
                    "CMS Preliminary Phase 2 | Upstream bar 8 | T1=3, T2=60 DAC":
                    "CMS Preliminary Phase 2 | Reference module, bar 8 | T1=3, T2=60 DAC",
                }
            else:
                replacements = {
                    "Downstream - reference bar [ps], centered":
                    "Module under test - reference bar [ps], centered",
                    "CMS Preliminary Phase 2 | Run 4399 | T2=60 DAC":
                    "CMS Preliminary Phase 2 | T2=60 DAC",
                }
            for old, x, y, size in remove_labels(page, replacements):
                if old == "Time difference [ns]":
                    side = "left" if x < width / 2 else "right"
                    fig.text(x / width, .5, rf"$t_{{\rm {side}}}-t_{{\rm MCP}}$ [ns]",
                             ha="center", va="center", rotation=90, fontsize=size)
                    continue
                title = old.startswith("CMS")
                fig.text(.5 if title else x / width, y / height, replacements[old],
                         ha="center" if title else "left", va="baseline",
                         fontsize=size if "Downstream" not in old else size * .88)
            changes = list(replacements.items())
        else:
            # The original raster is retained. Masks cover label-only regions,
            # never plotted data, ticks, curves, or hardware connections.
            def label(box, text, point, size, color="white", rotation=0):
                fig.add_artist(Rectangle(box[:2], box[2], box[3],
                                         transform=fig.transFigure, color=color, linewidth=0))
                fig.text(*point, text, ha="center", va="center", fontsize=size,
                         rotation=rotation, linespacing=1.05)
                changes.append(dict(box=box, label=text))

            if name == "setup":
                label((.435, .452, .066, .062), "Reference\nmodule", (.468, .483), 8.5, "#eeeeee")
                label((.508, .458, .072, .061), "Module\nunder test", (.544, .488), 8.5, "#eeeeee")
                label((.664, .039, .238, .121), "Scope trigger\noutput to TOFHIR\ntrigger channel",
                      (.783, .100), 12, "#82d34e")
            else:
                label((.017, .140, .035, .660),
                      r"$(\Delta t_{\rm FEBd}/\Delta t_{\rm scope}-1)\times10^5$",
                      (.035, .485), 11, rotation=90)
                label((.620, .060, .245, .041),
                      r"$\Delta t_{\rm FEBd}/\Delta t_{\rm scope}$", (.760, .079), 10)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="pdf", transparent=True)
        plt.close(fig)
        buffer.seek(0)
        page.merge_page(PdfReader(buffer).pages[0])
        target = OUTPUT / f"{name}.pdf"
        writer = PdfWriter()
        writer.add_page(page)
        writer.write(target)
        records.append(dict(figure=target.name, source=str(source.relative_to(ROOT.parent.parent)),
                            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                            sha256=hashlib.sha256(target.read_bytes()).hexdigest(), changes=changes))
    manifest = dict(no_refitting=True, numerical_results_unchanged=True,
                    mapping_source="TimeCalibration/channel_mapping.py",
                    mapping={"reference_bar8": {"left": 137, "right": 150},
                             "module_under_test_bar3": {"left": 234, "right": 243}},
                    figures=records)
    (OUTPUT / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    prepare()
