"""Relabel the saved Figure 5 plotting field; do not reconstruct or refit data."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "assets/note2_figure5/original_figure5.jpg"
OUTPUT = ROOT / "assets/note2_residual_envelope"


def prepare():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    original_record = json.loads((SOURCE.parent / "provenance.json").read_text())
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert digest == original_record["image_sha256"]
    # Register against all labelled grid lines, then crop only the old axes/text.
    x_pixels = np.array([127.5, 370.8, 614.3, 857.7, 1101.4, 1344.7, 1588.0])
    q_ticks = np.arange(150, 451, 50)
    y_pixels = np.array([130.2, 303.8, 477.6, 651.4, 825.2])
    ps_ticks = np.array([20, 10, 0, -10, -20])
    x_map = np.polyfit(x_pixels, q_ticks, 1)
    y_map = np.polyfit(y_pixels, ps_ticks, 1)
    crop = (129, 42, 1587, 941)
    left, top, right, bottom = crop
    extent = [float(np.polyval(x_map, left - .5)),
              float(np.polyval(x_map, right - .5)),
              float(np.polyval(y_map, bottom - .5)),
              float(np.polyval(y_map, top - .5))]
    with Image.open(SOURCE) as source:
        assert source.size == (1630, 1030)
        field = np.asarray(source.convert("RGB"))[top:bottom, left:right].copy()

    with plt.rc_context({"pdf.fonttype": 42, "font.size": 12}):
        fig, ax = plt.subplots(figsize=(8.4, 5.2))
        fig.subplots_adjust(left=.14, right=.98, bottom=.15, top=.78)
        ax.imshow(field, extent=extent, origin="upper", aspect="auto",
                  interpolation="none", zorder=1)
        ax.axhspan(-10, 10, color="#359c79", alpha=.10, zorder=2)
        for y in (-10, 10):
            ax.axhline(y, color="#23795d", ls="--", lw=1.2, zorder=3)
        ax.set(xlim=(150, 450), ylim=(-28, 28), xticks=q_ticks,
               yticks=[-20, -10, 0, 10, 20], xlabel=r"Charge proxy $Q$ [ADC]",
               ylabel=r"Residual $r(Q)$ [ps]")
        ax.tick_params(direction="out")
        blue = "#5b98bb"
        handles = [Line2D([], [], marker="o", color=blue, ls="none", label="Residual points"),
                   Line2D([], [], color=blue, lw=2, label="Original linear fit"),
                   Patch(facecolor="#359c79", alpha=.2, label=r"$\pm10$ ps guide")]
        ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, 1.015),
                  ncol=3, frameon=False, fontsize=10, handlelength=1.7, columnspacing=1.3)
        fig.text(.14, .97, r"$\bf{CMS}$ Preliminary Phase 2", fontsize=14, va="top")
        fig.text(.14, .90, "Reference module, bar 8, left SiPM", fontsize=12, va="top")
        fig.savefig(OUTPUT / "awc_residual_envelope.pdf")
        fig.savefig(OUTPUT / "awc_residual_envelope.png", dpi=180)
        plt.close(fig)

    record = {
        "source": str(SOURCE.relative_to(ROOT)), "source_sha256": digest,
        "source_figure": "Supplied detector Note 2, Figure 5",
        "treatment": "Original raster plotting field preserved; axes redrawn in ps with a +/-10 ps guide. No point digitization, fit, centering or event processing.",
        "crop_pixels": crop, "extent_q_ps": extent,
        "q_tick_registration_pixels": x_pixels.tolist(),
        "ps_tick_registration_pixels": y_pixels.tolist(),
        "max_registration_residual_ps": float(np.max(np.abs(np.polyval(y_map, y_pixels) - ps_ticks))),
        "ps_per_pixel": float(abs(y_map[0])),
        "raster_field_sha256": hashlib.sha256(field.tobytes()).hexdigest(),
        "residual_definition": "r(Q) = binned MCP-referenced leading-edge time - AWC / mean_slope(Q) - constant offset",
        "envelope_ps": [-10, 10],
        "envelope_interpretation": "Visual residual-walk scale guide, not a confidence band, all-point bound, or single-hit timing resolution.",
        "uncertainty": original_record["uncertainty"],
        "conclusion": "The nearly flat original residual trend supports the AWC form at roughly the 10 ps scale in this example; some points and error bars extend beyond the guide.",
    }
    (OUTPUT / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    prepare()
