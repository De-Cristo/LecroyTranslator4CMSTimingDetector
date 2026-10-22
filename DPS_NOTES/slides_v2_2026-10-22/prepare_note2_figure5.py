"""Extract original Note 2 Figure 5 unchanged and add a CMS figure header."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets/note2_figure5"


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    source = ROOT.parent / "Note2.pdf"
    jpeg = ASSETS / "original_figure5.jpg"
    provenance = ASSETS / "provenance.json"
    if source.exists():
        page = PdfReader(source).pages[11]
        assert len(page.images) == 1
        original = page.images[0]
        assert original.image.size == (1630, 1030)
        jpeg.write_bytes(original.data)
        record = {
            "source": "DPS_NOTES/Note2.pdf",
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "pdf_page": 12,
            "printed_page": 11,
            "figure": 5,
            "image": original.name,
            "image_sha256": hashlib.sha256(original.data).hexdigest(),
            "treatment": "Original embedded image unchanged; CMS label added outside image.",
            "uncertainty": "Original error bars preserved; their statistical definition is not specified in the supplied Figure 5 caption. No numerical refit or uncertainty interpretation added.",
        }
        provenance.write_text(json.dumps(record, indent=2) + "\n")
    record = json.loads(provenance.read_text())
    assert hashlib.sha256(jpeg.read_bytes()).hexdigest() == record["image_sha256"]
    plt.rcParams["pdf.fonttype"] = 42
    fig = plt.figure(figsize=(8.15, 5.55))
    ax = fig.add_axes([0, 0, 1, 5.15 / 5.55])
    with Image.open(jpeg) as original:
        ax.imshow(original, interpolation="none")
    ax.set_axis_off()
    fig.text(.10, .975, r"$\bf{CMS}$ Preliminary Phase 2", fontsize=14, va="top")
    fig.savefig(ASSETS / "figure5.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
