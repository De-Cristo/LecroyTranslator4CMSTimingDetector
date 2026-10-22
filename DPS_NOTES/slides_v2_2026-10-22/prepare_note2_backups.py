"""Remove only the CMS Preliminary text object from exploratory backup figures."""
import hashlib
import json
from pathlib import Path

from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parent
NAMES = ("awc_summary_selected", "internal_predictor")


def prepare():
    output = ROOT / "assets/note2_internal_backups"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for name in NAMES:
        source = ROOT.parent / "note2_dual_threshold/figures" / f"{name}.pdf"
        page = PdfReader(source).pages[0]
        before = page.extract_text()
        content = page.get_contents()
        remove = []
        start = None
        for index, (args, operator) in enumerate(content.operations):
            if operator == b"BT":
                start = index
                text = ""
            elif start is not None and operator == b"Tj":
                text += str(args[0])
            elif start is not None and operator == b"TJ":
                text += "".join(str(item) for item in args[0] if isinstance(item, str))
            elif operator == b"ET" and start is not None:
                if text == "CMS Preliminary Phase 2":
                    remove.append((start, index + 1))
                start = None
        assert len(remove) == 1, (source, remove)
        start, end = remove[0]
        content.operations = content.operations[:start] + content.operations[end:]
        page.replace_contents(content)
        writer = PdfWriter()
        writer.add_page(page)
        target = output / f"{name}.pdf"
        writer.write(target)
        after = PdfReader(target).pages[0].extract_text()
        normalize = lambda value: "".join(value.split())
        assert normalize(after) == normalize(before.replace("CMS Preliminary Phase 2", ""))
        assert "Preliminary" not in after
        records.append({"source": str(source.relative_to(ROOT.parent)),
                        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "figure": target.name,
                        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                        "change": "Removed exactly one CMS Preliminary text object; plot drawing operations unchanged."})
    (output / "provenance.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    prepare()
