"""Remove only the CMS Preliminary text object from exploratory backup figures."""
import hashlib
import json
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from pypdf.generic import ArrayObject

ROOT = Path(__file__).resolve().parent
NAMES = ("awc_summary_selected", "internal_predictor")


def prepare(note="note2_dual_threshold", names=NAMES, source_dir=None):
    output = ROOT / f"assets/{note.split('_')[0]}_internal_backups"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for name in names:
        source = (source_dir or ROOT.parent / note / "figures") / f"{name}.pdf"
        page = PdfReader(source).pages[0]
        before = page.extract_text()
        content = page.get_contents()
        remove = []
        shortened = 0
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
                elif text.startswith("CMS Preliminary Phase 2 | "):
                    # Keep the setting and event-count annotation on diagnostic plots.
                    text_ops = [j for j in range(start, index)
                                if content.operations[j][1] in (b"Tj", b"TJ")]
                    assert len(text_ops) == 1
                    args, operator = content.operations[text_ops[0]]
                    assert operator == b"TJ"
                    prefix = "CMS Preliminary Phase 2 | "
                    consumed = ""
                    for j, item in enumerate(args[0]):
                        if isinstance(item, str):
                            if consumed == prefix:
                                break
                            consumed += str(item)
                            assert prefix.startswith(consumed)
                    assert consumed == prefix
                    # Retain original font-encoded string objects and suffix kerning.
                    content.operations[text_ops[0]] = ([ArrayObject(args[0][j:])], b"TJ")
                    shortened += 1
                start = None
        assert len(remove) + shortened == 1, (source, remove, shortened)
        if remove:
            start, end = remove[0]
            content.operations = content.operations[:start] + content.operations[end:]
        page.replace_contents(content)
        writer = PdfWriter()
        writer.add_page(page)
        target = output / f"{name}.pdf"
        writer.write(target)
        after = PdfReader(target).pages[0].extract_text()
        normalize = lambda value: "".join(value.split())
        removed = "CMS Preliminary Phase 2 | " if shortened else "CMS Preliminary Phase 2"
        assert normalize(after) == normalize(before.replace(removed, ""))
        assert "Preliminary" not in after
        records.append({"source": str(source.relative_to(ROOT.parent)),
                        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "figure": target.name,
                        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                        "change": "Removed CMS Preliminary label; remaining text and plot drawing operations unchanged."})
    (output / "provenance.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    prepare()
