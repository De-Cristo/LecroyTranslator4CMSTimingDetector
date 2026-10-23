"""Compile the submission presentation from Note 1 without altering review PDFs."""
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
NOTE = ROOT / "note1_mcp_reference"


def main():
    build = NOTE / "build/submission"
    build.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
             "-output-directory=build/submission", "submission.tex"],
            cwd=NOTE, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        if result.returncode:
            raise RuntimeError(result.stdout[-6000:])
    log = (build / "submission.log").read_text(errors="replace")
    problems = [line for line in log.splitlines()
                if any(word in line for word in ("Overfull", "Missing character", "Font Warning"))
                or "undefined" in line.lower()]
    assert not problems, problems
    reader = PdfReader(build / "submission.pdf")
    assert len(reader.pages) == 12
    text = "\n".join(page.extract_text() for page in reader.pages)
    for phrase in ("draft", "internal review", "internal discussion", "not approved",
                   "pending approval", "not for public"):
        assert phrase not in (text + str(reader.metadata)).lower(), phrase
    assert "Abstract" in reader.pages[1].extract_text()
    assert "References" in reader.pages[10].extract_text()
    assert "Backup: left-minus-right timing check" in reader.pages[11].extract_text()
    assert "cms-dpg-conveners-mtd@cern.ch" in reader.pages[0].extract_text()
    for number in (3, 4, 5, 6, 8, 9, 10):
        assert "Preliminary Phase 2" in reader.pages[number - 1].extract_text(), number
    citations = {dest: reader.get_destination_page_number(value) + 1
                 for dest, value in reader.named_destinations.items()
                 if dest.startswith("beamerbib")}
    assert len(citations) == 7 and set(citations.values()) == {11}
    review = PdfReader(NOTE / "slides.pdf")

    def content(page):
        value = re.sub(r"\s+", "", page.extract_text())
        for label in ("Internal review draft \u2013 22 October 2026", "V2 slides \u2013 draft 1",
                      "Discussion draft; not approved for public release.",
                      "Internal discussion only \u2013 diagnostic, not for public presentation"):
            value = value.replace(re.sub(r"\s+", "", label), "")
        return re.sub(r"\d+/(?:11|12)$", "", value)

    for index, page in enumerate(reader.pages):
        assert content(page) == content(review.pages[index]), index + 1
    output = ROOT / "submission"
    output.mkdir(exist_ok=True)
    target = output / "note1_mcp_reference.pdf"
    shutil.copy2(build / "submission.pdf", target)
    report = {
        "pages": 12, "reference_page": 11, "backup_pages": 1,
        "cover_pages": 0, "review_notices_removed": True,
        "scientific_content_unchanged": True, "cms_preliminary_labels_retained": True,
        "citation_destinations": citations, "overflow": 0, "undefined_citations": 0,
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "source": "note1_mcp_reference/submission.tex",
    }
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
