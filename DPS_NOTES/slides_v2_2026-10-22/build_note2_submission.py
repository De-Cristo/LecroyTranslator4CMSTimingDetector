"""Compile submission-format Note 2 without changing the scientific content."""
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
NOTE = ROOT / "note2_dual_threshold"


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
    assert len(reader.pages) == 11
    text = "\n".join(page.extract_text() for page in reader.pages)
    for phrase in ("draft", "internal review", "internal discussion", "not approved",
                   "pending approval", "not for public"):
        assert phrase not in (text + str(reader.metadata)).lower(), phrase
    assert "Abstract" in reader.pages[1].extract_text()
    assert "References" in reader.pages[10].extract_text()
    assert "cms-dpg-conveners-mtd@cern.ch" in reader.pages[0].extract_text()
    for number in (5, 7, 8, 9, 10):
        assert "Preliminary Phase 2" in reader.pages[number - 1].extract_text(), number
    citations = {dest: reader.get_destination_page_number(value) + 1
                 for dest, value in reader.named_destinations.items()
                 if dest.startswith("beamerbib")}
    assert len(citations) == 11 and set(citations.values()) == {11}
    audit = json.loads((NOTE / "citation_audit.json").read_text())
    for entry in audit["references"]:
        dest = "beamerbib" + entry["key"]
        for slide in entry["body_slides"]:
            links = [a.get_object() for a in reader.pages[slide - 1].get("/Annots", [])]
            assert any(str(a.get("/A", {}).get("/D", "")) == dest
                       or str(a.get("/Dest", "")) == dest for a in links), (dest, slide)
    review = PdfReader(NOTE / "slides.pdf")
    assert len(review.pages) == len(reader.pages)

    def content(page):
        value = re.sub(r"\s+", "", page.extract_text())
        for label in ("Internal review draft \u2013 22 October 2026", "V2 slides \u2013 draft 1",
                      "Discussion draft; not approved for public release."):
            value = value.replace(re.sub(r"\s+", "", label), "")
        return value

    for index, page in enumerate(reader.pages):
        assert content(page) == content(review.pages[index]), index + 1
    output = ROOT / "submission"
    output.mkdir(exist_ok=True)
    target = output / "note2_dual_threshold.pdf"
    shutil.copy2(build / "submission.pdf", target)
    report = {
        "pages": 11, "reference_page": 11, "backup_pages": 0,
        "cover_pages": 0, "review_notices_removed": True,
        "scientific_content_unchanged": True, "cms_preliminary_labels_retained": True,
        "citation_destinations": citations, "body_citation_links_verified": True,
        "overflow": 0, "undefined_citations": 0,
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "source": "note2_dual_threshold/submission.tex",
    }
    (output / "note2_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
