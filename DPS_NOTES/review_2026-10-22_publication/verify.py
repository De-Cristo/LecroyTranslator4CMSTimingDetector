"""Check the N1-6 documentation-only candidate packaging."""
import hashlib
import json
import re
import unicodedata
import zipfile
from pathlib import Path

from pypdf import PdfReader

OUT = Path(__file__).resolve().parent
DPS = OUT.parent


def text(page):
    return unicodedata.normalize("NFKC", page.extract_text())


def main():
    reports = {}
    for name in ("note1_mcp_reference", "note2_dual_threshold"):
        note = DPS / name
        source = (note / "main.tex").read_text()
        references = (note / "references.tex").read_text()
        assert references == (DPS / "draft_support/references.tex").read_text()
        cited = {key for group in re.findall(r"\\cite\{([^}]+)\}", source)
                 for key in group.split(",")}
        assert cited == set(re.findall(r"\\bibitem\{([^}]+)\}", references))
        review = PdfReader(note / "slides.pdf")
        candidate = PdfReader(note / "publication_candidate_slides.pdf")
        review_text = "\n".join(map(text, review.pages))
        candidate_text = "\n".join(map(text, candidate.pages))
        assert "Review queue before publication" in review_text
        assert "Review queue before publication" not in candidate_text
        assert "not approved for public release" in text(candidate.pages[0])
        for index in (0, 1):
            assert re.sub(r"\d+/\d+\s*$", "", text(review.pages[index])) == re.sub(
                r"\d+/\d+\s*$", "", text(candidate.pages[index]))
        cover = PdfReader(note / "cover.pdf")
        assert len(cover.pages) == 1
        size = cover.pages[0].mediabox
        assert abs(float(size.width) - 595.28) < 1
        assert abs(float(size.height) - 841.89) < 1
        packaged = PdfReader(note / "publication_candidate_with_cover.pdf")
        assert len(packaged.pages) == len(candidate.pages) + 1
        assert text(packaged.pages[0]) == text(cover.pages[0])
        assert [text(p) for p in packaged.pages[1:]] == list(map(text, candidate.pages))
        if name == "note1_mcp_reference":
            assert "Backup: comparing clock estimators" in review_text
            assert "Backup: comparing clock estimators" not in candidate_text
            assert "Backup: residual segment offsets" in candidate_text
        else:
            assert "N2-1" not in review_text and "N2-1" not in candidate_text
            assert "same nominal detector setup" in candidate_text
            assert "Cross-run response differences" in candidate_text
            for index in range(2, 8):
                assert f"N2-{index}" in candidate_text
        with zipfile.ZipFile(note / "source_bundle.zip") as bundle:
            for path in ("main.tex", "publication_candidate.tex", "references.tex",
                         "metadata.tex", "cover.tex", "slides_preamble.tex"):
                assert bundle.read(path) == (note / path).read_bytes()
        for stem in ("main", "cover", "publication_candidate"):
            log = (note / f"build/{stem}.log").read_text(errors="replace")
            assert "Overfull" not in log and "undefined" not in log.lower()
        reports[name] = dict(review_slides=len(review.pages),
                             candidate_slides=len(candidate.pages),
                             candidate_with_cover_pages=len(packaged.pages),
                             citations=sorted(cited), source_bundle_checked=True,
                             layout_overflow=0, unresolved_citations=0)
    combined = DPS / "both_notes_review_v1.pdf"
    assert len(PdfReader(combined).pages) == sum(r["review_slides"] + 1 for r in reports.values())
    report = dict(status="passed", notes=reports,
                  combined_pages=len(PdfReader(combined).pages),
                  combined_sha256=hashlib.sha256(combined.read_bytes()).hexdigest(),
                  review_status="N1-6 local work complete; convener template, DP metadata, cover handling and release approval pending",
                  note2_review_status="N2-1 closed under author's nominal-setup assumption, not independently verified; N2-2 through N2-7 remain open",
                  analysis_tests="not rerun: calibration inputs, figures and analysis code unchanged; previous checks reused")
    (OUT / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
