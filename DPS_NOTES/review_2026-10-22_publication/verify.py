"""Check the N1-6 documentation-only candidate packaging."""
import csv
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
        preamble = (note / "slides_preamble.tex").read_text()
        assert preamble == (DPS / "draft_support/slides_preamble.tex").read_text()
        repo = re.search(r"\\newcommand\{\\AnalysisRepo\}\{([^}]+)\}", preamble).group(1)
        revision = re.search(r"\\newcommand\{\\AnalysisRevision\}\{([0-9a-f]{40})\}", preamble).group(1)
        expected_links = set()
        assert r"\RepoDir" not in source
        for path in re.findall(r"\\RepoFile\{([^}]+)\}", source):
            path = path.replace(r"\#", "#")
            expected_links.add(f"{repo}/blob/{revision}/{path}")
        for deck, content in ((review, review_text), (candidate, candidate_text)):
            urls = {str(a.get_object().get("/A", {}).get("/URI", ""))
                    for page in deck.pages for a in page.get("/Annots", [])}
            assert {u for u in urls if u.startswith(repo)} == expected_links
            assert f"fixed code revision {revision[:7]}" in " ".join(content.split())
            for local_path in ("FEBD_Sync/", "TimeCalibration/", "Q_Walk/", "DPS_NOTES/",
                               "/afs/", "repository-relative", "CMS-Note/cmsnote",
                               "2026-10-22_clock_quality"):
                assert local_path not in content
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
            for content in (review_text, candidate_text):
                assert "Timing validation using the two bar ends" not in content
                assert "Timing validation against the MCP reference" in content
                assert "Same core-fit definition as the end difference" not in content
            assert "Backup: comparing clock estimators" in review_text
            assert "Backup: comparing clock estimators" not in candidate_text
            assert "Backup: residual segment offsets" in candidate_text
        else:
            assert "N2-1" not in review_text and "N2-1" not in candidate_text
            assert "same nominal detector setup" in candidate_text
            assert "Cross-run response differences" in candidate_text
            assert "N2-2" not in review_text and "N2-2" not in candidate_text
            assert "Timing reconstruction and threshold scale" in candidate_text
            assert "Landau" in candidate_text and "does not set this cut" in candidate_text
            for index in (3, 4):
                assert f"N2-{index}" not in candidate_text and f"N2-{index}" not in review_text
            assert not json.loads((note / "candidate_report.json").read_text())["scientific_review_comments_retained"]
            assert "N2-5" not in review_text and "N2-5" not in candidate_text
            assert "N2-6" not in review_text and "N2-6" not in candidate_text
            assert "N2-7" not in review_text and "N2-7" not in candidate_text
            for deck in (review_text, candidate_text):
                normalized = " ".join(deck.split())
                assert "K is learned from MCP-calibrated training channels" in normalized
                assert "does not demonstrate reference-free calibration" in normalized
                assert "independent validation across operating conditions" in normalized
                assert "prediction uncertainties unpropagated" in normalized
                assert deck.index("Result and next validation step") < deck.index(
                    "Backup: an internal predictor for channel AWC") < deck.index(
                    "Backup: predict a bar excluded from the model fit")
            assert "Channel coverage and limitations" in candidate_text
            assert "AWC versus inverse slope coefficient: all completed fits" in candidate_text
            assert "A selected subset does not establish a universal law" in candidate_text
            assert "illustrative, not quality-selected" in " ".join(candidate_text.split())
            snapshot = note / "data/sandbox_test/Q_Walk/results/2026-10-22_clock_quality"
            with (snapshot / "channels.csv").open() as stream:
                channels = {int(r["channel"]): r for r in csv.DictReader(stream)}
            expected_up = {130, 133, 134, 135, 136, 137, 139, 149, 150, 151, 154, 155, 156, 159}
            expected_down = {235, 236, 239, 240, 242, 244, 245}
            assert set(channels) == expected_up | expected_down
            assert json.loads((DPS / "review_2026-10-22_n2_6/failures.json").read_text()) == [
                {"channel": 238, "error": "No measured profile overlap"}]
            for ch, width in ((235, "3.10"), (244, "16.99")):
                row = channels[ch]
                assert f'{float(row["overlap_max"]) - float(row["overlap_min"]):.2f}' == width
                assert width in candidate_text
                assert int(row["profile4076_points_in_overlap"]) == 1
                assert int(row["profile4399_points_in_overlap"]) == 1
            for ch in (130, 236, 239):
                row = channels[ch]
                assert float(row["awc_p16"]) < 0 < float(row["awc_p84"])
                assert float(row["awc_p84"]) - float(row["awc_p16"]) > 100
            assert all(float(channels[ch]["awc_ua"]) < 0 for ch in (136, 245))
            with (snapshot / "awc_summary_special_10_channels_no_ch240_linear.csv").open() as stream:
                selected = list(csv.DictReader(stream))
            assert {int(r["channel"]) for r in selected} == {
                133, 134, 135, 137, 150, 151, 154, 155, 156, 159}
            assert len(selected) == 10
            for ch in (137, 154, 159, 242):
                figure = PdfReader(note / f"figures/comparison_ch{ch}.pdf")
                assert "Model difference" in " ".join(text(figure.pages[0]).split())
            assert "RMS shape mismatch" in " ".join(candidate_text.split())
            assert "not corrected-event timing resolution" in " ".join(candidate_text.split())
            assert "not an event-time distribution" in " ".join(candidate_text.split())
            assert "not a timing-resolution measurement" in " ".join(candidate_text.split())
            for removed in ("Residual check: is a straight line sufficient?",
                            "Model and selection choices change the inferred AWC"):
                assert removed not in candidate_text and removed not in review_text
            for deck in (review_text, candidate_text):
                normalized = " ".join(deck.split())
                assert "Linearity is a working approximation, not a validated precision model" in normalized
                assert "Model and selection dependence are excluded from the quoted errors" in normalized
                assert "slope-only statistical" in normalized
                assert "reported separately, not total calibration uncertainties" in normalized
            assert "The current review items are closed for this exploratory version" in " ".join(review_text.split())
            assert "Channel 137: inverse-slope and MCP walk shapes" in candidate_text
            assert "MCP-fit uncertainty and charge weighting" in candidate_text
            for entry in json.loads((note / "data/n2_4_manifest.json").read_text()):
                assert hashlib.sha256((note / entry["snapshot"]).read_bytes()).hexdigest() == entry["sha256"]
            assert (note / "data/n2_4_uncertainty_table.tex").read_bytes() == (
                DPS / "review_2026-10-22_n2_4/uncertainty_table.tex").read_bytes()
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
                             github_revision=revision, github_links_checked=len(expected_links),
                             layout_overflow=0, unresolved_citations=0)
    combined = DPS / "both_notes_review_v1.pdf"
    assert len(PdfReader(combined).pages) == sum(r["review_slides"] + 1 for r in reports.values())
    report = dict(status="passed", notes=reports,
                  combined_pages=len(PdfReader(combined).pages),
                  combined_sha256=hashlib.sha256(combined.read_bytes()).hexdigest(),
                  review_status="N1-6 local work complete; convener template, DP metadata, cover handling and release approval pending",
                  note2_review_status="All current Note2 review items closed within documented scope. N2-3/N2-4 temporarily closed by author for exploratory linear model and conditional statistics with separate weighting sensitivity; permanent limitations retained, not precision-model validation or a total error budget. Next pass is page by page; numerical results unchanged.",
                  analysis_tests="previous clock-quality tests reused; N2-2 ROOT replay and 17 focused tests documented in review_2026-10-22_n2_2/README.md; frozen numerical results unchanged")
    (OUT / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
