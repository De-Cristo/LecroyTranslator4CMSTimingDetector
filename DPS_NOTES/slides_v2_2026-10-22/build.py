"""Build the two short slide decks from existing validated figures; no refitting."""
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw
from pypdf import PdfReader, PdfWriter
from prepare_note2_backups import NAMES as BACKUP_FIGURES, prepare as prepare_backups
from prepare_note1_labels import NAMES as NOTE1_LABELS, prepare as prepare_note1_labels
from prepare_note1_matching_ratio import prepare as prepare_note1_matching
from prepare_note1_drift import prepare as prepare_note1_drift

ROOT = Path(__file__).resolve().parent
NOTES = ("note1_mcp_reference", "note2_dual_threshold")
EXPECTED_SLIDES = {"note1_mcp_reference": 10, "note2_dual_threshold": 10}
THRESHOLD_FIGURES = {
    "figures/threshold_walk_profiles.pdf": "vth1_3_vth2_60/upstream_walk_profiles.pdf",
    "figures/threshold_histograms.pdf": "before_after_all_thresholds.pdf",
    "figures/threshold_width_summary.pdf": "width_vs_threshold.pdf",
}


def run(command, cwd):
    result = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(result.stdout[-6000:])


def main():
    prepare_backups()
    prepare_note1_labels()
    prepare_note1_matching()
    prepare_note1_drift()
    setup = ROOT / "assets/note1_setup"
    (setup / "build").mkdir(exist_ok=True)
    run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
         "-output-directory=build", "setup.tex"], setup)
    setup_log = (setup / "build/setup.log").read_text(errors="replace")
    assert not any(issue in setup_log for issue in ("Overfull", "Missing character", "Font Warning"))
    shutil.copy2(setup / "build/setup.pdf", setup / "setup.pdf")
    combined = PdfWriter()
    report = {}
    for name in NOTES:
        note = ROOT / name
        original = ROOT.parent / name
        source = (note / "main.tex").read_text()
        figures = set(re.findall(r"\\PlotSlide\{[^}]+\}\{([^}]+)\}", source))
        figures.update(re.findall(r"\{figures/([^}]+)\.pdf\}", source))
        (note / "figures").mkdir(exist_ok=True)
        assets = ["CMSlogo_black_label_1024.png", "review_values.tex"]
        assets += [f"figures/{f}.pdf" for f in sorted(figures)]
        manifest = []
        for relative in assets:
            origin = original / relative
            if name == "note1_mcp_reference" and relative in THRESHOLD_FIGURES:
                origin = (ROOT.parent.parent / "Q_Walk/results/2026-10-22_bar_pair_by_threshold"
                          / THRESHOLD_FIGURES[relative])
            if name == "note2_dual_threshold" and relative == "figures/original_figure5.pdf":
                origin = ROOT / "assets/note2_figure5/figure5.pdf"
            if name == "note1_mcp_reference" and Path(relative).stem in NOTE1_LABELS:
                origin = ROOT / "assets/note1_public_labels" / Path(relative).name
            if name == "note1_mcp_reference" and relative == "figures/setup.pdf":
                origin = setup / "setup.pdf"
            if name == "note1_mcp_reference" and relative == "figures/event_matching.pdf":
                origin = ROOT / "assets/note1_matching_ratio/event_matching.pdf"
            if name == "note1_mcp_reference" and relative == "figures/clock_residuals.pdf":
                origin = ROOT / "assets/note1_drift/clock_residuals.pdf"
            is_backup = name == "note2_dual_threshold" and Path(relative).stem in BACKUP_FIGURES
            if is_backup:
                origin = ROOT / "assets/note2_internal_backups" / Path(relative).name
            target = note / relative
            shutil.copy2(origin, target)
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            assert digest == hashlib.sha256(origin.read_bytes()).hexdigest()
            manifest.append(dict(file=relative, source=os.path.relpath(origin, ROOT.parent), sha256=digest))
            if relative.startswith("figures/"):
                figure_text = PdfReader(target).pages[0].extract_text()
                assert ("Preliminary Phase 2" in figure_text) != is_backup
        (note / "asset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (note / "build").mkdir(exist_ok=True)
        for _ in range(2):
            run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                 "-output-directory=build", "main.tex"], note)
        log = (note / "build/main.log").read_text(errors="replace")
        problems = [line for line in log.splitlines()
                    if any(word in line for word in ("Overfull", "Missing character", "Font Warning"))
                    or "undefined" in line.lower()]
        assert not problems, (name, problems)
        shutil.copy2(note / "build/main.pdf", note / "slides.pdf")
        pdf = PdfReader(note / "slides.pdf")
        assert len(pdf.pages) == EXPECTED_SLIDES[name], (name, len(pdf.pages))
        reference_index = 9 if name == "note1_mcp_reference" else 8
        assert "References" in pdf.pages[reference_index].extract_text()
        for page in pdf.pages[reference_index + 1:]:
            text = page.extract_text()
            assert "Backup:" in text and "Internal discussion only" in text
            assert "Preliminary" not in text
        assert sum("References" in page.extract_text() for page in pdf.pages) == 1
        entry_pattern = r"\\bibitem\[(\d+)\]\{([^}]+)\}([\s\S]*?)(?=\\bibitem|\\end\{thebibliography\})"
        entries = lambda path: [(number, key, " ".join(body.split()))
                                for number, key, body in re.findall(entry_pattern, path.read_text())]
        assert entries(note / "references.tex") == entries(original / "references.tex")
        assert "cms-dpg-conveners-mtd@cern.ch" in pdf.pages[0].extract_text()
        assert ("Abstract" if name == "note1_mcp_reference" else "Purpose") in pdf.pages[1].extract_text()
        for page in pdf.pages:
            assert float(page.mediabox.width) > float(page.mediabox.height)
            combined.add_page(page)
        preview = note / "preview"
        preview.mkdir(exist_ok=True)
        run(["pdftoppm", "-scale-to", "1100", "-png", "slides.pdf", "preview/slide"], note)
        pages = sorted(preview.glob("slide-*.png"))[:len(pdf.pages)]
        for start in range(0, len(pages), 6):
            sheet = Image.new("RGB", (1100, 1035), "#dddddd")
            draw = ImageDraw.Draw(sheet)
            for i, path in enumerate(pages[start:start + 6]):
                with Image.open(path) as img:
                    img.thumbnail((530, 305))
                    x, y = 10 + (i % 2) * 550, 28 + (i // 2) * 345
                    sheet.paste(img, (x, y))
                    draw.text((x, y - 19), f"Slide {start+i+1}", fill="black")
            sheet.save(preview / f"contact_sheet_{start//6+1}.png")
        report[name] = dict(slides=len(pdf.pages), content_slides=len(pdf.pages)-1, reference_slides=1,
                            main_slides=reference_index, backup_slides=len(pdf.pages)-reference_index-1,
                            reference_entries=len(entries(note / "references.tex")),
                            cover_pages=0, figures=len(figures),
                            overflow=0, unresolved_citations=0,
                            sha256=hashlib.sha256((note / "slides.pdf").read_bytes()).hexdigest())
        print(name, json.dumps(report[name]), flush=True)
    combined.add_metadata({"/Title": "BTL walk calibration: short slide drafts, version 2",
                           "/Author": "CMS", "/Subject": "Internal review; not approved for public release"})
    with (ROOT / "both_notes_slides_v2.pdf").open("wb") as stream:
        combined.write(stream)
    report["combined_pages"] = len(combined.pages)
    report["status"] = "compiled; internal review draft"
    (ROOT / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
