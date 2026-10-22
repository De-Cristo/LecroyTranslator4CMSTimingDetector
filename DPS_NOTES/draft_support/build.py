"""Compile, package and render the two local review drafts."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import unicodedata
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw
from pypdf import PdfReader, PdfWriter

DPS = Path(__file__).resolve().parents[1]
NOTES = ("note1_mcp_reference", "note2_dual_threshold")


def run(command, cwd):
    result = subprocess.run(command, cwd=cwd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        raise RuntimeError(f"{command}\n{result.stdout[-6000:]}")
    return result.stdout


def build(note):
    out = note / "build"
    out.mkdir(exist_ok=True)
    for filename in ("slides_preamble.tex", "cover.tex", "references.tex"):
        shutil.copy2(DPS / "draft_support" / filename, note / filename)
    for stem in ("main", "cover"):
        for _ in range(2):
            run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                 "-output-directory=build", f"{stem}.tex"], note)
        log = (out / f"{stem}.log").read_text(errors="replace")
        overflow = [line for line in log.splitlines() if "Overfull" in line]
        if overflow:
            raise RuntimeError(f"{note.name}/{stem}: layout overflow:\n" + "\n".join(overflow))
        if "undefined" in log.lower():
            raise RuntimeError(f"{note.name}/{stem}: unresolved LaTeX reference or command")
    shutil.copy2(out / "main.pdf", note / "slides.pdf")
    shutil.copy2(out / "cover.pdf", note / "cover.pdf")
    writer = PdfWriter()
    for filename in ("cover.pdf", "slides.pdf"):
        for page in PdfReader(note / filename).pages:
            writer.add_page(page)
    writer.add_metadata({"/Title": note.name.replace("_", " ") + " (review draft)",
                         "/Author": "CMS", "/Subject": "Internal review draft, not approved for public release"})
    with (note / "note_with_cover.pdf").open("wb") as stream:
        writer.write(stream)
    slides = PdfReader(note / "slides.pdf")
    assert "cms-dpg-conveners-mtd@cern.ch" in slides.pages[0].extract_text()
    assert "Abstract" in slides.pages[1].extract_text()
    assert len(PdfReader(note / "cover.pdf").pages) == 1
    figures = json.loads((note / "figure_manifest.json").read_text())
    for figure in figures:
        text = PdfReader(note / figure["figure"]).pages[0].extract_text()
        assert "Preliminary Phase 2" in text, figure["figure"]
    preview = note / "preview"
    preview.mkdir(exist_ok=True)
    run(["pdftoppm", "-scale-to", "1100", "-png", str(note / "slides.pdf"),
         str(preview / "slide")], note)
    pages = sorted(preview.glob("slide-*.png"))[:len(slides.pages)]
    sheets = []
    for start in range(0, len(pages), 6):
        sheet = Image.new("RGB", (1100, 1035), "#dddddd")
        draw = ImageDraw.Draw(sheet)
        for i, page in enumerate(pages[start:start+6]):
            img = Image.open(page).convert("RGB")
            img.thumbnail((530, 305))
            x, y = 10 + (i % 2)*550, 28 + (i // 2)*345
            sheet.paste(img, (x, y))
            draw.text((x, y-19), f"Slide {start+i+1}", fill="black")
        destination = preview / f"contact_sheet_{start//6+1}.png"
        sheet.save(destination)
        sheets.append(str(destination.relative_to(note)))
    with zipfile.ZipFile(note / "source_bundle.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(note.rglob("*")):
            rel = path.relative_to(note)
            if not path.is_file() or rel.parts[0] in ("build", "preview"):
                continue
            if path.name in ("source_bundle.zip", "slides.pdf", "cover.pdf", "note_with_cover.pdf", "build_report.json",
                             "publication_candidate_slides.pdf", "publication_candidate_with_cover.pdf", "candidate_report.json"):
                continue
            if rel.parts[0] == "figures" and path.suffix == ".png":
                continue
            archive.write(path, rel)
    report = dict(slides=len(slides.pages), cover_pages=1,
                  combined_pages=len(PdfReader(note / "note_with_cover.pdf").pages),
                  labelled_figures=len(figures), overfull_boxes=0,
                  preview_sheets=sheets, status="internal review draft")
    (note / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(note.name, json.dumps(report), flush=True)


def build_candidate(note):
    """Build a separately numbered candidate without internal-only backup pages."""
    for _ in range(2):
        run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
             "-output-directory=build", "publication_candidate.tex"], note)
    log = (note / "build/publication_candidate.log").read_text(errors="replace")
    assert "Overfull" not in log and "undefined" not in log.lower(), note.name
    slides_path = note / "publication_candidate_slides.pdf"
    shutil.copy2(note / "build/publication_candidate.pdf", slides_path)
    reader = PdfReader(slides_path)
    texts = [unicodedata.normalize("NFKC", page.extract_text()) for page in reader.pages]
    assert "cms-dpg-conveners-mtd@cern.ch" in texts[0]
    assert "Abstract" in texts[1]
    assert any("References" in text and "Cleveland" in text for text in texts)
    assert not any("Review queue before publication" in text for text in texts)
    if note.name == "note1_mcp_reference":
        forbidden = ("Backup: comparing clock estimators", "Backup: clock-phase reconstruction",
                     "Backup: historical end-difference", "Backup: historical bar")
        assert not any(title in text for title in forbidden for text in texts)
        assert any("Backup: residual segment offsets" in text for text in texts)
    # An optional local cover preview is separate from the slides-only deliverable.
    writer = PdfWriter()
    for path in (note / "cover.pdf", slides_path):
        for page in PdfReader(path).pages:
            writer.add_page(page)
    writer.add_metadata({"/Author": "CMS", "/Title": note.name + " (unapproved publication candidate)",
                         "/Subject": "Template, DP number and CMS approval pending; not for submission"})
    cover_path = note / "publication_candidate_with_cover.pdf"
    with cover_path.open("wb") as stream:
        writer.write(stream)
    report = dict(status="unapproved candidate; convener confirmation pending",
                  slides=len(reader.pages), with_cover_pages=len(writer.pages),
                  overfull_boxes=0, unresolved_citations=0, dp_number=None,
                  template_officially_confirmed=False, cover_handling_confirmed=False,
                  scientific_review_comments_retained=(note.name == "note2_dual_threshold"
                      and r"\Review{" in (note / "main.tex").read_text()),
                  files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (slides_path, cover_path)})
    (note / "candidate_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(note.name, "candidate:", json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", action="store_true", help="Redraw figures before compiling")
    parser.add_argument("--candidates", action="store_true", help="Also build unapproved publication-candidate variants")
    args = parser.parse_args()
    if args.assets:
        run([sys.executable, str(DPS / "draft_support/build_assets.py")], DPS)
    for name in NOTES:
        build(DPS / name)
        if args.candidates:
            build_candidate(DPS / name)
