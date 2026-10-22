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

ROOT = Path(__file__).resolve().parent
NOTES = ("note1_mcp_reference", "note2_dual_threshold")
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
            target = note / relative
            shutil.copy2(origin, target)
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            assert digest == hashlib.sha256(origin.read_bytes()).hexdigest()
            manifest.append(dict(file=relative, source=os.path.relpath(origin, ROOT.parent), sha256=digest))
            if relative.startswith("figures/"):
                assert "Preliminary Phase 2" in PdfReader(target).pages[0].extract_text()
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
        assert len(pdf.pages) == 11, (name, len(pdf.pages))
        assert "References" in pdf.pages[-1].extract_text()
        assert sum("References" in page.extract_text() for page in pdf.pages) == 1
        entry_pattern = r"\\bibitem\[(\d+)\]\{([^}]+)\}([\s\S]*?)(?=\\bibitem|\\end\{thebibliography\})"
        entries = lambda path: [(number, key, " ".join(body.split()))
                                for number, key, body in re.findall(entry_pattern, path.read_text())]
        assert entries(note / "references.tex") == entries(original / "references.tex")
        assert "cms-dpg-conveners-mtd@cern.ch" in pdf.pages[0].extract_text()
        assert ("Purpose" if name == "note1_mcp_reference" else "Abstract") in pdf.pages[1].extract_text()
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
        report[name] = dict(slides=len(pdf.pages), content_slides=10, reference_slides=1,
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
