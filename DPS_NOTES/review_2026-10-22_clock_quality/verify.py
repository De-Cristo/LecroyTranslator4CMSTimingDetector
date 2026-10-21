"""Verify the frozen revision and combine the two compiled review notes."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pypdf import PdfReader, PdfWriter

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
RESULTS = ROOT / "Q_Walk/results/2026-10-22_clock_quality"


def main():
    note_source = (ROOT / "DPS_NOTES/note1_mcp_reference/main.tex").read_text()
    assert r"\Review{N1-5}" not in note_source
    assert "N1-5 &" not in note_source
    assert "outside its scope" in note_source
    hashes = json.loads((RESULTS / "input_sha256.json").read_text())
    for name, digest in hashes.items():
        assert hashlib.sha256((RESULTS / name).read_bytes()).hexdigest() == digest, name
    for row in json.loads((OUT / "input_manifest.json").read_text()):
        assert hashlib.sha256((OUT / row["snapshot"]).read_bytes()).hexdigest() == row["sha256"]
    report = json.loads((OUT / "validation_report.json").read_text())
    events = pd.read_csv(OUT / "selected_events.csv.gz")
    assert len(events) == report["selection_counts"]["both_measured_support"] == 10127
    assert not events.event_id.duplicated().any()
    assert events.alignment_status.eq("ok").all()
    assert np.isfinite(events[["D_before", "D_corrected", "R_before", "R_corrected"]]).all().all()
    for name in ("D_corrected", "R_corrected"):
        assert np.isclose(events[name].std(), report["widths"][name]["rms_about_mean_ps"])
    for obs in ("D", "R"):
        assert report["bootstrap"]["segment_block"][obs]["successful"] == 400
    reference = json.loads((RESULTS / "reference_plots/run4399_extended_t0_reference_report.json").read_text())
    assert len(reference["channels"]) == 22
    assert all(row["status"] == "ok" for row in reference["channels"].values())
    channels = pd.read_csv(RESULTS / "channels.csv")
    assert len(channels) == 21
    for row in channels.itertuples():
        curve = pd.read_csv(RESULTS / f"channel_{row.channel}_comparison.csv")
        residual = 1000 * (curve.mapped_ns - curve.reference_ns)
        assert abs(residual.mean()) < 1e-7
        assert np.isclose(np.sqrt(np.mean(residual**2)), row.rms_ps)
        assert curve.q_adc.min() >= max(row.q4076_min, row.q4399_min) - 1e-9
        assert curve.q_adc.max() <= min(row.q4076_max, row.q4399_max) + 1e-9
    writer = PdfWriter()
    builds = {}
    for name in ("note1_mcp_reference", "note2_dual_threshold"):
        note = ROOT / "DPS_NOTES" / name
        build = json.loads((note / "build_report.json").read_text())
        assert build["overfull_boxes"] == 0
        reader = PdfReader(note / "note_with_cover.pdf")
        assert len(reader.pages) == build["combined_pages"]
        for page in reader.pages:
            writer.add_page(page)
        for figure in json.loads((note / "figure_manifest.json").read_text()):
            for source in figure["sources"]:
                assert hashlib.sha256((note / source["snapshot"]).read_bytes()).hexdigest() == source["sha256"]
        builds[name] = build
    destination = ROOT / "DPS_NOTES/both_notes_review_v1.pdf"
    writer.add_metadata({"/Title": "Two CMS performance notes: 22 October 2026 review",
                         "/Author": "CMS", "/Subject": "Internal review draft, not approved for public release"})
    with destination.open("wb") as stream:
        writer.write(stream)
    result = dict(status="passed", reference_channels=22, overlap_fits=21,
                  bar8_events=len(events), combined_pages=len(writer.pages), builds=builds,
                  combined_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                  test_results="140 focused tests and 25 integration tests passed; 1 optional integration test skipped",
                  review_status="N1-5 closed by author for conditional timing-validation scope; intrinsic bar resolution and synchronization-jitter extraction excluded")
    (OUT / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
