"""Verify the saved N2-2 audit and export a compact per-channel table."""
import hashlib
import json
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def main():
    report = json.loads((OUT / "audit.json").read_text())
    assert report["status"] == "passed"
    assert len(report["channels"]) == 22
    for group in ("input_hashes", "code_hashes"):
        for name, digest in report[group].items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    for row in report["channels"]:
        assert all(row["checks"].values())
        assert row["selected_missing_fine_calibration"] == 0
        assert row["prompt_missing_fine_calibration"] == 0
        assert row["threshold_settings"] == [[113621.0, 10, 35, 7.8125]]
        assert row["fitted_profile_bins"] == (9 if row["channel"] == 135 else 10)
    note = ROOT / "DPS_NOTES/note2_dual_threshold"
    source = (note / "main.tex").read_text()
    assert r"\Review{N2-2}" not in source and "N2-2 &" not in source
    assert "The Landau--Gaussian fit does not set this cut" in source
    for stem in ("main", "publication_candidate"):
        log = (note / f"build/{stem}.log").read_text(errors="replace")
        assert "Overfull" not in log and "undefined" not in log.lower()
    fields = ("channel", "module", "pair_hits", "prompt_hits", "selected_hits",
              "q_cut_adc", "selected_dt_min_ps", "selected_dt_max_ps",
              "selected_missing_fine_calibration", "selected_below_100ps",
              "populated_profile_bins", "fitted_profile_bins")
    pd.DataFrame([{k: r[k] for k in fields} for r in report["channels"]]).to_csv(
        OUT / "channel_counts.csv", index=False)
    result = dict(status="passed", replayed_channels=22,
                  selected_missing_calibration=0,
                  review_status="N2-2 closed; exact frozen definitions reproduced; selection/model sensitivity remains N2-3/N2-6",
                  note2_slides=len(PdfReader(note / "slides.pdf").pages),
                  note2_candidate_slides=len(PdfReader(note / "publication_candidate_slides.pdf").pages),
                  combined_pages=len(PdfReader(ROOT / "DPS_NOTES/both_notes_review_v1.pdf").pages),
                  audit_sha256=hashlib.sha256((OUT / "audit.json").read_bytes()).hexdigest())
    (OUT / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
