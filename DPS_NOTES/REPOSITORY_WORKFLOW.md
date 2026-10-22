# Note review and repository synchronization

## Standing authorization

The author requests that note contents stay synchronized with the online
repository and be committed/pushed after each accepted review point.

- Working analysis directory: `sandbox_test` beside the repository.
- Repository checkout:
  `/afs/cern.ch/work/l/lichengz/private/MTD/TB_2025/LecroyTranslator4CMSTimingDetector`
- Remote: `git@github.com:De-Cristo/LecroyTranslator4CMSTimingDetector.git`
- Branch at setup: `master`, tracking `origin/master`.

This authorization covers the note review workflow and necessary synchronization
of cited analysis code. It does not authorize unrelated commits, force pushes,
or an official CMS submission.

## After each accepted review point

1. Edit the affected LaTeX and update the review status in the READMEs.
2. Rebuild the affected note, inspect changed pages, and regenerate
   `both_notes_review_v1.pdf` from the two current `note_with_cover.pdf` files.
3. Inspect the destination checkout and fetch its remote. Preserve unrelated
   staged, tracked and untracked work. Reconcile divergence without force.
4. Copy the note sources, figure PDFs, source-data snapshots, manifests,
   documentation and current review PDFs to the repository's `DPS_NOTES/`.
   Exclude build intermediates, rendered previews, caches, redundant ZIP bundles,
   generated figure PNGs, and the original supplied Note1/Note2 PDFs. The supplied
   template ZIP also stays local; its extracted sources are included.
5. Compare all scripts cited by the affected slides against the working analysis
   tree. Synchronize relevant differences after inspection, without unrelated
   analysis refactoring. Current N1-2 references:
   `FEBD_Sync/simple_sync.py`,
   `TimeCalibration/ch192_vs_trigger_lowess.py`,
   `TimeCalibration/ch192_vs_trigger.py`,
   `TimeCalibration/bar_processing.py`.
6. Verify copied files by SHA-256 and record the result in `sync_manifest.json`
   in the repository. Stage explicit note/cited-code paths only. Inspect the
   staged summary, then make a descriptive review-point commit and push it.
7. Verify the remote branch points to the pushed commit and report its hash.
   A failed push remains unfinished synchronization; report the failure.

Do not overwrite changed destination files without first understanding those
changes. Do not delete unrelated files while synchronizing.

## Reproducibility scope

The checked-in PDFs, figure PDFs, LaTeX and numerical snapshots support review
and recompilation of the decks. Running `draft_support/build.py` without
`--assets` recompiles from the included figures. The `--assets` route currently
uses analysis-tree paths from the working `sandbox_test` environment; the full
ROOT datasets and complete Q_Walk analysis tree are not part of this note upload.
Do not describe this synchronization as reproducing the original event analysis.

The 22 October clock-quality update also includes the changed Q_Walk ingestion
and reference runner, their local module dependencies and focused tests, plus
the refreshed numerical reference/closure artifacts. Extraction caches and
ROOT inputs remain local. Full event regeneration still requires the documented
EOS inputs and analysis environment. Numerical snapshots permit verification of
the displayed results without those ROOT files.

The N2-2 closure also includes the run4076 pair/profile helpers, focused tests
and `review_2026-10-22_n2_2/` audit. Its ROOT replay reproduces the frozen
selection without inheriting the newer 100 ps minimum-rise default. No
production analysis code or frozen numerical input was changed for that closure;
the added helpers are copies of the existing local implementations. Full replay
still requires the documented run4076 `23_e.root` file on EOS.

The current notes remain marked as review drafts. Publishing them in the
repository does not remove any unresolved scientific review item or imply CMS
approval.
