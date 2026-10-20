CMS DP note: Beamer slides plus A4 cover

OVERLEAF
1. Upload cms_dp_note_template.zip as a new project.
2. Use pdfLaTeX and select main.tex as the main document for the slides.
3. Edit metadata.tex: title, DP number, date, abstract, contact email.
4. Replace the plot placeholder in main.tex with your plot and caption.
5. To generate the A4 cover, change the main document to cover.tex and recompile.
   Both documents read the same title and abstract from metadata.tex.

SLIDES
The plain 16:9 deck has three slides:
1. Title; author CMS; clickable cms-dpg-conveners-mtd@cern.ch contact.
2. Abstract, identical to the abstract on the cover.
3. Reusable plot-and-caption layout. Duplicate this frame for each result.
The CMS logo appears in a reserved header on every slide.
Avoid [plain] frames, which suppress the header/logo and footer.
Change aspectratio=169 to aspectratio=43 for a 4:3 deck if needed and recheck layout.

PLOTS
Upload a figure into figures/, then replace \DPPlotPlaceholder with
\DPPlot{figures/your_plot.pdf}.
This adds 'CMS Preliminary Phase 2' immediately above that individual plot.
Use the wrapper separately for each plot if adding multiple panels.
The optional height can be changed, e.g. \DPPlot[40mm]{figures/your_plot.pdf}.
For a plot already containing that exact label, use
\DPPlotEmbedded{figures/your_plot.pdf} to avoid duplicating it.
The Beamer wrapper labels the displayed panel, not the source image itself.
If distributing plots as separate files, include the same label in each
plot's source/export so it stays present outside the slide deck.

CAPTIONS
Replace the entire caption placeholder with a detailed, self-contained paragraph.
Explain: what is measured and why; sample and operating conditions; selection;
axes and units; the meaning of markers, lines and colours; uncertainties;
and the main result and its significance. Define abbreviations and explain
specialist terms for a general audience. No numerical results are supplied
by this template. Remove all guidance and placeholders before publication.

COVER AND PUBLICATION
cover.tex is the A4 cover adapted from the previously downloaded CR template.
main.tex is the slide content; its first page is the title slide.
They compile independently, without shell escape or a multi-stage build.
Follow the publication workflow when combining the cover and slides; do not
add a second cover if the publication system generates one automatically.
This is a working adaptation, not a verified official CMS DP template.
