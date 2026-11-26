# Editable figure PDF exports

## November 19: full-test Siamese evaluation

S20/S21 now use the extended cohort with the original Siamese checkpoint and unchanged preprocessing. The 96 original scores are preserved exactly; the full S21 comparison has 65/132 positive differences. S20 uses 65 two-replicate configurations. See [current folders, results and validation](<sim_to_exp_diffusion/controlnet_essential/Markdown%20files/siamese_full_test_132.md>). This supersedes older 96-image source notes for the active S20/S21 PDFs.

## November 19, 2026: full-test image metrics

Figure 4d and Supplementary Figure 18a now use **132 held-out images (96 primary + 36 fixed-seed), from 67 configurations**. Both image-metric notebooks have been updated and executed. LPIPS, SSIM, ORB and CMMD are calculated from the matched full cohort; S18a uses the corrected seed-conditioned generator for both cohorts. This supersedes earlier 96-image/manual-summary provenance for these panels. Previous notebooks and exports are retained in the result folder.

See [analysis details and current results](<sim_to_exp_diffusion/controlnet_essential/Markdown%20files/full_test_image_metrics_132.md>), including the changed seed-conditioned comparison and the separate 96-image area-regression panel. Active PDFs and their manifest/validation entries have been refreshed.

Figure numbering follows the current November 5, 2026 main manuscript and SI in
`Manuscript/`, rather than older figure numbers in notebook filenames.

## Using the notebook cells

New code cells tagged `vector-pdf-export` have been inserted immediately after
the relevant original plots. Each export cell initializes its own PDF helper;
it can be run independently once its preceding source cell has populated the
required figure/data. Use the existing `pytorch_PA_patternprediction` environment
and the notebook's normal working directory inside this project.

For simple plots, the new cell saves the existing Matplotlib Figure, including
Figures closed by `plt.show()` or `plt.close(fig)`. Where an original plotting
helper creates local Figures without returning them, the new cell repeats only
the plotting using already-computed tensors, scores, or summary values. It does
not run training, inference, or re-estimate analysis metrics.

Every original notebook cell, output, execution count and metadata was preserved,
as was its original JSON formatting. No model, training, preprocessing, analysis,
checkpoint selection, or existing saving code was changed.

## Output locations and sources

Each part has a folder with the exact name `vector based plots`:

| Folder                                                  | Requested figures                                                                                             |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `seed_to_sim1_sim2_deterministic/vector based plots/` | Fig. 1C (simulation and experimental reconstruction grids), Fig. 2C, Fig. 3C/E, S2A/B and combined S2, S25A/B |
| `sim_to_exp_diffusion/vector based plots/`            | Fig. 4C/D, S15B/C/D/E, S16C, S18A/B/C, S20, S21, S25C                                                         |
| `information_encoding_decoding/vector based plots/`   | Fig. 5F/G, S18D, S24A/B/C/D, S25D and combined S25                                                            |

Each folder's `README.md` maps every output filename to its source notebook.
`vector_export_manifest.json` also records the zero-based **original** source
cell index before export cells were inserted. There are 44 planned PDFs because
Fig. 1C has two grids, Fig. 5F includes its combined plot and separate metric
panels, and S25 includes both its combined plot and individual panels.

In particular:

- Revised Fig. 2C is the SSIM-versus-seed-count scatter plot in
  `Scatterplot_SSIM_seedvariation_SuppFig7.ipynb`. Its export cell replots the
  existing score arrays because the source cell does not retain its Figure.
- Revised Fig. 3C/E are the validation-loss curves in the two notebooks still
  named `Prediction_datademand_*_Fig4.ipynb`.
- Revised Fig. 4C is in `morphometric_variation_analysis.ipynb`; revised Fig. 4D
  and S18A share the two-group plotting cell in
  `inference_quantmetrics_simtoexp.ipynb`.
- S15B uses the final full-test-set Fourier cell, not the earlier exploratory
  spectra/calibration plots.
- S16C uses the final aggregate replicate-variability ratio cell in
  `morphometric_variation_analysis.ipynb`.
- S20 and S21 come from `DissimilarityScore_Seeding.ipynb`, rather than the older
  `Display_Fig6_SuppFig20.ipynb` notebook with outdated numbering.
- S25 is exported from the root `Main_model_loss_curves.ipynb`. Its individual
  panels are routed to their corresponding parts; standalone panels include
  the original shared legend.

## PDF editability

`vector_plot_export.py` saves native Matplotlib PDF artists with embedded
TrueType fonts (`pdf.fonttype = 42`). Artist and axes rasterization are disabled
for saving and restored afterward, including automatically rasterized colorbar
meshes. Standalone panel exports temporarily hide adjacent axes and shared
figure decorations to prevent neighboring content appearing in the crop.
Fig. 3E explicitly applies the configured Arial font to all text and its
logarithmic tick labels before export, while retaining the existing text sizes.

Open a PDF in Inkscape using text import that preserves text, then ungroup its
objects as needed. Arial must be installed on the editing computer to retain
its appearance when text is edited. Saving from Inkscape as PDF retains vector
plot objects; avoid exporting a PNG and placing that PNG in a PDF.

**Image limitation:** Fig. 1C's source images, VAE latents, and reconstructions
are pixel data. They remain embedded images in the PDF and can be moved/scaled
individually; their internal pixels are not vector paths. Converting them to
paths would require tracing and would change their scientific representation.

## Source issues needing manuscript review

These were preserved rather than changed as part of the export work:

1. **S15C uncertainty wording:** its caption calls the black diamond error bars
   standard deviations, but `morphometric_variation_analysis.ipynb` computes
   plate-bootstrap confidence intervals with `CONFIDENCE_LEVEL = 0.95`.
2. **S20 prediction order — corrected:** the notebook and vector PDF now place
   grayscale predictions left and binary predictions right, with matching tick
   settings. The underlying cached distances were retained. Replace the older
   manuscript image with the corrected PDF.
3. **Fig. 3E point omission:** the augmented-data plotting cell explicitly
   deletes the fourth sorted data point before plotting. The export retains
   that omission; confirm it is intentional.
4. **Fixed summary inputs:** Fig. 4D/S18A now calculate all metrics on the full 132-image cohort
   (see the November 19 update above). Fig. 5F still uses a fixed summary table.
   Exporting preserves these values rather than refreshing them from later
   evaluation runs. Confirm they are the intended final results.
5. **S24 uncertainty documentation:** the configuration-diversity/mixture
   notebook's introduction still describes 95% normal confidence intervals;
   its current plotting functions use `tolerance_f1_std` directly (SD).

Validation and the generated-file inventory are recorded in
`vector_export_validation.json`. Some PDFs were generated from existing loss
logs or saved metric summaries during verification. For entries not yet
written, run their new export cells in the corresponding populated notebook;
the earlier export checks did not repeat generative-model inference.

Fig. 4C/D have also been generated by running their existing notebook analysis
and export cells. The KDE peaks reproduce 5.07 mm (simulation), 4.02 mm
(experiment), and 3.81 mm (diffusion). Fig. 4D recomputes LPIPS, SSIM, and ORB
on the existing 96-image cohort and retains the notebook's specified CMMD
values. No new diffusion samples were generated. Both PDFs embed only Arial
and contain no rasterized plot artists. Measurement CSVs and
`Fig4C_export_run.json` / `Fig4D_export_run.json` are saved alongside the PDFs
so later formatting changes can reuse the exact numerical results.

## Supplementary Fig. 2 hybrid image/text PDFs

`SSIM_ExpandSim_SuppFig1.ipynb` now has two new `hybrid-pdf-export` cells for the
current S2A/B simulation/experimental reconstruction grids. The main image/text
layout, nine selected examples, SSIM calculation and Arial typeface follow the
existing display helper. Only image artists are rasterized, at 300 dpi in the
original figure dimensions; the SSIM numbers remain separate vector text.
There are two individual PDFs and a two-page combined PDF in the deterministic
part's `vector based plots` folder. All three passed checks of actual embedded
image DPI, page counts, and TrueType Arial text objects.

This export reran the original VAE reconstruction cells. The encoder samples its
posterior, so the reconstructed pixels and SSIM scores can differ slightly from
the previously displayed notebook figure. Original cells/outputs were preserved;
this run's exact arrays, filenames and scores are cached under `ssim source data`
for future re-exports without inference. The dedicated hybrid validation JSON
records the comparison to the previous displayed scores.

All 18 displayed SSIM values in this export match the previous notebook rendering
to three decimal places.

## Supplementary grids and S20–21 (November 13)

S3–5, S8–9 and S13–14 now have completed 300 dpi PDFs containing 100 separate
image objects each. See the deterministic part's
`vector based plots/IMAGE_GRID_EXPORTS.md` for filenames and source mapping.
All seven original grids were recreated by the same display helper; no original
notebook cells or outputs were changed. Their exact display arrays are cached.

S20–21 vector PDFs are generated in the simulation-to-experiment part. The
manuscript's S20 image exactly matched the notebook output before the prediction-order correction. S21
has the same 20 histogram counts and differs only by image resizing in the
manuscript. Both vector PDFs embed Arial and contain no raster image objects.
See `sim_to_exp_diffusion/vector based plots/SuppFig20_21_SOURCE_AUDIT.md` for
quantitative evidence and the preserved prediction-title / distance-terminology
issues needing manuscript review.

S20’s prediction order was subsequently corrected: grayscale now appears left
and binary right, with their respective axis ticks. The existing cached scores
were reused. S21’s metric is unchanged: `F.pairwise_distance` with default p=2
computes Euclidean distance between unnormalized 64-dimensional Siamese
embeddings. Cosine dissimilarity would instead require
`1 - F.cosine_similarity(...)` and a fresh metric evaluation.

## Supplementary Fig. 7 A/B

Both panels have been generated in the deterministic `vector based plots`
folder. A has 20 raster image objects; B has 21. All image placements verify at
300 dpi. Both include editable Arial manuscript row labels and panel letters;
B additionally contains seven editable SSIM numbers matching the manuscript.
See `SuppFig7_EXPORTS.md` in that folder. Two new source notebook export cells
were added while preserving every original cell/output. This export used the
existing trained model and VAE decoder and cached the exact arrays for future
formatting changes without inference.

## Completed final manuscript export batch

Fig. 2D, S10–12, S15A–E, S16A–C, S18A–F, S19 and S23 are now generated. See [FINAL_MANUSCRIPT_PDFS.md](FINAL_MANUSCRIPT_PDFS.md) for clickable files, exact raster/vector content, source/caching details, native raster page sizes and validation. The new batch contains 28 PDFs, including individual S18A metrics and S18B KDE plots.

## Source-generation correction for S11, S12, S19 and S23

The initial manuscript-raster conversions of these figures have been replaced by PDFs saved at 300 dpi from the existing source notebook figures. S11 and S12 are saved as separate source panels; S19 retains its grid filename; S23 has an image-only version plus its original editable F1-score grid. The manuscript schematic is omitted. Superseded snapshots are archived. See [the current file index](FINAL_MANUSCRIPT_PDFS.md) for the 31 active PDFs, source details and validation.

S17B is also generated at 300 dpi from the seed-conditioned plotting cell in `sim_to_exp_diffusion/controlnet_essential/plot_Fig5.ipynb`. Both PDF and PNG are saved without added labels. The PDF contains 25 separate image placements, each verified at 300 dpi.

S18E was corrected after its initial PDF mistakenly repeated S18F. E now contains total skeleton length (mm); F contains total skeleton tips. Both were regenerated from the original plotting cells, verified point-by-point against the source table, and guarded against saving the wrong active plot.
