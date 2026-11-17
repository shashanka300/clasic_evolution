# TMLR submission checklist

*Started 2026-11-17. Requirements from the TMLR author guide and editorial policies (jmlr.org/tmlr), read
2026-11-17.*

## What TMLR requires

- **Template:** the PDF must be built with the TMLR style file (github.com/JmlrOrg/tmlr-style-file).
- **Double-blind:** the PDF and the supplementary material must be anonymised. Don't link to any version that names
  the authors. Posting to arXiv is allowed at any time.
- **Length:** any length, but it should be justified by the content; unusually long main text may slow review.
  Appendices go after the references in the same PDF.
- **Supplementary material:** up to 100 MB, PDF or ZIP. Reviewers look at it at their discretion.
- **Broader-impact statement:** required only if the work carries a significant risk of harm. Not needed here.
- **Platform:** OpenReview. Every author needs a complete profile (affiliations, conflicts, publication history,
  funding). The author list is fixed at submission.
- **Acceptance criteria:** claims supported by accurate, convincing and clear evidence, and of interest to some of
  TMLR's audience. Novelty is not required, but incorrect novelty claims are grounds for rejection.
- **No fees.**

## Status

| item | status |
|---|---|
| LaTeX manuscript `paper/tmlr/main.tex` + `references.bib` | done; builds with `tectonic main.tex` (15 pages incl. appendix), no undefined refs |
| TMLR style file (`tmlr.sty`, `tmlr.bst`) | **needs your OK to download** from the official repo; `main.tex` switches to it automatically when the files are in `paper/tmlr/` |
| Anonymised PDF | done (authors "Anonymous", no repo links, no machine or user names) |
| Supplementary ZIP | the release export (`scripts/make_release.sh`) is 54 MB and already scrubbed, so it fits the 100 MB limit; rebuild it from the current commit before zipping |
| Authors, affiliations, OpenReview profiles | **you** |
| Title | provisional; **you** confirm |
| Bessa et al. (e-means) and Bloch et al. (Flash-GMM) first names in `references.bib` | to fill from the arXiv pages |
| Hoffman et al. 2010 author order | proceedings say Hoffman, Bach, Blei; check the PDF |
| References not yet checked online | about 20 (datasets, libraries, scaling-law papers); listed in `paper/references.md` |
| Appendix B (hyperparameter grids) | **TODO**: fill from the job files |
| Figures | vector PDFs from `results/figures/`; check legibility at print size |

## Likely reviewer questions (prepare rebuttal text)

1. *"This is stepwise EM renamed."* We agree and say so in §1. The contributions are the cross-family time-to-batch
   study, the revival ablation and the law.
2. *"The unchanged k-means law is almost as good as your per-cell law."* True on median LOAO (0.155% vs 0.149%,
   Appendix D). The per-cell law's gain is concentrated on LDA, and it is worse on HMMs. This is stated in Appendix D;
   consider moving it into §5.5.
3. *"Only 3 seeds for timing and the rare-group ablation."* State it; the epoch results use 5 seeds.
4. *"The reference is EM@25, not converged EM."* Limitation 2. Converged-reference numbers would strengthen this.
5. *"LDA contradicts the story."* The paper already reports it as the exception throughout.
