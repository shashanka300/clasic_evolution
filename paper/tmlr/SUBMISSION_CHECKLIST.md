# TMLR submission checklist

*Started 2026-11-17. Requirements from the TMLR author guide and editorial policies (jmlr.org/tmlr), read
2026-11-17. `paper/tmlr/main.tex` is now the canonical manuscript; `paper/draft.md` is the older Markdown source.*

## What TMLR requires

- **Template:** the PDF must be built with the TMLR style file (github.com/JmlrOrg/tmlr-style-file).
- **Double-blind:** the PDF and the supplementary material must be anonymised. Don't link to any version that names
  the authors. Posting to arXiv is allowed at any time.
- **Length:** any length, but it should be justified by the content. Appendices go after the references in the same
  PDF.
- **Abstract:** one paragraph.
- **Supplementary material:** up to 100 MB, PDF or ZIP.
- **Broader-impact statement:** required only if the work carries a significant risk of harm. Not needed here.
- **Platform:** OpenReview. Every author needs a complete profile (affiliations, conflicts, publication history,
  funding). The author list is fixed at submission.
- **Acceptance criteria:** claims supported by accurate, convincing and clear evidence, and of interest to some of
  TMLR's audience. Novelty is not required, but incorrect novelty claims are grounds for rejection.
- **No fees.**

## Status

| item | status |
|---|---|
| Manuscript `main.tex`, built with the official TMLR style (`tmlr.sty`, `tmlr.bst`, `fancyhdr.sty`, Apache-2.0) | done; `tectonic main.tex` gives `main.pdf` with no undefined references and no overfull boxes |
| Abstract in one paragraph | done |
| Appendix A–F (data, grids, repairs, law fits, rare-group revival, compute) | done |
| Bibliography `references.bib` | done; author lists checked on arXiv for e-means, Weiss, Flash-KMeans, Flash-GMM; 24 entries checked online, the rest are standard references |
| Supplementary ZIP `supplementary.zip` (code, job files, 44,197 run records, figures, paper sources) | done, 5.4 MB (1,309 files), built by `scripts/make_release.sh` from the current commit; scanned for user, machine, repo and employer names. Not committed (gitignored); rebuild before upload if anything changes |
| **Authors, affiliations ("Independent Researcher"), OpenReview profiles** | **you** (your profile should still list your current employer for conflicts) |
| **Employer IP / publication clearance** | **you**, before submitting |
| Title | done (2026-11-17): "When Running Averages Are Enough: Few-Epoch EM for Mixtures, Sequences, Topics and Factorizations" |
| Upload on OpenReview | **you** |

## Likely reviewer questions (prepare rebuttal text)

1. *"This is stepwise EM renamed."* We agree and say so in §1. The contributions are the cross-family time-to-batch
   study, the revival ablation and the law.
2. *"The unchanged k-means law is almost as good as your per-cell law."* Now stated in §5.5 and Appendix D: the gain
   is concentrated on LDA, and the k-means law predicts HMMs better.
3. *"Only 3 seeds for timing and the rare-group ablation."* Stated; the epoch results use 5 seeds.
4. *"The reference is EM@25, not converged EM."* Limitation 2. Converged-reference numbers would strengthen this.
5. *"LDA contradicts the story."* The paper already reports it as the exception throughout.
