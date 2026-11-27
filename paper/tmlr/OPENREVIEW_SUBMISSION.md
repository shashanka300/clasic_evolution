# OpenReview / TMLR submission sheet

Copy-paste values for the TMLR submission form (openreview.net → Transactions on Machine Learning Research →
Submit). Field names on the live form may differ slightly; fill what it asks for.

## Files

| field | file |
|---|---|
| PDF (must be anonymous) | `paper/tmlr/main.pdf` (21 pages; "Anonymous authors"; checked: no name or email) |
| Supplementary material (ZIP, ≤ 100 MB) | `paper/tmlr/supplementary.zip` (about 6 MB, anonymized) |
| **Do not upload** | `paper/tmlr/main_preprint.pdf` (named version, for arXiv / camera-ready only) |

## Title

When Running Averages Are Enough: Few-Epoch EM for Mixtures, Sequences, Topics and Factorizations

## Abstract (plain text)

How many passes over the data does expectation-maximization (EM) need? Full-batch EM makes one pass per iteration and is usually run for tens of iterations, while its mini-batch form, stochastic-approximation (stepwise) EM, needs a step size tuned for each problem. We study stepwise EM with three additions: a step size annealed over a fixed epoch budget, revival of components that stop receiving data, and a rule that sets the initial step size from the size of the problem. We test it on Gaussian mixtures, hidden Markov models, latent Dirichlet allocation (LDA), dictionary learning and matrix factorization. On 12 of 19 dataset configurations it reaches the objective that full-batch EM attains after 25 iterations within 3 epochs, and where it matches within 5 epochs it is 5-82x faster in CPU wall-clock time. Against stepwise EM tuned on the same validation grid and given the same revival rule, it gets there sooner on 16 of 19 configurations. An ablation shows that revival is what makes stochastic EM work where components die, and the annealed schedule is what helps where they do not. LDA is the consistent exception: tuned stepwise EM is as good there, and the best step size is about five times smaller than the rule predicts. A step-size rule fitted for k-means transfers unchanged to the four families other than LDA, and a pre-registered test shows that one cross-family rule beats separate per-family constants. Code and all run records are released.

## Keywords (if asked)

stochastic-approximation EM; online EM; stepwise EM; mini-batch learning; Gaussian mixture models; hidden Markov
models; latent Dirichlet allocation; dictionary learning; step-size schedules; hyperparameter laws

## TL;DR (if asked)

With a budget-annealed step, component revival and a step-size law, stepwise EM matches 25 iterations of full-batch
EM within a few epochs on four of five latent-variable families; LDA is the exception.

## Submission length (if asked)

Long submission: the main text runs past 12 pages (references start on page 15; appendices follow).

## Declarations (author must confirm each)

| item | proposed answer | confirm |
|---|---|---|
| Authors | the submitting author only, via OpenReview profile (author list is fixed after submission) | ☐ |
| Profile | complete: Independent Researcher; current employer listed under Education & Career History for conflicts | ☐ |
| Funding | none (self-funded) | ☐ |
| Competing interests | none | ☐ |
| Human subjects / IRB | not applicable (public datasets only, no human subjects) | ☐ |
| Dual submission | not under review elsewhere; not a previously published conference paper | ☐ |
| Employer IP / publication clearance | obtained | ☐ |
| Broader impact statement | not included (no significant risk of harm) | ☐ |
| LLM use | assistive tools used; author takes full responsibility (TMLR policy allows this) | ☐ |

## After submitting

1. TMLR emails you asking for action-editor recommendations: pick editors who work on EM, variational inference,
   stochastic optimization or clustering, from the TMLR editorial board list.
2. Keep the OpenReview forum URL; the camera-ready needs it (`\openreview` in `main.tex`).
