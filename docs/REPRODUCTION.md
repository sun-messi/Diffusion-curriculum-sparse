# Reproduction and figure provenance

## Figure sources

The new README images are the manuscript-derived assets already reviewed for the project website:
- `curriculum.png`: local manuscript `image/infographic_with_samples.png`.
- `uvit-results.png`: rendered from local manuscript `image/uvit_three_way_orig.pdf`.

They are referenced by the local ICML2026.tex manuscript. This provenance does not establish that the local manuscript is identical to the final workshop PDF. They are not regenerated experiments. Old root PNGs are preserved but no longer used by README.

## Code review

The original mask class docstring described both patch-embedding and decoder hooks, but the actual trainer installs only the patch-embedding hook. Documentation now describes the implementation. The class was extracted to `libs/curriculum.py` so tests do not import training, datasets, W&B and FID dependencies. Valid input behavior is preserved, including the historical minimum of one active channel even at sparsity=1.

Invalid sparsity ratios, nonfinite values, input dimensions and time intervals now fail explicitly. Training schedules, loss formula, inference bypass and checkpoints are unchanged.

The old print-only `test_sparsity_mask.py` is a manual diagnostic, not an assertion-based regression suite. Use `python -m unittest discover -s tests -v`.

## Validation boundary

CPU tests and Python syntax checks were run for this refresh. No dataset download, checkpoint sampling, GPU/multi-GPU training, FID reproduction, or performance benchmark was run. CUDA/library compatibility and data access remain environment-dependent.

Other inherited LDM/text-to-image and Jacobian scripts retain their own configuration and asset requirements. They are not advertised as validated by the small mask test.

Before an experimental comparison, match batch size, seed, optimizer, training budget, reference statistics and sampler across configs; supplied configs should not be assumed to differ in only one factor.
