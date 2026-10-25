# tools/

Analysis scripts. Nothing here is part of the training pipeline itself.

## `compare_baseline.py`

Measures what this method added over the Structure-from-Motion result it was
trained on. Both representations are rendered from the **same** recovered
camera poses at the same resolution and scored against the same photographs —
the SfM point cloud drawn as z-buffered points, the Gaussians through
`gsplat/render.py` — so the comparison is like for like rather than a
comparison of renderers.

```bash
python run_pipeline.py              # must have run
python tools/compare_baseline.py
```

Writes `output/<dataset>/07_splatting/baseline_comparison.json` with per-view
scores, and the two figures the README displays,
`docs/images/before_after.png` and `docs/images/improvement.png`.
