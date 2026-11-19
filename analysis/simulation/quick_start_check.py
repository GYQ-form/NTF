"""Quick-start acceptance for UniST env + InterpolAI weights (Step 0).

Builds two adjacent train slices from the 10w simulation data
(slice_3 / slice_5, 2000 cells each) and runs the official
predict_middle_slice in mode=fast (geometry/weights check) and mode=inr
(full SUICA_pro pipeline check). Prints timing and output structure.
"""

import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "6")

import numpy as np
import anndata

# InterpolAI SavedModel segfaults on GPU with pip TF 2.19 (CUDA 12.5 stack,
# driver 575); CPU inference is fast enough (1024x1024 tile ~4.3 s) and is
# inference-only, so TF is pinned to CPU. SUICA_pro (torch) still uses the GPU.
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

MODE = sys.argv[1] if len(sys.argv) > 1 else "fast"
WORK = sys.argv[2] if len(sys.argv) > 2 else None

import unist
from unist import predict_middle_slice
from unist.one_line_code.paths import require_interpolai_model

print("unist version:", getattr(unist, "__version__", "unknown"))
print("InterpolAI model dir:", require_interpolai_model())

DATA = "/home/gongyuqiao/ur_annotation/NTF/mytrain/data/simulation/scale_test/10w.h5ad"
ad = anndata.read_h5ad(DATA)
rng = np.random.default_rng(0)

slices = {}
for sid in ["slice_3", "slice_5"]:
    sub = ad[ad.obs["slice_id"] == sid].copy()
    idx = rng.choice(sub.n_obs, size=min(2000, sub.n_obs), replace=False)
    slices[sid] = sub[idx].copy()
    print(sid, "n_obs:", slices[sid].n_obs, "z median:",
          float(np.median(slices[sid].obsm["spatial"][:, 2])))

t0 = time.time()
kwargs = dict(mode=MODE, spatial_key="spatial")
if MODE == "fast":
    kwargs["transfer_expression"] = True
else:
    kwargs["work_dir"] = WORK or "/tmp/unist_quickstart_inr"
out = predict_middle_slice(slices["slice_3"], slices["slice_5"], **kwargs)
dt = time.time() - t0

print(f"=== mode={MODE} finished in {dt:.1f}s ===")
print("out n_obs x n_vars:", out.n_obs, "x", out.n_vars)
print("obs columns:", out.obs.columns.tolist())
print("layers:", list(out.layers.keys()))
print("obsm keys:", list(out.obsm.keys()))
print("X dtype:", out.X.dtype, "X range:", float(np.min(out.X)), float(np.max(out.X)))
print("z of output:", float(np.median(np.asarray(out.obsm['spatial'])[:, 2])))
print("uns unist keys:", list(out.uns.get("unist", {}).keys()))
