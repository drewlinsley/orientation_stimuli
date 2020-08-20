import numpy as np
import os
from glob import glob


no_surround_paths = glob(os.path.join("orientation_probe_no_surround_outputs", "*.npz"))  # noqa
surround_paths = glob(os.path.join("orientation_probe_outputs", "*.npz"))
no_surround_paths.sort(key=os.path.getmtime)
surround_paths.sort(key=os.path.getmtime)
no_surround_path = no_surround_paths[-1]
surround_path = surround_paths[-1]
no_surround = np.load(no_surround_path, allow_pickle=True, encoding="latin1")
surround = np.load(surround_path, allow_pickle=True, encoding="latin1")

outdir = "both_outputs"
if not os.path.exists(outdir):
    os.makedirs(outdir)

s = {}
for k, v in surround.items():
    s[k] = v


n = {}
for k, v in no_surround.items():
    n[k] = v

n["test_dict"] = np.concatenate((n["test_dict"], s["test_dict"]), 0)
np.savez(os.path.join(outdir, "both_outputs"), **n)


# Fix up the meta, too
meta = np.load(
    os.path.join("orientation_probe_no_surround_outputs", "1.npy"),
    allow_pickle=True,
    encoding="latin1")
meta = np.concatenate((meta, meta), 0)
np.save(os.path.join(outdir, "1.npy"), meta)
