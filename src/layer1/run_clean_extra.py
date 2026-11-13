"""Layer 1 (extra): many more CLEAN slices of every dataset.

Why? The detectors had only 51 healthy examples out of 1,475 rows, so they never learned
where "healthy" ends and "mildly flawed" begins. Clean rows need no flaw conditions, so
they are cheap to make: every random slice here is an untouched, healthy version of a dataset.

Usage:
    python -m src.layer1.run_clean_extra           # the full run
    python -m src.layer1.run_clean_extra --quick   # tiny test run
"""
import sys
import time

import numpy as np
import pandas as pd

from src.layer1.conditions import build_condition_list
from src.layer1.run_experiments import RAW_DIR, RESULTS_DIR, RANDOM_STATE, make_slices, run_one

N_CLEAN_SLICES = 12


def main(quick=False):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    clean_condition = build_condition_list()[0]          # the "clean" condition: nothing is broken
    names = sorted(p.stem for p in RAW_DIR.glob("*.csv"))
    n_slices = N_CLEAN_SLICES
    out_file = RESULTS_DIR / "clean_extra_v2.csv"

    if quick:
        names = names[:2]
        n_slices = 2
        out_file = RESULTS_DIR / "clean_extra_v2_quick.csv"

    print(f"{len(names)} datasets x {n_slices} clean slices = {len(names) * n_slices} rows\n")

    rows, failures = [], 0
    start = time.time()

    for ds_index, name in enumerate(names):
        clean_df = pd.read_csv(RAW_DIR / f"{name}.csv")
        rng = np.random.default_rng([RANDOM_STATE + 1000, ds_index])    # different seed: brand-new slices
        slices = make_slices(clean_df, rng, n_slices)[1:]               # skip 'full' (already in the main file)

        for i, (_, slice_df) in enumerate(slices, start=1):
            try:
                rows.append(run_one(name, f"cleanx{i}", slice_df, clean_condition, RANDOM_STATE))
            except Exception as e:
                failures += 1
                print(f"FAILED: {name} | cleanx{i} -> {e}")

        pd.DataFrame(rows).to_csv(out_file, index=False)                # save as we go
        print(f"Finished {name}: {len(rows)} rows so far ({time.time() - start:.0f}s)")

    out = pd.DataFrame(rows)
    print(f"\nDone. Saved {len(out)} rows to {out_file} (failures: {failures})")
    nan_counts = out.isna().sum()
    nan_counts = nan_counts[nan_counts > 0]
    if len(nan_counts):
        print("\nColumns containing NaN (tell me if you see this):")
        print(nan_counts.to_string())


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)