"""
run_all.py
==========
Master script to run all analysis steps in order.

Usage:
    python run_all.py          # run everything
    python run_all.py --quick  # run only data prep + descriptive (no models)

Execution order:
    00 -> 01 -> 02,03 -> 04 -> 05 -> 06,07 -> 08 -> 09 -> 10

Steps 02/03 and 06/07 are independent of each other but depend on prior steps.
"""

import subprocess
import sys
import time
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Scripts in execution order
SCRIPTS_QUICK = [
    ("00_data_loading.py",        "Loading and cleaning raw data"),
    ("01_yang_zhang_volatility.py", "Computing Yang-Zhang volatility"),
    ("02_descriptive_analysis.py", "Descriptive statistics and price plots"),
    ("03_volatility_analysis.py",  "Volatility analysis and stylized facts"),
    ("04_har_features.py",         "Building HAR features and targets"),
    ("05_proxy_validation.py",     "Yang-Zhang vs Realized Vol validation"),
]

SCRIPTS_MODELS = [
    ("06_rolling_har_ols.py",      "Rolling HAR-OLS (benchmark)"),
    ("07_rolling_har_qr.py",       "Rolling HAR-QR (core model)"),
    ("07b_rolling_har_x_qr.py",    "Rolling HAR-X-QR (with ENSO)"),
    ("08_benchmarks.py",           "Historical Vol + GARCH benchmarks"),
    ("09_var_backtest.py",         "VaR backtesting"),
    ("10_summary_plots.py",        "Summary comparison plots"),
    ("11_har_vs_har_x_comparison.py", "HAR-QR vs HAR-X-QR comparison (ENSO effects)"),
    ("12_model_confidence_set.py",    "Model Confidence Set (Hansen et al. 2011)"),
]


def run_script(script_name, description):
    """Run a single Python script and report timing."""
    path = os.path.join(BASE_DIR, script_name)
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"  Running: {script_name}")
    print(f"{'='*60}\n")

    start = time.time()
    result = subprocess.run(
        [sys.executable, path],
        capture_output=False,
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n  ERROR: {script_name} failed (exit code {result.returncode})")
        return False
    else:
        print(f"\n  Completed in {elapsed:.1f}s")
        return True


if __name__ == "__main__":
    quick_mode = "--quick" in sys.argv

    scripts = SCRIPTS_QUICK if quick_mode else SCRIPTS_QUICK + SCRIPTS_MODELS
    mode_label = "QUICK MODE (data + descriptive only)" if quick_mode else "FULL PIPELINE"

    print(f"\n{'#'*60}")
    print(f"  Cocoa Volatility Analysis Pipeline")
    print(f"  Mode: {mode_label}")
    print(f"  Scripts: {len(scripts)}")
    print(f"{'#'*60}")

    total_start = time.time()
    failed = []

    for script_name, desc in scripts:
        ok = run_script(script_name, desc)
        if not ok:
            failed.append(script_name)
            print(f"\n  Stopping due to error in {script_name}")
            break

    total_elapsed = time.time() - total_start

    print(f"\n{'#'*60}")
    print(f"  Pipeline Complete")
    print(f"  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    if failed:
        print(f"  FAILED: {', '.join(failed)}")
    else:
        print(f"  All {len(scripts)} scripts succeeded")
    print(f"{'#'*60}")
