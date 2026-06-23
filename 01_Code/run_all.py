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
    ("07c_rolling_har_d_qr.py",    "Rolling HAR-D-QR (crisis dummy only)"),
    ("07d_rolling_har_xd_qr.py",   "Rolling HAR-XD-QR (ENSO + crisis dummy)"),
    ("07f_rolling_har_x_lagged_qr.py", "Lag-decay diagnostic (lagged ENSO)"),
    ("07h_rolling_har_x_vol_qr.py",    "HAR-X-QR with ENSO volatility"),
    ("08_benchmarks.py",           "Historical Vol + GARCH benchmarks"),
]

SCRIPTS_EVAL = [
    ("compute_har_ols_ql.py",          "HAR-OLS pseudo-quantile loss table"),
    ("07e_precrisis_evaluation.py",    "Pre-crisis evaluation (HAR-QR vs HAR-X-QR)"),
    ("07e2_precrisis_vol_evaluation.py", "Pre-crisis evaluation (ENSO vol specs)"),
    ("12_model_confidence_set.py",     "Model Confidence Set (Hansen et al. 2011)"),
    ("13_stationarity_tests.py",       "ADF and Phillips-Perron tests"),
    ("18_enso_regime_test.py",         "ENSO regime test (4-way comparison)"),
    ("20_regime_test_metrics.py",      "Regime test multi-metric robustness"),
    ("21_boundary_sensitivity.py",     "Crisis-dummy boundary sensitivity (slow)"),
    ("22_coefficient_estimates.py",    "Full-sample coefficient tables"),
    ("23_christoffersen_coverage.py",  "Christoffersen coverage tests"),
]

SCRIPTS_FIGURES = [
    ("09_comparison_plots.py",         "Comparison plots"),
    ("10_upper_tail_comparison.py",    "Upper tail comparison"),
    ("10b_crisis_zoom.py",             "Crisis zoom plot"),
    ("10c_crisis_zoom_harxqr_vs_harqr.py", "Crisis zoom HAR-X-QR vs HAR-QR"),
    ("11_exceedance_plot.py",          "Exceedance plot"),
    ("14_mcs_heatmap.py",              "MCS heatmap"),
    ("15_loss_heatmaps.py",            "Loss heatmaps"),
    ("16_mse_mae_comparison.py",       "MSE/MAE comparison"),
    ("17_robustness_metrics_plot.py",  "Robustness metrics plot"),
    ("07i_enso_cocoa_ccf.py",          "ENSO-cocoa cross-correlation"),
    ("07k_expanding_window_enso_gain.py", "Expanding-window ENSO gain"),
    ("07k_lag_decay_figure.py",        "Lag-decay figure"),
    ("07l_expanding_window_enso_matrix.py", "Expanding-window ENSO matrix"),
    ("07l_enso_vol_comparison_figure.py",   "ENSO vol comparison figure"),
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

    if quick_mode:
        scripts = SCRIPTS_QUICK
        mode_label = "QUICK MODE (data + descriptive only)"
    else:
        scripts = SCRIPTS_QUICK + SCRIPTS_MODELS + SCRIPTS_EVAL + SCRIPTS_FIGURES
        mode_label = "FULL PIPELINE"

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
