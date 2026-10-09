"""Shared output paths for experiment results and generated figures."""
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results" / "generated"
FIGURES_DIR = ROOT / "figures" / "generated"
FIGURE_DATA_DIR = ROOT / "figures" / "data"


def result_path(name):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / name


def figure_path(name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR / name


def figure_data_path(name):
    FIGURE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURE_DATA_DIR / name
