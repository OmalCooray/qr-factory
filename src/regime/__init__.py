"""Regime detection — multiple detector implementations."""

from .filter import TrendRangeFilter, RegimeInfo
from .protocol import RegimeLabel, RegimeDetector
from .adapter_hmm import HMMRegimeDetector
from .choppiness import ChoppinessDetector
from .hurst import HurstDetector
from .bbw_detector import BBWidthDetector
from .gmm_detector import GMMDetector
from .kmeans_detector import KMeansDetector
from .markov_switching import MarkovSwitchingDetector
from .pelt_detector import PELTDetector
from .synthetic import generate_synthetic_data

__all__ = [
    "TrendRangeFilter",
    "RegimeInfo",
    "RegimeLabel",
    "RegimeDetector",
    "HMMRegimeDetector",
    "ChoppinessDetector",
    "HurstDetector",
    "BBWidthDetector",
    "GMMDetector",
    "KMeansDetector",
    "MarkovSwitchingDetector",
    "PELTDetector",
    "generate_synthetic_data",
]
