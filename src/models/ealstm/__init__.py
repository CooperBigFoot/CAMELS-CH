"""EA-LSTM model implementation module.

EA-LSTM (Entity-Aware LSTM) is a model architecture based on the paper:
"Towards learning universal, regional, and local hydrological behaviors via 
machine learning applied to large-sample datasets" by Kratzert et al. (2019)
https://hess.copernicus.org/articles/23/5089/2019/

This module provides implementations for:
1. EALSTMConfig - Configuration class for EA-LSTM models
2. EALSTM - PyTorch implementation of the EA-LSTM architecture
3. LitEALSTM - PyTorch Lightning wrapper for training and evaluation
"""

from .config import EALSTMConfig
from .model import EALSTM, EALSTMCell
from .lightning import LitEALSTM

__all__ = [
    "EALSTMConfig",
    "EALSTM",
    "EALSTMCell",
    "LitEALSTM",
]
