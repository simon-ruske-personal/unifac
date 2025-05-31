from dataclasses import dataclass
from pathlib import Path

import numpy as np
import os

@dataclass
class UnifacModelParameters():
    Q: np.ndarray[tuple[int], np.dtype[float]]
    R: np.ndarray[tuple[int], np.dtype[float]]
    Data_main: np.ndarray[tuple[int], np.dtype[int]]
    Data_2: np.ndarray[tuple[int, int], np.dtype[float]]

    @staticmethod
    def build_default():
        model_parameter_directory = Path(os.path.dirname(__file__)) / "model_parameters"
        Q = np.genfromtxt(model_parameter_directory / "Q.txt", dtype='float64')
        R = np.genfromtxt(model_parameter_directory / "R.txt", dtype='float64')
        data_main = np.genfromtxt(model_parameter_directory / "UFC_Data_main.txt", dtype='int')
        data_2 = np.genfromtxt(model_parameter_directory / "UFC_Data2.txt", dtype='float64')

        return UnifacModelParameters(Q, R, data_main, data_2)