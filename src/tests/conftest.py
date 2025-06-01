import os
from pathlib import Path
import pytest

import numpy as np
from utils.load_test_files import load_files

@pytest.fixture()
def test_file_directory() -> Path:
  directory_of_file = os.path.dirname(__file__)
  return Path(directory_of_file) / "test_files"

@pytest.fixture()
def expectation_directory() -> Path:
  directory_of_file = os.path.dirname(__file__)
  return Path(directory_of_file) / "expectations"

@pytest.fixture()
def test_inputs_183_83(test_file_directory: Path):
  return load_files(
    test_file_directory / "group_flag_183_83.txt", 
    test_file_directory / "group_stoich_183_83.txt"
  )

@pytest.fixture()
def test_inputs_2727_83(test_file_directory: Path):
  return load_files(
    test_file_directory / "group_flag_2727_83.txt",
    test_file_directory / "group_stoich_2727_83.txt",
  )

@pytest.fixture()
def test_inputs_103666_83(test_file_directory: Path):
  return load_files(
    test_file_directory / "group_flag_103666_83.txt",
    test_file_directory / "group_stoich_103666_83.txt",
  )

@pytest.fixture()
def test_expectations_183_83(expectation_directory: Path):
  return np.genfromtxt(expectation_directory / "output_183_83.txt")

@pytest.fixture()
def test_expectations_2727_83(expectation_directory: Path):
  return np.genfromtxt(expectation_directory / "output_2727_83.txt")

@pytest.fixture()
def test_expectations_103666_83(expectation_directory: Path):
  return np.genfromtxt(expectation_directory / "output_103666_83.txt")