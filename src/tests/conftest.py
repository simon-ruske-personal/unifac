import os
from pathlib import Path
import pytest
from scipy.sparse import coo_matrix

import numpy as np

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

def load_files(file_name_flag: str, file_name_stoich: str, delimiter: str = ' '):

  n_rows, rows, cols, vals, group_flags = parse_input_files(file_name_flag, file_name_stoich, delimiter)

  group_flag_array = np.array(list(group_flags), 'int')
  group_flag_array.sort()
  
  max_group_num = len(group_flag_array)
  d = dict(zip(group_flag_array, range(max_group_num)))

  for i, _ in enumerate(cols):
     cols[i] = d[cols[i]]

  v = coo_matrix((vals, (rows, cols)), shape = (n_rows, max_group_num))
  v = v.toarray()	

  return v, group_flag_array

def parse_input_files(file_name_flag: str, file_name_stoich: str, delimiter: str):

  with (
    open(file_name_flag, "r") as flag_file,
    open(file_name_stoich, "r") as stoich_file,
  ):
    
    n_rows, _ = flag_file.readline().strip('\n').split()
    stoich_file.readline().strip('\n').split() # ignore duplicate
    n_rows = int(n_rows)

    rows, cols, vals, group_flags = [], [], [], set()

    for i, (flag_line, stoich_line) in enumerate(zip(flag_file, stoich_file)):

      parsed_flag_line = flag_line.split(delimiter)[:-1]
      parsed_stoich_line = stoich_line.split(delimiter)[:-1]

      for flag, stoich in zip(parsed_flag_line, parsed_stoich_line):
        
        if flag == '00':
          continue
        
        rows.append(i)
        cols.append(int(flag))
        vals.append(int(stoich))
        group_flags.add(flag)

  return n_rows, rows, cols, vals, group_flags