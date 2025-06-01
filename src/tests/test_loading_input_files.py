import os
from pathlib import Path
import numpy as np
import pytest
from src.utils.load_test_files import load_files

@pytest.mark.parametrize(
    "group_filename, stoich_filename, expected_molecules, expected_groups, v_checksum, group_checksum",
    [
        ("group_flag_183_83.txt", "group_stoich_183_83.txt", 183, 23, 1353, 2579),
        ("group_flag_2727_83.txt", "group_stoich_2727_83.txt", 2727, 56, 20375, 4274),
        ("group_flag_103666_83.txt", "group_stoich_103666_83.txt", 103666, 38, 1715767, 3137),
    ]
)
def test_load_file_checksums(
    group_filename: str,
    stoich_filename: str,
    expected_molecules: int,
    expected_groups: int,
    v_checksum: float,
    group_checksum: float,
):
    
    test_path = Path(os.path.dirname(__file__)) / "test_files"
    v, group_flag_array = load_files(test_path / group_filename, test_path / stoich_filename)
    
    assert v.shape == (expected_molecules, expected_groups)
    assert group_flag_array.shape == (expected_groups,)
    assert v.sum() == pytest.approx(v_checksum)
    assert group_flag_array.sum() == pytest.approx(group_checksum)