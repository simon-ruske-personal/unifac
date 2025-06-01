import numpy as np
import pytest
from ..unifac.model_parameters import UnifacModelParameters

def test_unifac_parameters_can_be_constructed():
    
    # Act
    parameters = UnifacModelParameters(
        np.array([1.0, 2.0, 3.0]),
        np.array([2.1, 3.4, 5.2]),
        np.array([1, 2, 3, 4, 5]),
        np.array([[1.1, 2.1, 3.2], [1.2, 3.2, 4.2]])
    )

    # Assert
    np.testing.assert_equal(parameters.Q, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_equal(parameters.R, np.array([2.1, 3.4, 5.2]))
    np.testing.assert_equal(parameters.Data_main, np.array([1, 2, 3, 4, 5]))
    np.testing.assert_equal(parameters.Data_2, parameters.Data_2, np.array([[1.1, 2.1, 3.2], [1.2, 3.2, 4.2]]))

def test_unifac_parameters_can_be_constructed_from_default_files():
    
    # Act 
    parameters = UnifacModelParameters.build_default()

    # Assert
    assert parameters.Q.shape == (572,)
    assert parameters.R.shape == (572,)
    assert parameters.Data_2.shape == (76, 76)
    assert parameters.Data_main.shape == (572,)

    assert pytest.approx(0.3652) == parameters.R[10]
    assert pytest.approx(0.12) == parameters.Q[10]
    assert pytest.approx(-664.4) == parameters.Data_2[14][42]