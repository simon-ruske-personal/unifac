import numpy as np

from src.unifac.model_parameters import UnifacModelParameters
from src.unifac.model import calculate_unifac_coefficients

def test_unifac_component_example_183_83(test_inputs_183_83, test_expectations_183_83):

    # Arrange
    v, group_flag_array = test_inputs_183_83

    number_of_molecules = len(v)
    x = np.ones(number_of_molecules) / number_of_molecules
    parameters = UnifacModelParameters.build_default()

    # Act
    gamma = calculate_unifac_coefficients(x, v, group_flag_array, 298.15, parameters)

    # Assert
    np.testing.assert_allclose(gamma, test_expectations_183_83)
    assert test_inputs_183_83
    assert test_expectations_183_83.shape == (183,)

def test_unifac_component_example_2727_83(test_inputs_2727_83, test_expectations_2727_83):

    # Arrange 
    v, group_flag_array = test_inputs_2727_83

    number_of_molecules = len(v)
    x = np.ones(number_of_molecules) / number_of_molecules
    parameters = UnifacModelParameters.build_default()

    # Act
    gamma = calculate_unifac_coefficients(x, v, group_flag_array, 298.15, parameters)

    # Assert
    np.testing.assert_allclose(gamma, test_expectations_2727_83)

def test_unifac_component_example_103666_83(test_inputs_103666_83, test_expectations_103666_83):

    # Arrange
    v, group_flag_array = test_inputs_103666_83

    number_of_molecules = len(v)
    x = np.ones(number_of_molecules) / number_of_molecules
    parameters = UnifacModelParameters.build_default()

    # Act
    gamma = calculate_unifac_coefficients(x, v, group_flag_array, 298.15, parameters)

    # Assert
    np.testing.assert_allclose(gamma, test_expectations_103666_83)