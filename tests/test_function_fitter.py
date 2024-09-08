import numpy as np
import pytest
from curvefit_gd import FunctionFitter


def test_infer_coefficients():
    # Dummy model function
    def dummy_model(x, coeffs):
        return coeffs[0] * x

    x_data = np.array([[1], [2], [3]])
    y_data = np.array([2, 4, 6])  # Target data

    fitter = FunctionFitter(model_func=dummy_model)
    assert fitter._infer_num_coefficients(x_data) == 1


def test_fit_model():
    # Dummy model function
    def dummy_model(x, coeffs):
        return coeffs[0] * x

    x_data = np.array([1, 2, 3])
    y_data = np.array([2, 4, 6])  # Target data

    fitter = FunctionFitter(model_func=dummy_model,
                            learning_rate=0.001, max_iterations=10000)
    fitter.fit(x_data, y_data)

    coeffs = fitter.get_coefficients()
    assert np.isclose(coeffs[0], 2.0, atol=0.5)


def test_prediction():
    # Dummy model function
    def dummy_model(x, coeffs):
        return coeffs[0] * x

    x_data = np.array([1, 2, 3])
    y_data = np.array([2, 4, 6])  # Target data

    fitter = FunctionFitter(model_func=dummy_model,
                            learning_rate=0.001, max_iterations=10000)
    fitter.fit(x_data, y_data)

    predictions = fitter.predict(np.array([[4]]))
    assert np.isclose(predictions[0], 8.0, atol=0.1)

if __name__ == "__main__":
    test_prediction()
