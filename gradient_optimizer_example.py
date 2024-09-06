import numpy as np
from main.gradient_descent_optimizer import FunctionFitter
from sklearn.preprocessing import StandardScaler


def model_function(x: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """
    A customizable model function for users. This example includes exponential 
    and quadratic terms, which can be modified as needed.

    Parameters:
    - x (np.ndarray): Input data, which may consist of multiple dimensions.
    - coefficients (np.ndarray): Model coefficients to be optimized.

    Returns:
    - np.ndarray: Predicted values based on the input data and coefficients.
    """
    x1, x2 = x[0], x[1]  # Assumes two features in the input data
    func = coefficients[0] * np.exp(x1) + (1 + coefficients[1] * x2**2)
    return func


def gradient_terms(x_data: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """
    Computes the gradient terms for each coefficient, to be used in the optimization process.
    Users may customize this function based on their model structure.

    Parameters:
    - x_data (np.ndarray): Input data used in the model.
    - coefficients (np.ndarray): Current model coefficients.

    Returns:
    - np.ndarray: Gradient terms for each coefficient.
    """
    term1 = np.exp(x_data[0])
    term2 = x_data[1]**2
    return np.array([term1, term2])


if __name__ == "__main__":
    # Step 1: Load your own data.
    # Here, artificial data is generated for demonstration purposes (users should replace this with real data).
    x1_data = np.linspace(1, 10, 100)  # Example of feature 1
    x2_data = np.linspace(7, 8, 100)   # Example of feature 2
    # Coefficients for generating the target data
    coefficients_to_generate_data = np.array([5, 9])

    # Combine the input data into a 2D array (2 features, 100 data points)
    x_data = np.vstack([x1_data, x2_data])

    # Generate the target data using the example model function, adding noise for realism
    y_data = model_function(
        x_data, coefficients_to_generate_data) + np.random.randn(100) * 0

    # Step 2: Scale the data (highly recommended for model stability)
    scaler = StandardScaler()
    # Scaling the data (transpose to match feature format)
    x_data_scaled = scaler.fit_transform(x_data.T).T

    # Important Note:
    # - If scaling is applied, ensure the same scaler object is used to transform any future input data.
    # - The coefficients learned from scaled data should only be applied to scaled inputs.
    # - If scaling is not used, the coefficients will work directly with raw input data, but model stability may be impacted.

    # Step 3: Initialize the FunctionFitter optimizer
    optimizer = FunctionFitter(
        model_func=model_function,    # Required: The model function to optimize
        # Optional: Base learning rate (default is 1e-3)
        learning_rate=1e-4,
        # Optional: Learning rate decay factor (default is 0)
        decay_factor=0,
        # Optional: Maximum number of iterations (default is 1000000)
        max_iterations=100000
        # user_gradients=gradient_terms  # Uncomment if custom gradients are provided
    )

    # Step 4: Run the optimization process to fit the model
    optimizer.fit(x_data_scaled, y_data)

    # Step 5: Output the optimized coefficients and the final error
    print("Optimized Coefficients:", optimizer.get_coefficients())
    print("Final Error (MSE):", optimizer.get_error())
