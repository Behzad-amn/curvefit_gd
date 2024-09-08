import numpy as np


class FunctionFitter:
    """
    A class for fitting a nonlinear multivariate model to input data using gradient descent.

    By default, numerical gradients are used for optimization. Users can also provide an
    analytical gradient function for more complex scenarios.
    """

    def __init__(self, model_func, learning_rate: float = 1e-3, decay_factor: float = 0, max_iterations: int = 100000,
                 user_gradients=None, error_tolerance: float = 1e-5, gradient_tolerance: float = 1e-5):
        """
        Initialize the FunctionFitter.

        Parameters:
        - model_func (callable): The model function that predicts values based on coefficients.
        - learning_rate (float): Initial learning rate for gradient descent. Default is 1e-4.
        - decay_factor (float): Factor to decay the learning rate over iterations. Default is 0.
        - max_iterations (int): Maximum number of iterations for optimization. Default is 100,000.
        - user_gradients (callable, optional): Function for computing analytical gradients. Default is None.
        - error_tolerance (float): The threshold for convergence based on error. Default is 1e-5.
        - gradient_tolerance (float): The threshold for convergence based on gradient norm. Default is 1e-5.
        """
        self.model_func = model_func
        self.base_learning_rate = learning_rate
        self.decay_factor = decay_factor
        self.max_iterations = max_iterations
        self.user_gradients = user_gradients
        self.error_tolerance = error_tolerance
        self.gradient_tolerance = gradient_tolerance
        self.coefficients = None
        self.final_error = None
        self.grad_type = 'Numerical'

    def _infer_num_coefficients(self, x_data):
        """Infer the number of coefficients required by the model function."""
        for i in range(1, 100):
            try:
                test_coefficients = np.ones(i)
                self.model_func(x_data, test_coefficients)
                return i
            except IndexError:
                continue
        raise ValueError("Unable to infer the correct number of coefficients.")

    def _calculate_error(self, x_data, y_data, coefficients):
        """Calculate the mean squared error between the predicted and actual target values."""
        predictions = self.model_func(x_data, coefficients)
        return np.mean((predictions - y_data) ** 2)

    def _compute_analytical_gradient(self, x_data, y_data, coefficients):
        """
        Compute the analytical gradient using user-provided gradient terms.
        
        This function assumes the user has provided a callable that returns gradient terms for each coefficient.
        """
        if self.user_gradients is None:
            raise ValueError("User-provided gradient function is missing.")

        predictions = self.model_func(x_data, coefficients)
        residuals = predictions - y_data
        grad_terms = self.user_gradients(x_data, coefficients)

        if len(grad_terms) != len(coefficients):
            raise ValueError(f"Expected {len(coefficients)} gradient terms, but got {
                             len(grad_terms)}.")

        grad = np.zeros_like(coefficients)
        for i in range(len(coefficients)):
            grad[i] = np.mean(2 * residuals * grad_terms[i])

        return grad

    def _compute_numerical_gradient(self, x_data, y_data, coefficients, epsilon=1e-6):
        """Calculate the gradient using numerical approximation (finite differences)."""
        grad = np.zeros_like(coefficients)
        for i in range(len(coefficients)):
            coefficients_step = np.copy(coefficients)
            coefficients_step[i] += epsilon
            loss_step = self._calculate_error(
                x_data, y_data, coefficients_step)
            loss = self._calculate_error(x_data, y_data, coefficients)
            grad[i] = (loss_step - loss) / epsilon
        return grad

    def _get_gradient(self, x_data, y_data, coefficients):
        """Return the analytical gradient if available, otherwise fall back to the numerical gradient."""
        if self.user_gradients:
            self.grad_type = 'Analytical'
            return self._compute_analytical_gradient(x_data, y_data, coefficients)
        else:
            return self._compute_numerical_gradient(x_data, y_data, coefficients)

    def _adjust_learning_rate(self, iteration):
        """Adjust the learning rate based on the current iteration and decay factor."""
        return self.base_learning_rate * (1 + self.decay_factor * iteration)

    def fit(self, x_data, y_data):
        """
        Fit the model to the data using gradient descent.

        Parameters:
        - x_data (np.array): Input features.
        - y_data (np.array): Target values.
        """
        self.num_coefficients = self._infer_num_coefficients(x_data)
        coefficients = np.random.randn(self.num_coefficients)

        for iteration in range(self.max_iterations):

            grad = self._get_gradient(x_data, y_data, coefficients)
            error = self._calculate_error(x_data, y_data, coefficients)

            if error < self.error_tolerance or np.linalg.norm(grad) < self.gradient_tolerance:
                print(f"Converged after {iteration} iterations.")
                self.coefficients = coefficients
                self.final_error = error
                return self

            learning_rate = self._adjust_learning_rate(iteration)
            coefficients -= learning_rate * grad

            if iteration % 10000 == 0:
                print(f"Iteration {iteration}/{self.max_iterations} | Learning rate: {learning_rate:.8f} | "
                      f"{self.grad_type} Gradient: {np.round(grad, 6)} | MSE Error: {error:.6f}")

        print("Failed to converge within the maximum number of iterations.")
        self.coefficients = coefficients
        self.final_error = error
        return self

    def predict(self, x_data):
        """Generate predictions using the trained coefficients."""
        if self.coefficients is None:
            raise ValueError("Model is not trained. Call fit() first.")
        return self.model_func(x_data, self.coefficients)

    def get_coefficients(self):
        """Return the trained coefficients."""
        if self.coefficients is None:
            raise ValueError("Model is not trained. Call fit() first.")
        return self.coefficients

    def get_error(self):
        """Return the final error after optimization."""
        if self.final_error is None:
            raise ValueError("Model is not trained. Call fit() first.")
        return self.final_error
