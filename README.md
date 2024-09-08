# Multivariate Nonlinear Gradient Descent Curve Fitting

This package provides a multivariate nonlinear curve fitter that utilizes gradient descent for optimization based on mean square error. It is designed for solving complex curve-fitting problems where the relationship between input data and target variables is nonlinear. The package supports numerical gradient calculation by default but allows users to specify an analytical gradient function for more complex models to improve performance and precision.

## Features
- **Flexible Model Definition**: Users can define their own model functions and gradients (optional).
- **Gradient Descent Optimization**: Automatically uses numerical gradients but can switch to analytical gradients when specified.
- **Data Scaling**: Optional data scaling for more stable optimization.
- **Customizable Learning Rate and Parameters**: Users can adjust the learning rate, decay factor, and other parameters to fine-tune the optimization process.

## Installation
To install the package, use the following command:

```bash
pip install curvefit_gd
```

You can then import the FunctionFitter using the following command

```bash
from curvefit_gd import FunctionFitter
```


## How it works

The optimization process follows these steps:

### 1. Specify the Model Function:
The model function defines the relationship between your input data and the target output. An example model could be a combination of exponential and quadratic terms.

#### Example Model Structure:

$$f(x_1, x_2) = c_0 \cdot e^{x_1} + (1 + c_1 \cdot x_2^2)$$
```python
def model_function(x, coefficients):

    x1, x2 = x[0], x[1]
    func = coefficients[0] * np.exp(x1) + (1 + coefficients[1] * x2**2)
    return func
```
Where x1, x2 are input features, and c0, c1 are the coefficients to be learned.

### 2. Optionally Specify the Gradient Function:

For more complex models, you may want to provide your own gradient function to improve optimization accuracy.

Example Gradient Terms:


To minimize the loss function using gradient descent, the gradients of the model with respect to the coefficients are calculated as follows:

The gradient with respect to (c0) is:

$$\frac{\partial f(x_1, x_2)}{\partial c_0} = e^{x_1}$$

The gradient with respect to (c1) is:

$$\frac{\partial f(x_1, x_2)}{\partial c_1} = x_2^2$$

```python
def gradient_terms(x_data, coefficients):

    term1 = np.exp(x_data[0])
    term2 = x_data[1]**2
    return np.array([term1, term2])
```

### 3. Fit the Model:
Provide your data and the model will ```fit()``` the curve to the data using gradient descent, adjusting coefficients to minimize the error.

### 4. Predict New Values:
After training, use the ```predict()``` method to generate predictions based on new input data.

## Scaling Data

It is highly recommended to scale your data for better stability in the optimization process. However, if you choose to scale your data, you must:

- Use the same scaling parameters for any future input data, ideally through the predict() method.
- If you choose not to scale your data, the coefficients will be easier to interpret but may result in less stable optimization.


## Class and Methods

### `FunctionFitter`
  
The primary class for performing curve fitting using gradient descent.

## Constructor
```python
FunctionFitter(model_func, learning_rate=1e-3, decay_factor=0, max_iterations=100000,user_gradients=None, error_tolerance=1e-5, gradient_tolerance=1e-5)

```
## Parameters

- **model_func** (_callable_): The function defining the relationship between input data and target values. **(required)**
  
- **learning_rate** (_float_): The initial learning rate for the optimizer. Default is `1e-3`. **(optional)**

- **decay_factor** (_float_): Decay factor for the learning rate over iterations. Default is `0`. **(optional)**

- **max_iterations** (_int_): Maximum number of iterations to perform during optimization. Default is `100,000`. **(optional)**

- **user_gradients** (_callable_): A user-defined function for calculating gradients. Default is `None` (numerical gradients will be used). **(optional)**

- **error_tolerance** (_float_): The threshold for convergence based on error reduction. Default is `1e-5`. **(optional)**

- **gradient_tolerance** (_float_): The threshold for convergence based on the gradient's norm. Default is `1e-5`. **(optional)**

---

### `fit(x_data, y_data)`

Fit the model to the input data using gradient descent.

- **x_data** (_numpy.ndarray_): Input data (features).

- **y_data** (_numpy.ndarray_): Target values.

---

### `predict(x_data)`

Generate predictions using the optimized model.

- **x_data** (_numpy.ndarray_): Input data (features).

---

### `get_coefficients()`

Return the optimized coefficients after fitting the model.

---

### `get_error()`

Return the final error (mean squared error) after the optimization process.


## Example Usage

1. Define your model function.
2. (Optional) Define your gradient function for complex models.
3. Fit the model to your data.
4. Use the trained model to make predictions.

For a detailed example, refer to the `gradient_optimizer_example.py` file in this repository.


## Recommendations

- **Data Scaling**: We highly recommend scaling your data for better optimization stability. However, if you scale, the coefficients must be applied to scaled inputs. The `predict()` method handles this if scaling is used.

- **Model Tuning**: Start with a basic set of parameters (learning rate, decay, etc.) and adjust based on model performance.

## License

This project is licensed under the MIT License.

