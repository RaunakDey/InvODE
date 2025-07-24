"""
Error Functions Module (erf)
============================

This module provides a collection of common error/loss functions for ODE parameter 
optimization. These functions are designed to work with the invode optimization 
framework and provide standardized metrics for model fitting.

All error functions follow the signature: error_func(y_pred) -> float
where y_pred is the model prediction and the function returns a scalar error value.
"""

import numpy as np
from typing import Union, Optional, Callable, Tuple
import warnings


class ErrorFunction:
    """
    Base class for error functions with data storage and validation.
    
    This class handles common functionality like data storage, validation,
    and provides a consistent interface for all error functions.
    """
    
    def __init__(self, data: np.ndarray, **kwargs):
        """
        Initialize error function with reference data.
        
        Parameters
        ----------
        data : np.ndarray
            Reference/observed data to compare predictions against.
        """
        self.data = np.asarray(data)
        if self.data.size == 0:
            raise ValueError("Data array cannot be empty")
    
    def __call__(self, y_pred: np.ndarray) -> float:
        """
        Compute error between prediction and reference data.
        
        Parameters
        ----------
        y_pred : np.ndarray
            Model predictions to compare against reference data.
            
        Returns
        -------
        float
            Computed error value.
        """
        raise NotImplementedError("Subclasses must implement __call__ method")
    
    def _validate_prediction(self, y_pred: np.ndarray) -> np.ndarray:
        """Validate and format prediction array."""
        y_pred = np.asarray(y_pred)
        
        if y_pred.shape != self.data.shape:
            raise ValueError(f"Prediction shape {y_pred.shape} does not match "
                           f"data shape {self.data.shape}")
        
        if not np.isfinite(y_pred).all():
            warnings.warn("Prediction contains non-finite values")
            return np.inf
            
        return y_pred


class MSE(ErrorFunction):
    """
    Mean Squared Error (MSE) function.
    
    Computes: MSE = (1/n) * Σ(y_pred - y_data)²
    
    This is the most common error function for continuous regression problems.
    It penalizes large errors more heavily than small ones due to the squaring.
    
    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([1.0, 2.0, 3.0, 4.0])
    >>> mse_func = MSE(data)
    >>> prediction = np.array([1.1, 2.2, 2.8, 4.1])
    >>> error = mse_func(prediction)
    >>> print(f"MSE: {error:.4f}")
    """
    
    def __call__(self, y_pred: np.ndarray) -> float:
        """Compute Mean Squared Error."""
        y_pred = self._validate_prediction(y_pred)
        if not np.isscalar(y_pred):
            residuals = y_pred - self.data
            return float(np.mean(residuals**2))
        return y_pred


class ChiSquaredMSE(ErrorFunction):
    """
    Chi-squared weighted Mean Squared Error.
    
    Computes: χ² = Σ((y_pred - y_data)² / σ²)
    
    This error function weights residuals by their expected variance (σ²),
    making it appropriate when different data points have different uncertainties.
    
    Parameters
    ----------
    data : np.ndarray
        Reference/observed data.
    sigma : np.ndarray or float
        Standard deviation/uncertainty for each data point. If float,
        assumes constant uncertainty across all points.
    normalize : bool, optional
        If True, normalize by number of data points (default False).
        
    Examples
    --------
    >>> data = np.array([1.0, 2.0, 3.0, 4.0])
    >>> sigma = np.array([0.1, 0.2, 0.1, 0.3])  # Different uncertainties
    >>> chi2_func = ChiSquaredMSE(data, sigma=sigma)
    >>> prediction = np.array([1.1, 2.2, 2.8, 4.1])
    >>> error = chi2_func(prediction)
    >>> print(f"Chi-squared: {error:.4f}")
    
    >>> # Constant uncertainty
    >>> chi2_func_const = ChiSquaredMSE(data, sigma=0.2)
    """
    
    def __init__(self, data: np.ndarray, sigma: Union[np.ndarray, float], 
                 normalize: bool = False):
        super().__init__(data)
        
        if np.isscalar(sigma):
            self.sigma = np.full_like(self.data, float(sigma))
        else:
            self.sigma = np.asarray(sigma)
            
        if self.sigma.shape != self.data.shape:
            raise ValueError(f"Sigma shape {self.sigma.shape} does not match "
                           f"data shape {self.data.shape}")
        
        if (self.sigma <= 0).any():
            raise ValueError("All sigma values must be positive")
            
        self.normalize = normalize
    
    def __call__(self, y_pred: np.ndarray) -> float:
        """Compute Chi-squared weighted error."""
        y_pred = self._validate_prediction(y_pred)
        if not np.isscalar(y_pred):
            residuals = y_pred - self.data
            chi_squared = np.sum((residuals / self.sigma)**2)
            
            if self.normalize:
                chi_squared /= len(self.data)
                
            return float(chi_squared)
        return y_pred




class LogLikelihood(ErrorFunction):
    """
    Gaussian Log-Likelihood Error Function.

    Computes the log-likelihood of the predicted values `y_pred` under the
    assumption that the observed data `data` follows a Gaussian distribution 
    with mean equal to `y_pred` and constant variance `sigma^2`.

    The log-likelihood is given by:
        LL(μ, σ²) = -n/2 * log(2πσ²) - 1/(2σ²) * Σ(yi - μ)²

    Parameters
    ----------
    data : np.ndarray
        Observed data points.
    sigma : float
        Standard deviation of the Gaussian noise. Must be positive.

    Raises
    ------
    ValueError
        If sigma is not positive or if the data array is empty.
    """

    def __init__(self, data: np.ndarray, sigma: float, **kwargs):
        super().__init__(data, **kwargs)
        if sigma <= 0:
            raise ValueError("Standard deviation sigma must be positive")
        self.sigma = sigma
        self.n = self.data.size

    def __call__(self, y_pred: np.ndarray) -> float:
        y_pred = self._validate_prediction(y_pred)
        if not np.isfinite(y_pred).all():
            return -np.inf  # Return log-likelihood as -inf if prediction is invalid

        residuals = self.data - y_pred
        squared_error = np.sum(residuals**2)
        ll = -0.5 * self.n * np.log(2 * np.pi * self.sigma**2) - (0.5 / self.sigma**2) * squared_error
        return ll
    

class MAE(ErrorFunction):
    """
    Mean Absolute Error (MAE) function.
    
    Computes: MAE = (1/n) * Σ|y_pred - y_data|
    
    MAE is more robust to outliers than MSE since it doesn't square the residuals.
    It provides a linear penalty for errors.
    
    Examples
    --------
    >>> data = np.array([1.0, 2.0, 3.0, 4.0])
    >>> mae_func = MAE(data)
    >>> prediction = np.array([1.1, 2.2, 2.8, 4.1])
    >>> error = mae_func(prediction)
    >>> print(f"MAE: {error:.4f}")
    """
    
    def __call__(self, y_pred: np.ndarray) -> float:
        """Compute Mean Absolute Error."""
        y_pred = self._validate_prediction(y_pred)
        if not np.isscalar(y_pred):
            residuals = np.abs(y_pred - self.data)
            return float(np.mean(residuals))
        return y_pred


class RMSE(ErrorFunction):
    """
    Root Mean Squared Error (RMSE) function.
    
    Computes: RMSE = √((1/n) * Σ(y_pred - y_data)²)
    
    RMSE is in the same units as the original data, making it more interpretable
    than MSE while maintaining the same optimization properties.
    
    Examples
    --------
    >>> data = np.array([1.0, 2.0, 3.0, 4.0])
    >>> rmse_func = RMSE(data)
    >>> prediction = np.array([1.1, 2.2, 2.8, 4.1])
    >>> error = rmse_func(prediction)
    >>> print(f"RMSE: {error:.4f}")
    """
    
    def __call__(self, y_pred: np.ndarray) -> float:
        """Compute Root Mean Squared Error."""
        y_pred = self._validate_prediction(y_pred)
        if not np.isscalar(y_pred):
            residuals = y_pred - self.data
            mse = np.mean(residuals**2)
            return float(np.sqrt(mse))
        return y_pred


class HuberLoss(ErrorFunction):
    """
    Huber Loss function (robust regression).
    
    Combines the best properties of MSE and MAE:
    - Quadratic for small errors (|error| <= delta)
    - Linear for large errors (|error| > delta)
    
    This makes it less sensitive to outliers than MSE while maintaining
    smoothness for optimization.
    
    Parameters
    ----------
    data : np.ndarray
        Reference/observed data.
    delta : float, optional
        Threshold for switching between quadratic and linear loss. Default is 1.0.
        
    Examples
    --------
    >>> data = np.array([1.0, 2.0, 3.0, 4.0])
    >>> huber_func = HuberLoss(data, delta=0.5)
    >>> prediction = np.array([1.1, 2.2, 2.8, 4.1])
    >>> error = huber_func(prediction)
    >>> print(f"Huber Loss: {error:.4f}")
    """
    
    def __init__(self, data: np.ndarray, delta: float = 1.0):
        super().__init__(data)
        if delta <= 0:
            raise ValueError("Delta must be positive")
        self.delta = delta
    
    def __call__(self, y_pred: np.ndarray) -> float:
        """Compute Huber Loss."""
        y_pred = self._validate_prediction(y_pred)
        if not np.isscalar(y_pred):
            residuals = np.abs(y_pred - self.data)
            
            # Quadratic for small errors, linear for large errors
            quadratic_mask = residuals <= self.delta
            quadratic_loss = 0.5 * residuals[quadratic_mask]**2
            linear_loss = self.delta * (residuals[~quadratic_mask] - 0.5 * self.delta)
            
            total_loss = np.sum(quadratic_loss) + np.sum(linear_loss)
            return float(total_loss / len(self.data))
        return y_pred


class RegularizedError(ErrorFunction):
    """
    Error function with L1, L2, or elastic net regularization.
    
    Combines a base error function with parameter regularization:
    Total Error = Base Error + λ₁ * L1_penalty + λ₂ * L2_penalty
    
    This is useful for preventing overfitting and promoting sparse solutions.
    
    Parameters
    ----------
    data : np.ndarray
        Reference/observed data.
    base_error : str or ErrorFunction
        Base error function ('mse', 'mae', 'rmse') or custom ErrorFunction instance.
    l1_lambda : float, optional
        L1 regularization strength (promotes sparsity). Default is 0.0.
    l2_lambda : float, optional  
        L2 regularization strength (promotes smoothness). Default is 0.0.
    param_getter : callable, optional
        Function to extract parameters for regularization. If None,
        regularization is not applied (requires external parameter passing).
        
    Examples
    --------
    >>> data = np.array([1.0, 2.0, 3.0, 4.0])
    >>> reg_func = RegularizedError(data, 'mse', l1_lambda=0.01, l2_lambda=0.1)
    >>> 
    >>> # Usage in optimization (parameters passed externally)
    >>> def error_with_params(y_pred, params):
    ...     base_error = reg_func(y_pred)
    ...     l1_penalty = np.sum(np.abs(list(params.values())))
    ...     l2_penalty = np.sum([p**2 for p in params.values()])
    ...     return base_error + 0.01 * l1_penalty + 0.1 * l2_penalty
    """
    
    def __init__(self, data: np.ndarray, base_error: Union[str, ErrorFunction] = 'mse',
                 l1_lambda: float = 0.0, l2_lambda: float = 0.0,
                 param_getter: Optional[Callable] = None):
        super().__init__(data)
        
        # Initialize base error function
        if isinstance(base_error, str):
            error_map = {
                'mse': MSE(data),
                'mae': MAE(data), 
                'rmse': RMSE(data)
            }
            if base_error not in error_map:
                raise ValueError(f"Unknown base error: {base_error}")
            self.base_error = error_map[base_error]
        else:
            self.base_error = base_error
            
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.param_getter = param_getter
        
        if l1_lambda < 0 or l2_lambda < 0:
            raise ValueError("Regularization parameters must be non-negative")
    
    def __call__(self, y_pred: np.ndarray, params: Optional[dict] = None) -> float:
        """
        Compute regularized error.
        
        Parameters
        ----------
        y_pred : np.ndarray
            Model predictions.
        params : dict, optional
            Parameter dictionary for regularization. If None, only base error is computed.
            
        Returns
        -------
        float
            Total error including regularization terms.
        """
        base_error_val = self.base_error(y_pred)
        
        if params is None:
            if self.l1_lambda > 0 or self.l2_lambda > 0:
                warnings.warn("Regularization requested but no parameters provided")
            return base_error_val
        
        # Compute regularization terms
        param_values = np.array(list(params.values()))
        
        l1_penalty = 0.0
        l2_penalty = 0.0
        
        if self.l1_lambda > 0:
            l1_penalty = self.l1_lambda * np.sum(np.abs(param_values))
            
        if self.l2_lambda > 0:
            l2_penalty = self.l2_lambda * np.sum(param_values**2)
        
        return float(base_error_val + l1_penalty + l2_penalty)


class WeightedError(ErrorFunction):
    """
    Weighted error function for handling different importance of data points.
    
    Applies weights to individual data points, allowing some measurements
    to contribute more to the total error than others.
    
    Parameters
    ----------
    data : np.ndarray
        Reference/observed data.
    weights : np.ndarray
        Weights for each data point. Higher weights = more importance.
    base_error : str, optional
        Base error type ('mse', 'mae'). Default is 'mse'.
        
    Examples
    --------
    >>> data = np.array([1.0, 2.0, 3.0, 4.0])
    >>> weights = np.array([1.0, 2.0, 1.0, 0.5])  # Different importance
    >>> weighted_func = WeightedError(data, weights, 'mse')
    >>> prediction = np.array([1.1, 2.2, 2.8, 4.1])
    >>> error = weighted_func(prediction)
    >>> print(f"Weighted MSE: {error:.4f}")
    """
    
    def __init__(self, data: np.ndarray, weights: np.ndarray, base_error: str = 'mse'):
        super().__init__(data)
        
        self.weights = np.asarray(weights)
        if self.weights.shape != self.data.shape:
            raise ValueError(f"Weights shape {self.weights.shape} does not match "
                           f"data shape {self.data.shape}")
        
        if (self.weights < 0).any():
            raise ValueError("All weights must be non-negative")
            
        # Normalize weights to sum to number of data points
        weight_sum = np.sum(self.weights)
        if weight_sum > 0:
            self.weights = self.weights * len(self.data) / weight_sum
        else:
            raise ValueError("Sum of weights must be positive")
            
        self.base_error = base_error
    
    def __call__(self, y_pred: np.ndarray) -> float:
        """Compute weighted error."""
        y_pred = self._validate_prediction(y_pred)
        if not np.isscalar(y_pred):
            if self.base_error == 'mse':
                residuals = (y_pred - self.data)**2
            elif self.base_error == 'mae':
                residuals = np.abs(y_pred - self.data)
            else:
                raise ValueError(f"Unknown base error: {self.base_error}")
                
            weighted_error = np.sum(self.weights * residuals) / len(self.data)
            return float(weighted_error)
        return y_pred


# Convenience functions for backward compatibility and ease of use
def mse(data: np.ndarray) -> MSE:
    """
    Will be deprecated in future versions.
    
    Create MSE error function.
    
    Parameters
    ----------  
    data : np.ndarray
        Reference data for comparison.
        
    Returns
    -------
    MSE
        Configured MSE error function.
        
    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([1.0, 2.0, 3.0])
    >>> error_func = mse(data)
    >>> prediction = np.array([1.1, 2.1, 2.9])
    >>> error = error_func(prediction)
    """
    return MSE(data)


def chisquared(data: np.ndarray, sigma: Union[np.ndarray, float], 
               normalize: bool = False) -> ChiSquaredMSE:
    """
    Will be deprecated in future versions.
    Create Chi-squared error function.
    
    Parameters
    ----------
    data : np.ndarray
        Reference data.
    sigma : np.ndarray or float
        Standard deviation for each data point.
    normalize : bool, optional
        Whether to normalize by number of points.
        
    Returns
    -------
    ChiSquaredMSE
        Configured Chi-squared error function.
    """
    return ChiSquaredMSE(data, sigma=sigma, normalize=normalize)


def mae(data: np.ndarray) -> MAE:
    """
    Will be deprecated in future versions.
     Create MAE error function."""
    return MAE(data)


def rmse(data: np.ndarray) -> RMSE:
    """
    Will be deprecated in future versions.Create RMSE error function.""" 
    return RMSE(data)


def huber(data: np.ndarray, delta: float = 1.0) -> HuberLoss:
    """
    Will be deprecated in future versions.Create Huber loss error function."""
    return HuberLoss(data, delta=delta)