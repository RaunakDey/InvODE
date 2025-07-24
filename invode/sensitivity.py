import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


class ODESensitivity:
    """
    A class for performing sensitivity analysis on ODE model parameters.
    
    This class provides methods to analyze how sensitive the model output is to
    changes in different parameters, using data from optimization history or
    direct parameter sampling.
    """
    
    def __init__(self, ode_func, error_func):
        """
        Initialize the ODESensitivity analyzer.
        
        Parameters
        ----------
        ode_func : callable
            The ODE solver function that takes a parameter dictionary and returns
            model output.
        error_func : callable
            Error/objective function that quantifies model fit quality.
        """
        self.ode_func = ode_func
        self.error_func = error_func
        
    def analyze_parameter_sensitivity(
        self, 
        candidates_df: pd.DataFrame,
        method: str = 'correlation',
        normalize: bool = True,
        min_samples: int = 4,
        exclude_columns: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Analyze parameter sensitivity using optimization candidates data.
        
        This method examines how changes in each parameter correlate with changes
        in the error function, providing insights into which parameters have the
        strongest influence on model performance.
        
        Parameters
        ----------
        candidates_df : pd.DataFrame
            DataFrame containing optimization candidates with parameter values and errors.
            Expected to have columns: 'iteration', 'rank', 'error', and parameter columns.
            This is typically obtained from ODEOptimizer.get_top_candidates_table().
            
        method : str, optional
            Method for calculating sensitivity. Options:
            
            - 'correlation': Pearson correlation between parameter values and errors
            - 'variance': Normalized variance of error with respect to parameter changes
            - 'gradient': Approximate gradient of error with respect to parameters
            - 'mutual_info': Mutual information between parameters and error
            - 'rank_correlation': Spearman rank correlation (robust to outliers)
            
            Default is 'correlation'.
            
        normalize : bool, optional
            If True, normalize sensitivity values to [0, 1] range for comparison.
            Default is True.
            
        min_samples : int, optional
            Minimum number of samples required for reliable sensitivity analysis.
            If fewer samples are available, a warning is issued. Default is 10.
            
        exclude_columns : List[str], optional
            List of column names to exclude from sensitivity analysis.
            By default, excludes ['iteration', 'rank', 'error'].
            
        Returns
        -------
        Dict[str, float]
            Dictionary mapping parameter names to their sensitivity values.
            Higher absolute values indicate greater sensitivity. For correlation
            methods, negative values indicate inverse relationships.
            
        Raises
        ------
        ValueError
            If candidates_df is empty, missing required columns, or contains insufficient data.
        TypeError
            If candidates_df is not a pandas DataFrame.
            
        Notes
        -----
        **Sensitivity Interpretation:**
        
        - **High sensitivity**: Small parameter changes cause large error changes
        - **Low sensitivity**: Parameter changes have minimal impact on error
        - **Negative correlation**: Increasing parameter decreases error
        - **Positive correlation**: Increasing parameter increases error
        
        **Method Details:**
        
        - **Correlation**: Measures linear relationship between parameter and error
        - **Rank Correlation**: Spearman correlation, robust to non-linear monotonic relationships
        - **Variance**: Quantifies error variability attributable to parameter
        - **Gradient**: Estimates local derivative of error w.r.t. parameter
        - **Mutual Info**: Captures non-linear parameter-error relationships
        
        The analysis uses all candidates from all iterations, providing a global
        view of parameter sensitivity across the optimization landscape.
        
        Examples
        --------
        Basic sensitivity analysis from optimizer results:
        
        >>> # After running optimization
        >>> optimizer = ODEOptimizer(...)
        >>> optimizer.fit()
        >>> 
        >>> # Get candidates table and analyze sensitivity
        >>> df = optimizer.get_top_candidates_table()
        >>> sensitivity = ODESensitivity(optimizer.ode_func, optimizer.error_func)
        >>> sensitivities = sensitivity.analyze_parameter_sensitivity(df)
        >>> 
        >>> # Display results sorted by sensitivity magnitude
        >>> for param, sens in sorted(sensitivities.items(), key=lambda x: abs(x[1]), reverse=True):
        ...     print(f"{param}: {sens:.4f}")
        alpha: -0.8234  # Highly sensitive, negative correlation
        beta: 0.6891    # Highly sensitive, positive correlation
        gamma: -0.3456  # Moderately sensitive
        delta: 0.1234   # Low sensitivity
        
        Compare different sensitivity methods:
        
        >>> methods = ['correlation', 'rank_correlation', 'variance', 'mutual_info']
        >>> results = {}
        >>> for method in methods:
        ...     sens = sensitivity.analyze_parameter_sensitivity(df, method=method)
        ...     results[method] = sens
        >>> 
        >>> # Create comparison DataFrame
        >>> comparison_df = pd.DataFrame(results)
        >>> print(comparison_df.round(4))
                  correlation  rank_correlation  variance  mutual_info
        alpha          -0.823            -0.801     0.745        0.234
        beta            0.689             0.712     0.523        0.189
        gamma          -0.346            -0.298     0.187        0.098
        delta           0.123             0.145     0.076        0.043
        
        Analyze sensitivity for specific iterations:
        
        >>> # Focus on later iterations (better convergence)
        >>> late_iterations = df[df['iteration'] >= 5]
        >>> late_sensitivities = sensitivity.analyze_parameter_sensitivity(late_iterations)
        >>> 
        >>> # Compare early vs late sensitivity
        >>> early_iterations = df[df['iteration'] <= 3]
        >>> early_sensitivities = sensitivity.analyze_parameter_sensitivity(early_iterations)
        
        Filter by rank to focus on best candidates:
        
        >>> # Only analyze top candidates from each iteration
        >>> top_candidates = df[df['rank'] == 1]
        >>> top_sensitivities = sensitivity.analyze_parameter_sensitivity(top_candidates)
        
        Custom column exclusions:
        
        >>> # Exclude additional metadata columns
        >>> sensitivities = sensitivity.analyze_parameter_sensitivity(
        ...     df, exclude_columns=['iteration', 'rank', 'error', 'timestamp']
        ... )
        """
        # Validate input
        if not isinstance(candidates_df, pd.DataFrame):
            raise TypeError("candidates_df must be a pandas DataFrame")
            
        if candidates_df.empty:
            raise ValueError("candidates_df is empty")
            
        # Check for required columns
        required_columns = ['error']
        missing_columns = [col for col in required_columns if col not in candidates_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Set default excluded columns
        if exclude_columns is None:
            exclude_columns = ['iteration', 'rank', 'error']
        else:
            # Ensure 'error' is not excluded (we need it for analysis)
            exclude_columns = [col for col in exclude_columns if col != 'error']
            
        # Get parameter columns
        param_columns = [col for col in candidates_df.columns if col not in exclude_columns]
        
        if not param_columns:
            raise ValueError("No parameter columns found after excluding specified columns")
        
        # Check minimum samples
        n_samples = len(candidates_df)
        if n_samples < min_samples:
            print(f"Warning: Only {n_samples} samples available. "
                  f"Results may be unreliable (recommended minimum: {min_samples})")
        
        # Calculate sensitivity for each parameter
        sensitivities = {}
        
        for param in param_columns:
            param_values = candidates_df[param].values
            errors = candidates_df['error'].values
            
            # Skip parameters with no variation
            if param_values.std() == 0:
                print(f"Warning: Parameter '{param}' has no variation, setting sensitivity to 0")
                sensitivities[param] = 0.0
                continue
            
            if method == 'correlation':
                # Pearson correlation coefficient
                corr, p_value = stats.pearsonr(param_values, errors)
                sensitivities[param] = corr if not np.isnan(corr) else 0.0
                
            elif method == 'rank_correlation':
                # Spearman rank correlation (robust to outliers)
                corr, p_value = stats.spearmanr(param_values, errors)
                sensitivities[param] = corr if not np.isnan(corr) else 0.0
                
            elif method == 'variance':
                # Variance-based sensitivity using binning approach
                try:
                    # Bin parameters and calculate error variance within bins
                    n_bins = min(10, n_samples // 3)
                    if n_bins < 2:
                        n_bins = 2
                    
                    param_bins = pd.cut(param_values, bins=n_bins, duplicates='drop')
                    df_temp = pd.DataFrame({'param_bins': param_bins, 'error': errors})
                    
                    bin_variances = df_temp.groupby('param_bins')['error'].var()
                    mean_bin_variance = bin_variances.mean()
                    total_variance = errors.var()
                    
                    # Sensitivity as fraction of total variance explained
                    if total_variance > 0:
                        sensitivity = 1 - (mean_bin_variance / total_variance)
                        sensitivities[param] = max(0, sensitivity)  # Ensure non-negative
                    else:
                        sensitivities[param] = 0.0
                        
                except Exception as e:
                    print(f"Warning: Variance calculation failed for '{param}': {e}")
                    sensitivities[param] = 0.0
                    
            elif method == 'gradient':
                # Approximate gradient using finite differences
                try:
                    # Sort by parameter value
                    sorted_indices = np.argsort(param_values)
                    sorted_params = param_values[sorted_indices]
                    sorted_errors = errors[sorted_indices]
                    
                    # Calculate approximate derivatives
                    if len(sorted_params) >= 2:
                        gradients = np.gradient(sorted_errors, sorted_params)
                        mean_gradient = np.mean(np.abs(gradients))
                        sensitivities[param] = mean_gradient
                    else:
                        sensitivities[param] = 0.0
                        
                except Exception as e:
                    print(f"Warning: Gradient calculation failed for '{param}': {e}")
                    sensitivities[param] = 0.0
                    
            elif method == 'mutual_info':
                # Mutual information between parameter and error
                try:
                    from sklearn.feature_selection import mutual_info_regression
                    
                    param_reshaped = param_values.reshape(-1, 1)
                    mi_score = mutual_info_regression(param_reshaped, errors, random_state=42)[0]
                    sensitivities[param] = mi_score
                    
                except ImportError:
                    raise ImportError("sklearn is required for mutual_info method")
                except Exception as e:
                    print(f"Warning: Mutual info calculation failed for '{param}': {e}")
                    sensitivities[param] = 0.0
                    
            else:
                raise ValueError(f"Unknown sensitivity method: '{method}'. "
                               f"Available methods: correlation, rank_correlation, variance, gradient, mutual_info")
        
        # Normalize if requested
        if normalize and sensitivities:
            sens_values = list(sensitivities.values())
            
            if method in ['correlation', 'rank_correlation']:
                # For correlation methods, preserve sign but normalize magnitude
                max_abs_sens = max(abs(s) for s in sens_values)
                if max_abs_sens > 0:
                    sensitivities = {k: v / max_abs_sens for k, v in sensitivities.items()}
            else:
                # For other methods, normalize to [0, 1]
                min_sens = min(sens_values)
                max_sens = max(sens_values)
                if max_sens > min_sens:
                    sensitivities = {
                        k: (v - min_sens) / (max_sens - min_sens) 
                        for k, v in sensitivities.items()
                    }
        return sensitivities
        
    def analyze_sensitivity_by_iteration(
        self,
        candidates_df: pd.DataFrame,
        method: str = 'correlation',
        normalize: bool = True
    ) -> pd.DataFrame:
        """
        Analyze how parameter sensitivity changes across optimization iterations.
        
        This method provides insights into how the importance of different parameters
        evolves as the optimization progresses, which can reveal whether certain
        parameters become more or less critical in later stages.
        
        Parameters
        ----------
        candidates_df : pd.DataFrame
            DataFrame containing optimization candidates from get_top_candidates_table().
        method : str, optional
            Sensitivity analysis method. Default is 'correlation'.
        normalize : bool, optional
            Whether to normalize sensitivity values. Default is True.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with iterations as rows and parameters as columns,
            containing sensitivity values for each iteration.
            
        Examples
        --------
        >>> df = optimizer.get_top_candidates_table()
        >>> sensitivity = ODESensitivity(optimizer.ode_func, optimizer.error_func)
        >>> iteration_sens = sensitivity.analyze_sensitivity_by_iteration(df)
        >>> print(iteration_sens)
        
        >>> # Plot evolution of parameter sensitivity
        >>> import matplotlib.pyplot as plt
        >>> plt.figure(figsize=(12, 6))
        >>> for param in iteration_sens.columns:
        ...     plt.plot(iteration_sens.index, iteration_sens[param], 
        ...              marker='o', label=param)
        >>> plt.xlabel('Iteration')
        >>> plt.ylabel('Parameter Sensitivity')
        >>> plt.title('Evolution of Parameter Sensitivity')
        >>> plt.legend()
        >>> plt.grid(True, alpha=0.3)
        >>> plt.show()
        """
        if 'iteration' not in candidates_df.columns:
            raise ValueError("DataFrame must contain 'iteration' column")
            
        iterations = sorted(candidates_df['iteration'].unique())
        param_columns = [col for col in candidates_df.columns 
                        if col not in ['iteration', 'rank', 'error']]
        
        sensitivity_by_iter = {}
        
        for iteration in iterations:
            iter_data = candidates_df[candidates_df['iteration'] == iteration]
            if len(iter_data) >= 2:  # Need at least 2 samples for correlation
                try:
                    iter_sensitivities = self.analyze_parameter_sensitivity(
                        iter_data, method=method, normalize=normalize, min_samples=2
                    )
                    sensitivity_by_iter[iteration] = iter_sensitivities
                except Exception as e:
                    print(f"Warning: Sensitivity analysis failed for iteration {iteration}: {e}")
                    # Fill with zeros
                    sensitivity_by_iter[iteration] = {param: 0.0 for param in param_columns}
            else:
                # Not enough data for this iteration
                sensitivity_by_iter[iteration] = {param: np.nan for param in param_columns}
        
        return pd.DataFrame(sensitivity_by_iter).T
    
    def analyze_sensitivity_by_rank(
        self,
        candidates_df: pd.DataFrame,
        method: str = 'correlation',
        normalize: bool = True
    ) -> pd.DataFrame:
        """
        Analyze parameter sensitivity for different candidate ranks.
        
        This method examines whether parameter sensitivity differs between
        the best candidates (rank 1) versus lower-ranked candidates, which
        can provide insights into parameter importance in high-performance regions.
        
        Parameters
        ----------
        candidates_df : pd.DataFrame
            DataFrame containing optimization candidates from get_top_candidates_table().
        method : str, optional
            Sensitivity analysis method. Default is 'correlation'.
        normalize : bool, optional
            Whether to normalize sensitivity values. Default is True.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with ranks as rows and parameters as columns,
            containing sensitivity values for each rank.
            
        Examples
        --------
        >>> df = optimizer.get_top_candidates_table()
        >>> sensitivity = ODESensitivity(optimizer.ode_func, optimizer.error_func)
        >>> rank_sens = sensitivity.analyze_sensitivity_by_rank(df)
        >>> print(rank_sens)
        
        >>> # Compare sensitivity between best and worst candidates
        >>> print("Best candidates (rank 1):")
        >>> print(rank_sens.loc[1])
        >>> print("\nWorst candidates (rank 3):")
        >>> print(rank_sens.loc[3])
        """
        if 'rank' not in candidates_df.columns:
            raise ValueError("DataFrame must contain 'rank' column")
            
        ranks = sorted(candidates_df['rank'].unique())
        param_columns = [col for col in candidates_df.columns 
                        if col not in ['iteration', 'rank', 'error']]
        
        sensitivity_by_rank = {}
        
        for rank in ranks:
            rank_data = candidates_df[candidates_df['rank'] == rank]
            if len(rank_data) >= 2:  # Need at least 2 samples
                try:
                    rank_sensitivities = self.analyze_parameter_sensitivity(
                        rank_data, method=method, normalize=normalize, min_samples=2
                    )
                    sensitivity_by_rank[rank] = rank_sensitivities
                except Exception as e:
                    print(f"Warning: Sensitivity analysis failed for rank {rank}: {e}")
                    sensitivity_by_rank[rank] = {param: 0.0 for param in param_columns}
            else:
                sensitivity_by_rank[rank] = {param: np.nan for param in param_columns}
        
        return pd.DataFrame(sensitivity_by_rank).T

    def create_sensitivity_summary(
        self,
        candidates_df: pd.DataFrame,
        methods: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Create a comprehensive summary of parameter sensitivities using multiple methods.
        
        Parameters
        ----------
        candidates_df : pd.DataFrame
            DataFrame containing optimization candidates from get_top_candidates_table().
        methods : List[str], optional
            List of sensitivity methods to compare. If None, uses all available methods.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with parameters as rows and different sensitivity methods as columns.
            
        Examples
        --------
        >>> df = optimizer.get_top_candidates_table()
        >>> sensitivity = ODESensitivity(optimizer.ode_func, optimizer.error_func)
        >>> summary = sensitivity.create_sensitivity_summary(df)
        >>> print(summary.round(4))
        
        >>> # Identify most consistently sensitive parameters
        >>> summary['mean_abs_sensitivity'] = summary.abs().mean(axis=1)
        >>> print(summary.sort_values('mean_abs_sensitivity', ascending=False))
        """
        if methods is None:
            methods = ['correlation', 'rank_correlation', 'variance', 'mutual_info']
        
        results = {}
        for method in methods:
            try:
                sensitivities = self.analyze_parameter_sensitivity(
                    candidates_df, method=method, normalize=True
                )
                results[method] = sensitivities
            except Exception as e:
                print(f"Warning: Method '{method}' failed: {e}")
                # Get parameter names from other successful methods or DataFrame
                param_names = [col for col in candidates_df.columns 
                             if col not in ['iteration', 'rank', 'error']]
                results[method] = {param: np.nan for param in param_names}
        
        return pd.DataFrame(results)
    
    def plot_sensitivity_analysis(
        self,
        sensitivities: Dict[str, float],
        title: str = "Parameter Sensitivity Analysis",
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a visualization of parameter sensitivities.
        
        Parameters
        ----------
        sensitivities : Dict[str, float]
            Dictionary of parameter sensitivities from analyze_parameter_sensitivity_from_history
        title : str, optional
            Plot title. Default is "Parameter Sensitivity Analysis".
        figsize : Tuple[int, int], optional
            Figure size as (width, height). Default is (10, 6).
        save_path : str, optional
            If provided, save the plot to this path.
            
        Returns
        -------
        plt.Figure
            The matplotlib figure object.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Sort parameters by sensitivity magnitude
        sorted_items = sorted(sensitivities.items(), key=lambda x: abs(x[1]), reverse=True)
        params = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]
        
        # Color bars based on sign (for correlation-based methods)
        colors = ['red' if v < 0 else 'blue' for v in values]
        
        # Create bar plot
        bars = ax.bar(range(len(params)), [abs(v) for v in values], 
                     color=colors, alpha=0.7, edgecolor='black', linewidth=1)
        
        # Customize plot
        ax.set_xlabel('Parameters', fontsize=12)
        ax.set_ylabel('Sensitivity Magnitude', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(params)))
        ax.set_xticklabels(params, rotation=45, ha='right')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Add legend for colors (if applicable)
        if any(v < 0 for v in values):
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='blue', alpha=0.7, label='Positive correlation'),
                Patch(facecolor='red', alpha=0.7, label='Negative correlation')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig