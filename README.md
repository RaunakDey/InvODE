
# 🚀 InvODE

**Intelligent Variational Optimization for Differential Equations**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Documentation Status](https://readthedocs.org/projects/invode/badge/?version=latest)](https://invode.readthedocs.io/en/latest/)
[![PyPI version](https://badge.fury.io/py/invode.svg)](https://badge.fury.io/py/invode)

A lightweight, powerful and intuitive Python library for parameter optimization in ordinary differential equation (ODE) systems. invode combines global optimization with local refinement techniques to efficiently find optimal parameter sets for complex dynamical systems.

---

## ✨ Features

### 🎯 **Hybrid Optimization Strategy**
- **Global Exploration**: Latin Hypercube Sampling for comprehensive parameter space coverage
- **Local Refinement**: Gradient-based optimization for precise convergence
- **Adaptive Sampling**: Progressive search region shrinking for efficient exploration

### 🔧 **Flexible Error Functions**
- Built-in metrics: MSE, MAE, RMSE, Chi-squared, Huber Loss
- **Regularization**: L1, L2, and Elastic Net penalties
- **Weighted Fitting**: Handle heteroscedastic data with confidence
- **Custom Metrics**: Easy integration of domain-specific error functions

### 📊 **Advanced Analysis Tools**
- **Sensitivity Analysis**: Identify critical parameters affecting model behavior
- **Optimization History**: Track convergence and parameter evolution
- **Visualization**: Built-in plotting for optimization progress and sensitivity

### ⚡ **High Performance**
- **Parallel Processing**: Multi-core evaluation of parameter candidates (to be done!)
- **Efficient Sampling**: Quasi-Monte Carlo methods for better space filling
- **Memory Optimized**: Handle large parameter spaces without memory overflow  (to be done!)

### 🛠️ **Developer Friendly**
- **Clean API**: Intuitive interface following scikit-learn conventions
- **Extensible**: Easy to add custom optimization methods and error functions 
- **Well Documented**: Comprehensive documentation with examples and tutorials

---

## 🚀 Quick Start

### Installation (to be done)

```bash
pip install invode
```

### Minimal Example

```python
import numpy as np
from scipy.integrate import odeint
import invode
from invode import erf

# Define your ODE system
def lotka_volterra(y, t, alpha, beta, delta, gamma):
    """Classic predator-prey model"""
    prey, predator = y
    dydt = [
        alpha * prey - beta * prey * predator,
        delta * prey * predator - gamma * predator
    ]
    return dydt

# Create ODE solver function
def ode_solver(params):
    """Solve ODE with given parameters"""
    t = np.linspace(0, 10, 100)
    y0 = [10, 5]  # Initial conditions
    solution = odeint(
        lotka_volterra, y0, t,
        args=(params['alpha'], params['beta'], params['delta'], params['gamma'])
    )
    return solution.flatten()  # Return flattened array

# Generate synthetic noisy data (in practice, use your experimental data)
true_params = {'alpha': 1.0, 'beta': 0.1, 'delta': 0.075, 'gamma': 1.5}
true_solution = ode_solver(true_params)
np.random.seed(42)
noisy_data = true_solution + np.random.normal(0, 0.5, len(true_solution))

# Set up optimization
optimizer = invode.ODEOptimizer(
    ode_func=ode_solver,
    error_func=erf.mse(noisy_data),  # Mean squared error
    param_bounds={
        'alpha': (0.1, 2.0),
        'beta': (0.01, 0.5), 
        'delta': (0.01, 0.3),
        'gamma': (0.5, 3.0)
    },
    varying_params=['alpha', 'beta', 'delta', 'gamma'],
    num_iter=20,
    n_samples=50,
    do_local_opt=True,
    parallel=True,
    verbose=True
)

# Run optimization
best_params, best_error = optimizer.fit()

print(f"Optimal parameters: {best_params}")
print(f"Final error: {best_error:.6f}")

# Analyze parameter sensitivity
sensitivity = invode.ODESensitivity(ode_solver, erf.mse(noisy_data))
candidates_df = optimizer.get_top_candidates_table()
sensitivities = sensitivity.analyze_parameter_sensitivity(candidates_df)

print("\nParameter Sensitivity:")
for param, sens in sorted(sensitivities.items(), key=lambda x: abs(x[1]), reverse=True):
    print(f"  {param}: {sens:.4f}")
```

### Expected Output
```
Iteration 1/20: Best error = 42.851
Iteration 2/20: Best error = 15.234  
...
Iteration 20/20: Best error = 2.891

[Local Optimization]
Refined parameters: {'alpha': 0.987, 'beta': 0.103, 'delta': 0.076, 'gamma': 1.485}
Refined error: 2.456

Optimal parameters: {'alpha': 0.987, 'beta': 0.103, 'delta': 0.076, 'gamma': 1.485}
Final error: 2.456123

Parameter Sensitivity:
  gamma: -0.8234  # Highly sensitive, negative correlation
  alpha: 0.7156   # Highly sensitive, positive correlation  
  beta: -0.4891   # Moderately sensitive
  delta: 0.2134   # Less sensitive
```

---

## 📊 Real-World Applications

### 🧬 **Systems Biology**
- Parameter estimation for biochemical reaction networks
- Pharmacokinetic/pharmacodynamic modeling
- Gene regulatory network analysis

### 🌡️ **Engineering Systems** 
- Control system parameter tuning
- Heat transfer and fluid dynamics modeling
- Chemical reactor optimization

### 🌍 **Environmental Modeling**
- Population dynamics and epidemiological models
- Climate system parameter estimation
- Ecosystem modeling and resource management

### 🔬 **Physics & Chemistry**
- Reaction kinetics parameter fitting
- Mechanical system identification
- Quantum system parameter estimation

---

## 📚 Documentation

### API Reference
- **[ODEOptimizer](https://invode.readthedocs.io/en/latest/api/optimizer.html)**: Main optimization class
- **[Error Functions](https://invode.readthedocs.io/en/latest/api/erf.html)**: Built-in and custom error metrics
- **[Sensitivity Analysis](https://invode.readthedocs.io/en/latest/api/sensitivity.html)**: Parameter importance analysis
- **[Utilities](https://invode.readthedocs.io/en/latest/api/utils.html)**: Helper functions and tools

### Tutorials & Examples
- **[Getting Started Guide](https://invode.readthedocs.io/en/latest/quickstart.html)**: Step-by-step introduction
- **[Advanced Optimization](https://invode.readthedocs.io/en/latest/tutorials/advanced.html)**: Complex parameter fitting scenarios  
- **[Custom Error Functions](https://invode.readthedocs.io/en/latest/tutorials/custom_errors.html)**: Creating domain-specific metrics
- **[Case Studies](https://invode.readthedocs.io/en/latest/case_studies/)**: Real-world applications and best practices

---

## 🛠️ Advanced Usage

### Custom Error Functions
```python
import invode.erf as erf

# Chi-squared for heteroscedastic data
sigma = np.array([0.1, 0.2, 0.15, 0.3])  # Different uncertainties
error_func = erf.chisquared(data, sigma=sigma)

# Regularized fitting to prevent overfitting
error_func = erf.RegularizedError(data, 'mse', l1_lambda=0.01, l2_lambda=0.1)

# Robust fitting with Huber loss
error_func = erf.huber(data, delta=1.5)
```

### Parallel Optimization
```python
optimizer = invode.ODEOptimizer(
    # ... other parameters
    parallel=True,          # Parallel candidate evaluation
    local_parallel=True,    # Parallel local refinement
    n_samples=100,          # More samples for better parallelization
)
```

### Sensitivity Analysis
```python
# Overall parameter sensitivity
sensitivity = invode.ODESensitivity(ode_func, error_func)
candidates_df = optimizer.get_top_candidates_table()

# Multiple analysis methods
summary = sensitivity.create_sensitivity_summary(
    candidates_df, 
    methods=['correlation', 'variance', 'mutual_info']
)

# Evolution over iterations
iteration_sens = sensitivity.analyze_sensitivity_by_iteration(candidates_df)

# Visualize results
fig = sensitivity.plot_sensitivity_analysis(sensitivities)
```

---

## 🤝 Contributing

We welcome contributions from the community! Whether you're fixing bugs, adding features, improving documentation, or sharing use cases, your help makes invode better for everyone.

### 🐛 **Bug Reports & Feature Requests**
- **Found a bug?** [Open an issue](https://github.com/RaunakDey/invode/issues) with:
  - Clear description of the problem
  - Minimal reproducible example
  - Python version and system information
  - Expected vs. actual behavior

- **Have an idea?** [Request a feature](https://github.com/RaunakDey/invode/issues) with:
  - Use case description
  - Proposed API or implementation approach
  - Examples of how it would be used

### 💻 **Code Contributions**

1. **Fork the repository** and create a feature branch:
   ```bash
   git checkout -b feature/amazing-feature
   ```

2. **Set up development environment**:
   ```bash
   pip install -e ".[dev]"  # Install in development mode
   pip install pytest black flake8 sphinx
   ```

3. **Follow coding standards**:
   - Code style: [Black](https://github.com/psf/black) formatter
   - Linting: [Flake8](https://flake8.pycqa.org/) compliance
   - Type hints: Encouraged for new code
   - Docstrings: [NumPy style](https://numpydoc.readthedocs.io/) for all public functions

4. **Write tests** for new functionality:
   ```bash
   pytest tests/test_your_feature.py -v
   ```

5. **Update documentation** if needed:
   ```bash
   cd docs
   make html
   ```

6. **Submit a pull request** with:
   - Clear description of changes
   - Link to related issues
   - Test results and coverage information

### 📖 **Documentation Contributions**
- Fix typos or improve clarity
- Add examples and tutorials
- Translate documentation
- Create video tutorials or blog posts

### 💡 **Other Ways to Help**
- ⭐ **Star the repository** to increase visibility
- 📢 **Share invode** with colleagues and on social media
- 📝 **Write blog posts** or tutorials about your use cases
- 🎤 **Present at conferences** or meetups
- 🧪 **Test beta releases** and provide feedback

### 🏆 **Recognition**
Contributors are recognized in our [CONTRIBUTORS.md](CONTRIBUTORS.md) file and release notes. Significant contributions may be acknowledged with co-authorship on relevant publications.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Why MIT?** We believe in open science and want invode to be as widely useful as possible. The MIT license allows both academic and commercial use while maintaining attribution.


---

## 🙏 Acknowledgments

- **SciPy Community**: For providing the foundational numerical computing tools
- **Contributors**: All the amazing people who have contributed code, documentation, and feedback
- **Users**: The researchers and engineers who trust invode with their important work
- **Academic Partners**: Universities and research institutions supporting open source development

---

<div align="center">

**Made with ❤️ for the scientific computing community**

[🏠 Website (tbd)](https://invode.org) • [📚 Docs (tbd)](https://invode.readthedocs.io/) • [🐛 Issues](https://github.com/RaunakDey/invode/issues) • [💬 Discussions](https://github.com/RaunakDey/invode/discussions)

</div>