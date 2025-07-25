Quickstart
==========

This guide shows how to get started with the `invode` package.

Installation
------------

You can install the package locally (until on PyPI) by cloning and using pip:

.. code-block:: bash

    git clone git@github.com:RaunakDey/InvODE.git
    cd InvODE
    pip install -r requirements.txt


Example Usage
-------------

Here's a minimal working example using `ODEOptimizer`:

.. code-block:: python

    from invode.optimizer import ODEOptimizer

    def my_ode(params):
        # Example model output
        return [1.0, 2.0]  # Replace with actual simulation output

    def my_error(sim_output):
        # Example error function
        return sum((x - y)**2 for x, y in zip(sim_output, [1.1, 2.1]))

    bounds = {'k1': (0.1, 1.0), 'k2': (0.01, 0.5)}

    optimizer = ODEOptimizer(
        ode_func=my_ode,
        error_func=my_error,
        param_bounds=bounds,
        n_samples=50,
        num_iter=5,
        verbose=True
    )

    best_params, best_error = optimizer.fit()

    print("Best Parameters:", best_params)
    print("Best Error:", best_error)

Summary and Plots
-----------------

You can view a summary and plot of the error history:

.. code-block:: python

    optimizer.summary()
    optimizer.plot_error_history()