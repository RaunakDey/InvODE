.. invode documentation master file, created by
   sphinx-quickstart on Mon Jul 21 20:33:46 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. Optimizer documentation master file, created by
   sphinx-quickstart on 2025-07-21.


Welcome to InvODE's documentation!
=====================================

**InvODE** is a Python package for fitting ODE-based models to data with support for:

- Parameter and initial condition estimation.
- Flexible solvers and constraints.
- Support for replicates, latent variables, and identifiability checks.
- Custom loss functions and regularization.
- Domain versatility: Examples from biology, pharmacology, physics, finance and beyond.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   api_reference
   case_studies/index  

.. toctree::
   :maxdepth: 2
   :caption: Features

   features/solvers
   features/parameter_fitting
   features/constraints
   features/replicates_latents
   features/identifiability
   features/local_minima

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   contributing
   testing
   changelog


