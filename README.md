# ATC-138 Testing Workflow

## Purpose

- Run example models using an unmodified installed version of the released ATC-138 Python package and compare against the main branch of `PBEE-Recovery` (MATLAB) within a reproducible workflow.
- Check comparison metrics across models to validate the release.

## Notes

- `run_model.ipynb` is the runner script used after installing ATC-138 Python. It loops through all test models in the target folder. `force_rebuild=True` is used to regenerate `simulated_inputs.json`. Users should ensure the latest installed package is being imported.
- `comparison_batch.py` runs batch comparisons across all models and generates summary comparison results.
- Input for model `1900_S1a_6_IM1` is included as an example to demonstrate the expected model format.