# ATC-138 Testing Workflow

## Purpose

- Run example models using an unmodified installed version of the released ATC-138 Python package and compare against the main branch of `PBEE-Recovery` (MATLAB) within a reproducible workflow.
- Check comparison metrics across models to validate the release.

## Notes

- `run_model.ipynb` is the Python runner used after installing the released ATC-138 package. It loops through all test models in the target folder. `force_rebuild=True` is used to regenerate `simulated_inputs.json`. Users should ensure the latest installed package is being imported.
- `run_single_model.m` runs a single MATLAB model. It also exports both `.mat` and `.json` recovery outputs for use in the comparison script.
- `run_models.m` uses `run_single_model.m` to run models in batch. It automatically searches for models under `inputs/example_inputs` and runs assessments for each model.
- `comparison_batch.py` runs batch comparisons across all models and generates summary comparison results.
- Input for model `1900_S1a_6_IM1` is included as an example to demonstrate the expected model format.
