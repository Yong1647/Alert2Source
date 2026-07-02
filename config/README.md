# Configs

These YAML files are sanitized templates based on the uploaded experiment configs.
Local absolute paths were replaced with portable repository-relative placeholders:

- `data/raw/` for raw multimodal inputs
- `data/processed/final_master_data_FINAL.csv` for the master station/sample table
- `outputs/` for generated models and paper outputs

Edit these paths before running Stage 0 training or SHAP diagnostics on the server.
