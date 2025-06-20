# Physics-Informed Machine Learning for Ethanol-Water Distillation Column

This repository presents a research-grade workflow for applying machine learning and physics-informed neural networks (PINN) to a simulated ethanol-water distillation column. The project demonstrates how to combine domain knowledge, feature engineering, and hybrid ML models for robust, interpretable predictions in chemical engineering.

## Project Highlights
- **Objective:** Predict distillate ethanol concentration using tray temperatures, flowrates, and engineered features, while enforcing physical constraints (mass balance, bounds).
- **Workflow:**
  - Data exploration, cleaning, and feature engineering (`eda_cleaning.ipynb`)
  - Baseline regression models: Linear Regression, Random Forest, XGBoost
  - Physics-Informed Neural Network (PINN) and hybrid Random Forest + PINN
  - Model evaluation with regression metrics and interpretability

## Key Features
- **Physics-Informed ML:** PINN enforces mass balance and value bounds during training for physically consistent predictions.
- **Hybrid Modeling:** Combines Random Forest for initial prediction and PINN for learning residuals, improving accuracy and physical realism.
- **Domain Feature Engineering:** Includes tray temperature gradients, flowrate ratios, and process-inspired statistics.
- **Reproducible Pipeline:** Modular code, clear documentation, and requirements for easy setup.

## File Structure
- `eda_cleaning.ipynb` — Data exploration, cleaning, and feature engineering
- `modelling.ipynb` — Model training, evaluation, and hybrid PINN workflow
- `pinn.py` — PINN model and custom loss function
- `cleaned_distillation.csv` — Cleaned dataset for modeling

## Getting Started
1. **Clone the repository:**
   ```bash
   git clone https://github.com/2004Snehal/Ethanol_Distillation_PINN.git
   cd Ethanol_Distillation_PINN
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run notebooks:**
   - Start with `eda_cleaning.ipynb` for data prep
   - Proceed to `modelling.ipynb` for model training and analysis

## Requirements
See `requirements.txt` for all dependencies.

## Citation
If you use this project or adapt the PINN workflow, please cite or acknowledge the repository.

## License
MIT License

## Dataset Citation
If the dataset is helpful please cite:
Cote-Ballesteros, J. E., Grisales Palacios, V. H., & Rodriguez-Castellanos, J. E. (2022). Un algoritmo de selección de variables de enfoque híbrido basado en información mutua para aplicaciones de sensores blandos industriales basados en datos. Ciencia E Ingeniería Neogranadina, 32(1), 59–70. https://doi.org/10.18359/rcin.5644
