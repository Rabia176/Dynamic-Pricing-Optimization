# Dynamic Pricing Optimization (Synthetic Dataset Demo)

This project demonstrates a dynamic pricing optimization pipeline:
- synthetic dataset of 1000 transactions (`data/pricing_synthetic_1000.csv`)
- train a demand model (XGBoost/RandomForest) to predict demand given price & features
- use the demand model to search for a price that maximizes predicted revenue = price * demand(price)
- includes scripts, notebook, and a pre-trained model for instant demo

Quick start:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/preprocess.py --raw data/pricing_synthetic_1000.csv --out data/processed/features.csv
python src/train_demand.py --features data/processed/features.csv --out experiments/demand_model.joblib
python src/optimize_price.py --model experiments/demand_model.joblib --product P1 --out predictions_price_opt.csv
```
