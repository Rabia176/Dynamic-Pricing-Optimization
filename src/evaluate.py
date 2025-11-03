import joblib, pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate(model_path, features_csv):
    art = joblib.load(model_path)
    model = art['model']
    feats = art['features']
    df = pd.read_csv(features_csv)
    X = df[feats]
    y = df['demand']
    preds = model.predict(X)
    print('MAE:', mean_absolute_error(y, preds))
    print('RMSE:', mean_squared_error(y, preds, squared=False))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--features', required=True)
    args = parser.parse_args()
    evaluate(args.model, args.features)
