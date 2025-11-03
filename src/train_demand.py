import pandas as pd, argparse, os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

def train(features_csv, out_model):
    df = pd.read_csv(features_csv)
    X = df.drop(columns=['tx_id','demand','revenue'])
    y = df['demand']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
    # try XGBoost
    model = xgb.XGBRegressor(n_estimators=200, random_state=42, n_jobs=4)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print('MAE:', mean_absolute_error(y_test, preds))
    print('RMSE:', mean_squared_error(y_test, preds, squared=False))
    os.makedirs(os.path.dirname(out_model), exist_ok=True)
    joblib.dump({'model': model, 'features': X.columns.tolist()}, out_model)
    print('Saved model to', out_model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    train(args.features, args.out)
