import joblib, pandas as pd, numpy as np, argparse

def optimize_price(model_path, product, out_csv=None):
    art = joblib.load(model_path)
    model = art['model']
    feats = art['features']
    # create candidate prices around observed competitor prices for product
    # here we will sample price range
    candidate_prices = np.linspace(1, 200, 200)
    # create feature grid for the product (use median values)
    median = {}
    # set defaults
    median_vals = {f:0 for f in feats}
    # find product_enc
    # assume product_enc is the smallest enc for demonstration
    median_vals['product_enc'] = 0
    median_vals['base_cost'] = 20
    median_vals['competitor_price'] = 20
    median_vals['promotion'] = 0
    median_vals['marketing'] = 100
    median_vals['month'] = 6
    median_vals['weekday'] = 2
    rows = []
    for p in candidate_prices:
        x = median_vals.copy()
        x['price'] = p
        X = pd.DataFrame([x])[feats]
        pred_demand = model.predict(X)[0]
        revenue = p * pred_demand
        rows.append({'price': float(p), 'pred_demand': float(pred_demand), 'pred_revenue': float(revenue)})
    df = pd.DataFrame(rows)
    best = df.loc[df['pred_revenue'].idxmax()]
    print('Best price:', best['price'], 'expected revenue:', best['pred_revenue'])
    if out_csv:
        df.to_csv(out_csv, index=False)
        print('Saved grid to', out_csv)
    return best, df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--product', required=True)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    optimize_price(args.model, args.product, args.out)
