import pandas as pd, argparse, os

def preprocess(df):
    # create simple encodings
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['weekday'] = df['date'].dt.weekday
    # one-hot or label encode product
    df['product_enc'] = df['product'].astype('category').cat.codes
    feats = ['product_enc','base_cost','competitor_price','price','promotion','marketing','month','weekday']
    out = df[['tx_id'] + feats + ['demand','revenue']].copy()
    return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    df = pd.read_csv(args.raw)
    out = preprocess(df)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print('Saved processed features to', args.out)
