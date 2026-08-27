import pandas as pd

df = pd.read_csv('data/raw/tea_cleaned.csv')
print('Shape:', df.shape)
print('\nColumns:', list(df.columns))
print('\nFirst 5 rows:')
print(df.head().to_string())
print('\nData types:')
print(df.dtypes)
print('\nUnique values in key columns:')
for c in df.columns:
    print(f'  {c}: {df[c].nunique()} unique')
print('\nMissing values:')
print(df.isnull().sum())
