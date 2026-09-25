import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from src.config import RAW_DATA_FILE

df = pd.read_csv(RAW_DATA_FILE)
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
