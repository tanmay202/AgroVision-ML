import pandas as pd, numpy as np

tr = pd.read_csv('data/processed/tea_train.csv')
te = pd.read_csv('data/processed/tea_test.csv')

pc = 'Modal Price (Rs./Quintal)'

high_mae_groups = [
    ('Parappanangadi', 'Other'),
    ('Dhanbad', 'Other'),
    ('Bishalgarh', 'Other'),
    ('Thirurrangadi', 'Other'),
    ('Sakhigopal', 'All Dust'),
]

print('=== HIGH-MAE GROUP DEEP DIVE ===')
for mkt, var in high_mae_groups:
    sub = tr[(tr['Market Name']==mkt) & (tr['Variety']==var)]
    diff = sub['future_modal_price'] - sub[pc]
    zero_pct = (diff==0).mean()*100
    print(f'{mkt}/{var}: n={len(sub)}, price_mean={sub[pc].mean():.0f}, zero={zero_pct:.0f}%, std={sub[pc].std():.0f}, change_std={diff.std():.0f}')
    print(f'  days_to_next dist: {sub["days_to_next"].value_counts().sort_index().to_dict()}')

print()
sub = tr[(tr['Market Name']=='Parappanangadi') & (tr['Variety']=='Other')].sort_values('Reported Date')
diff = (sub['future_modal_price'] - sub[pc]).dropna()
print('Parappanangadi non-zero changes:')
print(diff[diff != 0].describe())
print('Last 20 training rows:')
print(sub[['Reported Date', pc, 'future_modal_price', 'days_to_next']].tail(20).to_string())
