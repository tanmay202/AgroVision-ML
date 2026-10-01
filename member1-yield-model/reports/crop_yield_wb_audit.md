# M1 Rice Yield — Data Audit

## Source

Raw file:
`D:\MCA\SEM3\AgroVision-ML\member1-yield-model\data\raw\rice.csv`

## Filter

- State: West Bengal
- Crop: Rice

## Counts

- Raw rows: 1,637
- West Bengal Rice rows: 1,637
- Final valid rows: 1,637

## Coverage

- Years: 1997 -
  2019
- Districts: 23
- Seasons: 3

## Missing / Invalid

- Invalid Area: 0
- Invalid Production: 0
- Duplicate modeling keys: 608

## Target

Yield is calculated only where Area > 0 and Production is valid:

`Yield_Kg_Ha = (Production_Tonnes / Area_Hectares) * 1000`

No missing values were estimated or imputed.

## Output

`D:\MCA\SEM3\AgroVision-ML\member1-yield-model\data\processed\rice_yield_wb.csv`


## Existing `yield` Column Verification

The raw dataset contained an existing `yield` column.

- Rows compared: 1,637
- Mean absolute difference:
  198212.704695
- Maximum absolute difference:
  447491.951088

The calculated value is used as the M1 target.
