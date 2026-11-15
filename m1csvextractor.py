import pandas as pd

# Load dataset
df = pd.read_csv(r"D:\MCA\SEM3\AgroVision-ML\member1-yield-model\data\raw\yield.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Clean text columns
df["State_Name"] = df["State_Name"].astype(str).str.strip()
df["Crop"] = df["Crop"].astype(str).str.strip()

# Extract West Bengal + Wheat
wb_potato = df[
    (df["State_Name"].str.casefold() == "west bengal") &
    (df["Crop"].str.casefold() == "potato")
].copy()

# Display results
print(wb_potato.head())
print("Total rows:", len(wb_potato))

# Save filtered dataset
wb_potato.to_csv("potato.csv", index=False)

print("Dataset saved successfully!")