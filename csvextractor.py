import pandas as pd

df = pd.read_csv("Rice.csv")

# Clean text columns
df["State_Name"] = df["State_Name"].astype(str).str.strip()
df["Crop"] = df["Crop"].astype(str).str.strip()

# Extract everything related to West Bengal + Rice
wb_rice = df[
    df["State_Name"].str.contains("west bengal", case=False, na=False) &
    df["Crop"].str.contains("rice", case=False, na=False)
]

print(wb_rice)
print("Total rows:", len(wb_rice))

# Save
wb_rice.to_csv("west_bengal_rice.csv", index=False)