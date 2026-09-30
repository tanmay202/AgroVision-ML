import pandas as pd

# Load your original CSV
df = pd.read_csv("Rice.csv")

# Keep only West Bengal
west_bengal = df[df["State Name"] == "West Bengal"]

# Save as a new CSV
west_bengal.to_csv("West_Bengal_Mandi.csv", index=False)

print(f"Extracted {len(west_bengal)} rows")