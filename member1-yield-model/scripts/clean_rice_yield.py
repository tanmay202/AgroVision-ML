import pandas as pd

INPUT = "member1-yield-model/data/raw/rice.csv"
OUTPUT = "member1-yield-model/data/processed/rice_yield_wb_clean.csv"

KEYS = ["District_Name", "Season", "Crop_Year"]

df = pd.read_csv(INPUT)

df["Area"] = pd.to_numeric(df["Area"], errors="coerce")
df["Production"] = pd.to_numeric(df["Production"], errors="coerce")

duplicate_keys = df[df.duplicated(KEYS, keep=False)][KEYS].drop_duplicates()

print("Duplicate keys:", len(duplicate_keys))

bad_groups = []

for _, key in duplicate_keys.iterrows():

    mask = (
        (df["District_Name"] == key["District_Name"]) &
        (df["Season"] == key["Season"]) &
        (df["Crop_Year"] == key["Crop_Year"])
    )

    g = df.loc[mask].copy()

    # Expected pattern: same area, one production exactly 100x the other
    same_area = g["Area"].nunique() == 1

    productions = sorted(g["Production"].dropna().unique())

    valid_pattern = (
        same_area
        and len(productions) == 2
        and productions[1] == productions[0] * 100
    )

    if not valid_pattern:
        bad_groups.append(g)

if bad_groups:
    print("\nWARNING: Unexpected duplicate patterns found.")
    print("Groups:", len(bad_groups))

    bad = pd.concat(bad_groups)
    print(bad[KEYS + ["Area", "Production", "yield"]].to_string(index=False))

    raise ValueError("Duplicate groups need manual review.")

# Remove duplicate unit versions.
# Keep the smaller Production value.
clean = (
    df.sort_values("Production")
      .drop_duplicates(KEYS, keep="first")
      .copy()
)

# Calculate yield from official units:
# Production = tonnes
# Area = hectares
clean["Rice_Yield_t_ha"] = clean["Production"] / clean["Area"]

# Convert tonnes/ha -> kg/ha
clean["Rice_Yield_kg_ha"] = clean["Rice_Yield_t_ha"] * 1000

# Final validation
assert not clean.duplicated(KEYS).any()
assert clean["Rice_Yield_kg_ha"].notna().all()

clean.to_csv(OUTPUT, index=False)

print("\n======================================")
print("RICE YIELD CLEANING COMPLETE")
print("======================================")
print("Rows:", len(clean))
print("Districts:", clean["District_Name"].nunique())
print("Years:", clean["Crop_Year"].min(), "-", clean["Crop_Year"].max())
print("Duplicate keys after cleaning:", clean.duplicated(KEYS).sum())
print("\nSeasons:")
print(clean["Season"].value_counts())
print("\nSample:")
print(
    clean[
        KEYS + ["Area", "Production", "Rice_Yield_t_ha", "Rice_Yield_kg_ha"]
    ].head(10).to_string(index=False)
)