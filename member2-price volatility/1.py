import os
import requests
import pandas as pd

output_dir = "agrovision/data/raw"
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, "mandi_data.csv")

api_key = "579b464db66ec23bdd000001cdd3946e44ce4aad7209ff7b23ac571b"
resource_id = "9ef84268-d588-465a-a308-a864a43d0070"

# Request JSON format instead of CSV
url = f"https://api.data.gov.in/resource/{resource_id}?api-key={api_key}&format=json&limit=10000"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

print("Downloading dataset from Data.gov.in via JSON endpoint...")
response = requests.get(url, headers=headers, timeout=30)

if response.status_code == 200:
    data = response.json()
    if "records" in data:
        df = pd.DataFrame(data["records"])
        df.to_csv(csv_path, index=False)
        print(f"Data successfully saved to {csv_path}")
        print(f"Downloaded {len(df)} rows and {len(df.columns)} columns.")
    else:
        print("API Response did not contain 'records':", data)
else:
    print(f"Failed to fetch data. Status Code: {response.status_code}")