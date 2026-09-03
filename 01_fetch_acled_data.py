import os
import time
import pandas as pd
import requests
from dotenv import load_dotenv

# 1. Load secret variables from the hidden .env file
load_dotenv()

ACLED_EMAIL = os.getenv("ACLED_EMAIL")
ACLED_PASSWORD = os.getenv("ACLED_PASSWORD")

# Verify that credentials loaded properly
if not ACLED_EMAIL or not ACLED_PASSWORD:
    raise ValueError(
        "Credentials missing! Please check your .env file in the project folder."
    )

# 2. Authenticate and get Token
token_url = "https://acleddata.com/oauth/token"
payload = {
    "username": ACLED_EMAIL,  # Loaded safely from .env
    "password": ACLED_PASSWORD,  # Loaded safely from .env
    "grant_type": "password",
    "client_id": "acled",
    "scope": "authenticated",
}

print("Logging in securely...")
token_res = requests.post(token_url, data=payload)

if token_res.status_code == 200:
    access_token = token_res.json().get("access_token")
    print("-> Token successfully generated!")
else:
    raise Exception(
        f"Login Failed ({token_res.status_code}): {token_res.text}"
    )

# 3. Set Up Pagination Loop
api_url = "https://acleddata.com/api/acled/read"
headers = {"Authorization": f"Bearer {access_token}"}

all_events = []
page = 1
limit_per_page = 5000

print(
    "Starting full data collection for Myanmar (Jan 1, 2021 - Jun 30, 2025)..."
)

while True:
    params = {
        "country": "Myanmar",
        "event_date": "2021-01-01|2025-06-30",
        "event_date_where": "BETWEEN",
        "limit": limit_per_page,
        "page": page,  # Requests Page 1, then Page 2, etc.
    }

    response = requests.get(api_url, headers=headers, params=params)

    if response.status_code != 200:
        print(f"Error on page {page}: {response.status_code}")
        break

    data = response.json().get("data", [])

    # If the page returns no events, we reached the end of the dataset
    if not data:
        print("-> Reached the end of the data stream.")
        break

    all_events.extend(data)
    print(
        f"Fetched Page {page} ({len(data)} records). Total so far: {len(all_events)}"
    )

    page += 1
    time.sleep(1)  # Pause 1 second between requests to respect server rates

# 4. Convert all pages into one Master DataFrame and Save
df_master = pd.DataFrame(all_events)
print(f"\nSuccess! Downloaded {len(df_master)} total conflict records.")

df_master.to_csv("myanmar_acled_2021_2025_full.csv", index=False)
print("Saved master file as: myanmar_acled_2021_2025_full.csv")