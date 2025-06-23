import requests
import urllib.parse
import time

def download_district_data(commodity_id, state_id, district_id, year):
    base_url = "https://dca.ceda.ashoka.edu.in/index.php/home/getcsv"
    # Build query string
    params = {
        "c": f'["{commodity_id}"]',  # c needs to be a JSON-like string
        "s": state_id,
        "t": district_id,
        "y": year
    }
    
    # Encode URL parameters correctly
    encoded_params = {k: urllib.parse.quote_plus(str(v)) for k, v in params.items()}
    full_url = f"{base_url}?c={encoded_params['c']}&s={params['s']}&t={params['t']}&y={params['y']}"
    
    print(f"Downloading from: {full_url}")
    
    response = requests.get(full_url)
    
    if response.status_code == 200 and response.content:
        # filename = f"commodity_{commodity_id}_state_{state_id}_district_{district_id}_{year}.csv"
        filename = f".csv"
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"✅ Saved: {filename}")
    else:
        print(f"❌ Failed to download data for {year}")


# t = 1 for all districts
commodity_id = 1
state_id = 3 
t = 1

for year in range(2009, 2026):
    download_district_data(commodity_id, state_id, district_id, year)
    time.sleep(2)  # Be respectful of server load
