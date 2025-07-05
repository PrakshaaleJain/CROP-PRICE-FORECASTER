import requests
import urllib.parse
import time
import pandas as pd
import os



def download_district_data(commodity_id, district_id, t, year, save_dir=""):
    '''
    Downloads historical data from base_url and saves them
    '''

    base_url = "https://dca.ceda.ashoka.edu.in/index.php/home/getcsv"
    # Build query string
    params = {
        "c": f'["{commodity_id}"]',  # c needs to be a JSON-like string
        "s": district_id,
        "t": t,
        "y": year
    }
    
    # Encode URL parameters correctly
    encoded_params = {k: urllib.parse.quote_plus(str(v)) for k, v in params.items()}
    full_url = f"{base_url}?c={encoded_params['c']}&s={params['s']}&t={params['t']}&y={params['y']}"
    
    response = requests.get(full_url)
    
    if response.status_code == 200 and response.content:
        filename = f"commodity_{commodity_id}_district_{district_id}_{year}.csv"
        filepath = os.path.join(save_dir, filename)
        with open(filepath, 'wb') as f:
            f.write(response.content)
        print(f"Saved: {filename}")
    else:
        print(f"Failed to download data for {year}")



commodity_id = 1    # 1 is for Wheat(Atta)
district_id = 3     # 3 is for Agra
t = 1               # always 1
save_dir = "datasets/"

for year in range(2009, 2026):
    download_district_data(commodity_id, district_id, t, year, save_dir)
    time.sleep(2)


# Concatenate all the donwloaded csv files
all_files = [os.path.join(save_dir, f) for f in os.listdir(save_dir) if f.endswith('.csv')]
df_list = [pd.read_csv(f) for f in all_files]
combined_df = pd.concat(df_list, ignore_index=True)
combined_df.to_csv(os.path.join(save_dir, f"commodity_{commodity_id}_district_{district_id}.csv"), index=False)

