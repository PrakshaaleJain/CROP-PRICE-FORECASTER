import requests
import pandas as pd
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv

load_dotenv()

url = "https://dca.ceda.ashoka.edu.in/index.php/home/download"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

select = soup.find("select", {"id": "centre"})
if select is None:
    raise Exception("Couldn't find the select element with id='centre'")

districts = []
for option in select.find_all("option"):
    value = option.get("value")
    name = option.text.strip()
    # Skip placeholder option
    if value != "0":
        districts.append({"district_id": value, "district_name": name})


# save the file 
path = os.environ.get("district_ID_path") # change to your own path and file name
df = pd.DataFrame(districts)
df.to_csv(path, index=True)