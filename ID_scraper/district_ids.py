import requests
import pandas as pd
from bs4 import BeautifulSoup

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

df = pd.DataFrame(districts)
df.to_csv(r"C:\Users\HP\OneDrive\Desktop\CROP PRICE FORCASTER\ID\district_id.csv")