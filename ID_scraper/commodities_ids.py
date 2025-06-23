from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import pandas as pd
import time

service = Service()  # Or use Service("path/to/chromedriver") if not in PATH
options = webdriver.ChromeOptions()
options.add_argument("--headless")  

driver = webdriver.Chrome(service=service, options=options)
driver.get("https://dca.ceda.ashoka.edu.in/index.php/home/download")
time.sleep(3)  # Wait for JS to load

commodity_dropdown = driver.find_element(By.ID, "commodity")
options = commodity_dropdown.find_elements(By.TAG_NAME, "option")

# Parse commodities
commodities = {
    option.text.strip(): int(option.get_attribute("value"))
    for option in options if option.get_attribute("value") != "0"
}

# Save to CSV
df = pd.DataFrame(list(commodities.items()), columns=["Commodity", "ID"])
df.to_csv(r"C:\Users\HP\OneDrive\Desktop\CROP PRICE FORCASTER\ID\commodity_ids.csv", index=False)

driver.quit()
