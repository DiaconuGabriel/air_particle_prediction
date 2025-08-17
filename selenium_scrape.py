import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, ElementClickInterceptedException, TimeoutException
import pandas as pd
from io import StringIO
import os
from dotenv import load_dotenv

load_dotenv()

def scrape_aqicn_data(
    url="https://aqicn.org/station/@124327/ro/",
    buttons=None,
    chrome_driver_path=os.getenv("chrome_driver_path"), # your chrome driver path
    output_csv="data/data.csv" # your output csv path
):
    if buttons is None:
        buttons = ['co2','R.H.','Press','Temp','O3','PM1','PM10','PM2.5']
    
    dataframes = []

    service = Service(chrome_driver_path)
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")

    with webdriver.Chrome(service=service, options=options) as driver:
        wait = WebDriverWait(driver, 15)
        driver.get(url)

        # Scroll till finds the relevant header
        while True:
            try:
                header = driver.find_element(
                    By.XPATH,
                    "//div[@class='section-content-header' and contains(text(),'Date istorice privind calitatea aerului')]"
                )
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", header)
                time.sleep(0.5)
                break
            except NoSuchElementException:
                driver.execute_script("window.scrollBy(0, 100);")
                time.sleep(0.5)

        time.sleep(1)

        for buton in buttons:
            try:
                btn = wait.until(
                    EC.element_to_be_clickable(
                        (By.XPATH, f"//div[@class='d3ui btns']/div[text()='{buton}'] | //div[@class='d3ui btns']/div[contains(@class,'d3ui btn') and normalize-space(string(.))='{buton}']")
                    )
                )
                driver.execute_script("arguments[0].scrollIntoView();", btn)
                try:
                    btn.click()
                except ElementClickInterceptedException:
                    driver.execute_script("arguments[0].click();", btn)
                
                time.sleep(5)

                export_btn = wait.until(EC.element_to_be_clickable((By.XPATH, "//div[text()='Exportați în CSV']")))
                driver.execute_script("arguments[0].scrollIntoView(true);", export_btn)
                try:
                    export_btn.click()
                except ElementClickInterceptedException:
                    driver.execute_script("arguments[0].click();", export_btn)

                driver.switch_to.window(driver.window_handles[1])
                wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                time.sleep(0.5)
                data = driver.find_element(By.TAG_NAME, "body").text

                # Get data for csv file
                lines = data.splitlines()
                csv_text = "\n".join(lines[3:])
                df = pd.read_csv(StringIO(csv_text), header=0)
                df = df[['date', 'median']].rename(columns={'median': f"{buton}_median"})
                dataframes.append(df)
                time.sleep(1)

                driver.close()
                driver.switch_to.window(driver.window_handles[0])

                time.sleep(1)

            except TimeoutException:
                print(f"Nu am găsit butonul sau datele pentru {buton}, trec mai departe.")
                continue

        # Convert date columns
        for i, df in enumerate(dataframes):
            dataframes[i]['date'] = pd.to_datetime(df['date'], utc=True)

        min_date = min(df['date'].min() for df in dataframes)
        max_date = max(df['date'].max() for df in dataframes)
        all_dates = pd.date_range(start=min_date, end=max_date, freq='D')
        df_final = pd.DataFrame({'date': all_dates})

        for df in dataframes:
            df_final = df_final.merge(df, on='date', how='left')

        df_final.fillna(method='ffill', inplace=True)
        df_final.to_csv(output_csv, index=False)
        print(f"Datele au fost salvate în {output_csv}")

