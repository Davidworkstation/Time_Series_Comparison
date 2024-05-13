from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import time
import pandas as pd
from collections import OrderedDict

# Initialize a Selenium WebDriver instance
driver = webdriver.Chrome()  # Or use any other WebDriver you prefer
driver.maximize_window()

# Load the webpage
url = 'https://www.macrotrends.net/stocks/charts/JPM/jpmorgan-chase/cash-flow-statement?freq=Q'
driver.get(url)

# Wait for the jqxGrid to be visible
wait = WebDriverWait(driver, 30)
jqxgrid = wait.until(EC.visibility_of_element_located((By.ID, 'jqxgrid')))

# Scroll down to trigger lazy loading of additional data
body = driver.find_element(By.TAG_NAME, 'body')
for _ in range(7):  # Adjust the number of scrolls as needed
    body.send_keys(Keys.PAGE_DOWN)
    time.sleep(0.3)  # Adjust the sleep time as needed

# Scroll right to ensure all content within the grid is loaded
for _ in range(3):  # Adjust the number of scrolls as needed
    body.send_keys(Keys.ARROW_RIGHT)
    time.sleep(1)  # Adjust the sleep time as needed

# Wait for a brief moment to ensure all data is loaded
time.sleep(5)  # Adjust the sleep time as needed

# Once the jqxGrid is loaded, extract its HTML content
html_content = driver.page_source

# Close the WebDriver
driver.quit()

# 2nd - second round
driver = webdriver.Chrome()  
driver.maximize_window()
url = 'https://www.macrotrends.net/stocks/charts/JPM/jpmorgan-chase/cash-flow-statement?freq=Q'
driver.get(url)
wait = WebDriverWait(driver, 30)
jqxgrid = wait.until(EC.visibility_of_element_located((By.ID, 'jqxgrid')))
body = driver.find_element(By.TAG_NAME, 'body')
for _ in range(5):  
    body.send_keys(Keys.PAGE_DOWN)
    time.sleep(0.5) 
scroll_button_id = 'jqxScrollBtnDownhorizontalScrollBarjqxgrid'
scroll_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, scroll_button_id)))
num_clicks = 150 
for _ in range(num_clicks):
    scroll_button.click()
    time.sleep(0.05)  
time.sleep(5)  
html_content2 = driver.page_source
driver.quit()

# 3rd - third round
driver = webdriver.Chrome()  
driver.maximize_window()
url = 'https://www.macrotrends.net/stocks/charts/JPM/jpmorgan-chase/cash-flow-statement?freq=Q'
driver.get(url)
wait = WebDriverWait(driver, 30)
jqxgrid = wait.until(EC.visibility_of_element_located((By.ID, 'jqxgrid')))
body = driver.find_element(By.TAG_NAME, 'body')
for _ in range(5): 
    body.send_keys(Keys.PAGE_DOWN)
    time.sleep(0.5)  
scroll_button_id = 'jqxScrollBtnDownhorizontalScrollBarjqxgrid'
scroll_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, scroll_button_id)))
num_clicks = 300 
for _ in range(num_clicks):
    scroll_button.click()
    time.sleep(0.05)  
time.sleep(5) 
html_content3 = driver.page_source
driver.quit()


# 4th - fourth round
driver = webdriver.Chrome()  
driver.maximize_window()
url = 'https://www.macrotrends.net/stocks/charts/JPM/jpmorgan-chase/cash-flow-statement?freq=Q'
driver.get(url)
wait = WebDriverWait(driver, 30)
jqxgrid = wait.until(EC.visibility_of_element_located((By.ID, 'jqxgrid')))
body = driver.find_element(By.TAG_NAME, 'body')
for _ in range(4):  
    body.send_keys(Keys.PAGE_DOWN)
    time.sleep(0.5) 
scroll_button_id = 'jqxScrollBtnDownhorizontalScrollBarjqxgrid'
scroll_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, scroll_button_id)))
num_clicks = 450 
for _ in range(num_clicks):
    scroll_button.click()
    time.sleep(0.05)  
time.sleep(5)  
html_content4 = driver.page_source
driver.quit()

# 5th - fifth round
driver = webdriver.Chrome()  
driver.maximize_window()
url = 'https://www.macrotrends.net/stocks/charts/JPM/jpmorgan-chase/cash-flow-statement?freq=Q'
driver.get(url)
wait = WebDriverWait(driver, 30)
jqxgrid = wait.until(EC.visibility_of_element_located((By.ID, 'jqxgrid')))
body = driver.find_element(By.TAG_NAME, 'body')
for _ in range(4):  
    body.send_keys(Keys.PAGE_DOWN)
    time.sleep(0.5) 
scroll_button_id = 'jqxScrollBtnDownhorizontalScrollBarjqxgrid'
scroll_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, scroll_button_id)))
num_clicks = 600
for _ in range(num_clicks):
    scroll_button.click()
    time.sleep(0.05)  
time.sleep(2.5)  
html_content5 = driver.page_source
driver.quit()

# Now you can parse the HTML content using BeautifulSoup
soup = BeautifulSoup(html_content, 'html.parser')
soup2 = BeautifulSoup(html_content2, 'html.parser')
soup3 = BeautifulSoup(html_content3, 'html.parser')
soup4 = BeautifulSoup(html_content4, 'html.parser')
soup5 = BeautifulSoup(html_content5, 'html.parser')

# Extract and print the cell values
rows = soup.select('div[id^="row"]')
rows2 = soup2.select('div[id^="row"]')
rows3 = soup3.select('div[id^="row"]')
rows4 = soup4.select('div[id^="row"]')
rows5 = soup5.select('div[id^="row"]')

rows_columns = soup.select('[role="columnheader"]')
rows2_colulmns = soup2.select('[role="columnheader"]')
rows3_colulmns = soup3.select('[role="columnheader"]')
rows4_colulmns = soup4.select('[role="columnheader"]')
rows5_colulmns = soup5.select('[role="columnheader"]')


data1_header = []
for column in rows_columns:
    # Find the span tag within each column
    span_tag = column.find('span')
    if span_tag:
        # Extract text from the span tag and strip any leading or trailing whitespace
        column_text = span_tag.get_text(strip=True)
        data1_header.append(column_text)

# Convert the data into a list of dictionaries
data2_header = []
for column in rows2_colulmns:
    # Find the span tag within each column
    span_tag = column.find('span')
    if span_tag:
        # Extract text from the span tag and strip any leading or trailing whitespace
        column_text = span_tag.get_text(strip=True)
        data2_header.append(column_text)

data3_header = []
for column in rows3_colulmns:
    # Find the span tag within each column
    span_tag = column.find('span')
    if span_tag:
        # Extract text from the span tag and strip any leading or trailing whitespace
        column_text = span_tag.get_text(strip=True)
        data2_header.append(column_text)

data4_header = []
for column in rows4_colulmns:
    # Find the span tag within each column
    span_tag = column.find('span')
    if span_tag:
        # Extract text from the span tag and strip any leading or trailing whitespace
        column_text = span_tag.get_text(strip=True)
        data2_header.append(column_text)

data5_header = []
for column in rows5_colulmns:
    # Find the span tag within each column
    span_tag = column.find('span')
    if span_tag:
        # Extract text from the span tag and strip any leading or trailing whitespace
        column_text = span_tag.get_text(strip=True)
        data2_header.append(column_text)

headers = data1_header + data2_header + data3_header + data4_header + data5_header
headers_ordered = list(OrderedDict.fromkeys(headers))
headers_datetime = [pd.to_datetime(header, errors='coerce') for header in headers_ordered]

data1 = []
for row in rows:
    cells = row.find_all('div', class_='jqx-grid-cell')
    row_data = {}
    for idx, cell in enumerate(cells):
        row_data[f'Column{idx + 1}'] = cell.text.strip()
    data1.append(row_data)

# Convert the data into a list of dictionaries
data2 = []
for row in rows2:
    cells = row.find_all('div', class_='jqx-grid-cell')
    row_data = {}
    for idx, cell in enumerate(cells):
        row_data[f'Column{idx + 1}'] = cell.text.strip()
    data2.append(row_data)

data3 = []
for row in rows3:
    cells = row.find_all('div', class_='jqx-grid-cell')
    row_data = {}
    for idx, cell in enumerate(cells):
        row_data[f'Column{idx + 1}'] = cell.text.strip()
    data3.append(row_data)

data4 = []
for row in rows4:
    cells = row.find_all('div', class_='jqx-grid-cell')
    row_data = {}
    for idx, cell in enumerate(cells):
        row_data[f'Column{idx + 1}'] = cell.text.strip()
    data4.append(row_data)

data5 = []
for row in rows5:
    cells = row.find_all('div', class_='jqx-grid-cell')
    row_data = {}
    for idx, cell in enumerate(cells):
        row_data[f'Column{idx + 1}'] = cell.text.strip()
    data5.append(row_data)

# Convert the list of dictionaries into DataFrame
df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)
df2.drop(columns=['Column1', 'Column2'], inplace=True)
df3 = pd.DataFrame(data3)
df3.drop(columns=['Column1', 'Column2'], inplace=True)
df4 = pd.DataFrame(data4)
df4.drop(columns=['Column1', 'Column2'], inplace=True)
df5 = pd.DataFrame(data5)
df5.drop(columns=['Column1', 'Column2'], inplace=True)

combined_df = pd.concat([df1, df2, df3, df4, df5], axis=1)

# Transpose the concatenated dataframe
combined_df_transposed = combined_df.T
# Drop duplicate rows
combined_df_transposed = combined_df_transposed.loc[~combined_df_transposed.duplicated(), :]
# Transpose the dataframe back to its original orientation
combined_df_cleaned = combined_df_transposed.T

new_row_df = pd.DataFrame([headers_datetime], columns=combined_df_cleaned.columns)
combined_df_cleaned = pd.concat([new_row_df, combined_df_cleaned], ignore_index=True)

print(combined_df_cleaned)
combined_df_cleaned.to_csv('JPcashflow.csv', index=False)

