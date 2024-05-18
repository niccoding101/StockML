import os
import re
import time
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from utils import data_string_to_float

# Path to fundamental data
statspath = "intraQuarter/_KeyStats/"

# Features to be parsed
features = [
    "Market Cap", "Enterprise Value", "Trailing P/E", "Forward P/E", "PEG Ratio", "Price/Sales",
    "Price/Book", "Enterprise Value/Revenue", "Enterprise Value/EBITDA", "Profit Margin",
    "Operating Margin", "Return on Assets", "Return on Equity", "Revenue", "Revenue Per Share",
    "Quarterly Revenue Growth", "Gross Profit", "EBITDA", "Net Income Avi to Common", "Diluted EPS",
    "Quarterly Earnings Growth", "Total Cash", "Total Cash Per Share", "Total Debt", "Total Debt/Equity",
    "Current Ratio", "Book Value Per Share", "Operating Cash Flow", "Levered Free Cash Flow", "Beta",
    "50-Day Moving Average", "200-Day Moving Average", "Avg Vol (3 month)", "Shares Outstanding",
    "Float", "% Held by Insiders", "% Held by Institutions", "Shares Short", "Short Ratio",
    "Short % of Float", "Shares Short (prior month)"
]

def download_html(ticker):
    try:
        link = f"http://finance.yahoo.com/quote/{ticker.upper()}/key-statistics"
        resp = requests.get(link)
        save_path = f"forward/{ticker}.html"
        with open(save_path, "w") as file:
            file.write(resp.text)
    except Exception as e:
        print(f"{ticker}: {str(e)}")
        time.sleep(2)

def check_yahoo():
    if not os.path.exists("forward/"):
        os.makedirs("forward/")
    
    ticker_list = [ticker for ticker in os.listdir(statspath) if ticker != ".DS_Store"]

    with ThreadPoolExecutor(max_workers=10) as executor:
        list(tqdm(executor.map(download_html, ticker_list), desc="Download progress:", unit="tickers"))

def parse_html(tickerfile, regex_patterns):
    ticker = tickerfile.split(".html")[0].upper()
    with open(f"forward/{tickerfile}", "r") as file:
        source = file.read().replace(",", "")
    
    value_list = []
    for variable, regex in regex_patterns.items():
        match = re.search(regex, source)
        if match:
            value_list.append(data_string_to_float(match.group(1)))
        else:
            value_list.append(np.nan)
    
    return [0, 0, ticker, 0, 0, 0, 0] + value_list

def forward():
    df_columns = [
        "Date", "Unix", "Ticker", "Price", "stock_p_change", "SP500", "SP500_p_change"
    ] + features

    regex_patterns = {
        feature: re.compile(rf">{re.escape(feature)}.*?(\-?\d+\.*\d*K?M?B?|N/A|NaN)%?(</td>|</span>)", re.DOTALL)
    for feature in features}

    tickerfile_list = [tickerfile for tickerfile in os.listdir("forward/") if tickerfile != ".DS_Store"]

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(tqdm(executor.map(parse_html, tickerfile_list, [regex_patterns]*len(tickerfile_list)), desc="Parsing progress:", unit="tickers"))

    df = pd.DataFrame(results, columns=df_columns)
    return df

if __name__ == "__main__":
    check_yahoo()
    current_df = forward()
    current_df.to_csv("forward_sample.csv", index=False)
