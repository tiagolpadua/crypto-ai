
# Imports
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow import DAG
from datetime import datetime, timedelta
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import requests
import csv
import zipfile
import urllib.request
import io
# files = [
#     'BTCUSDT_2023_minute-sample.csv',
#     'BTCUSDT_2020_minute-sample.csv'
# ]
# files = [
#     '../btcdata/Binance_BTCUSDT_2020_minute.csv',
#     '../btcdata/Binance_BTCUSDT_2021_minute.csv',
#     '../btcdata/Binance_BTCUSDT_2022_minute.csv',
#     '../btcdata/Binance_BTCUSDT_2023_minute.csv'
# ]
# 'https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2023-09.zip'
BASE_URL = 'https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/'
START_MONTH = 8
START_YEAR = 2017
# END_MONTH = 1
# END_YEAR = 2018          
END_MONTH = 9
END_YEAR = 2023

def get_data():
    data_without_header = []
    curr_month = START_MONTH
    curr_year = START_YEAR
    while True:
        print('curr_month', curr_month)
        print('curr_year', curr_year)
        url = 'https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-' + str(curr_year) + '-' + f"{curr_month:02}" + '.zip'
        print(url)
        filehandle, _ = urllib.request.urlretrieve(url)
        zip_file_object = zipfile.ZipFile(filehandle, 'r')
        first_file = zip_file_object.namelist()[0]
        file = zip_file_object.open(first_file)
        content = io.TextIOWrapper(file, encoding='UTF-8', newline='')
        cr = csv.reader(content, delimiter=',')
        data_without_header.extend(list(cr))
        curr_month += 1
        if (curr_month > 12):
            curr_month = 1
            curr_year += 1
        if (curr_month > END_MONTH and curr_year >= END_YEAR):
            break
    return data_without_header
def transform_data(quotes):
    now = datetime.now().strftime("%Y_%m_%d_-%H_%M_%S")
    open_time = []
    open = []
    high = []
    low = []
    close = []
    volume = []
    close_time = []
    quote_volume = []
    count = []
    taker_buy_volume = []
    taker_buy_quote_volume = []
    for element in quotes:
        open_time.append(int(element[0]))
        open.append(float(element[1])) 
        high.append(float(element[2])) 
        low.append(float(element[3])) 
        close.append(float(element[4])) 
        volume.append(float(element[5]))
        close_time.append(int(element[6]))
        quote_volume.append(float(element[7]))
        count.append(int(element[8]))
        taker_buy_volume.append(float(element[9]))
        taker_buy_quote_volume.append(float(element[10]))
    d = {
        "open_time": open_time,
        "open": open,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "close_time": close_time,
        "quote_volume": quote_volume,
        "count": count,
        "taker_buy_volume": taker_buy_volume,
        "taker_buy_quote_volume": taker_buy_quote_volume
    }
    df = pd.DataFrame(data=d)
    df.sort_values(by=['open_time'])
    print(df)
    table = pa.Table.from_pandas(df)
    file_name = "data_" + now + ".parquet"
    pq.write_table(table, file_name)
d = get_data()
transform_data(d)