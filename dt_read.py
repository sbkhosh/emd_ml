#!/usr/bin/python3

import backtrader as bt
import numpy as np
import os
import pandas as pd
import yaml

from dt_help import Helper
from pandas.tseries.offsets import BDay
from yahoofinancials import YahooFinancials

class DataProcessor():
    def __init__(self, input_directory, output_directory, input_prm_file):
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.input_prm_file = input_prm_file

    def __repr__(self):
        return(f'{self.__class__.__name__}({self.input_directory!r}, {self.output_directory!r}, {self.input_prm_file!r})')

    def __str__(self):
        return('input directory = {}, output directory = {}, input parameter file  = {}'.\
               format(self.input_directory, self.output_directory, self.input_prm_file))

    @Helper.timing
    def read_prm(self):
        filename = os.path.join(self.input_directory,self.input_prm_file)
        with open(filename) as fnm:
            self.conf = yaml.load(fnm, Loader=yaml.FullLoader)
        self.start_date = self.conf.get('start_date')
        self.end_date = self.conf.get('end_date')
        self.ticker = self.conf.get('ticker')

    @Helper.timing
    def process(self):
        start_date = self.start_date
        end_date = self.end_date
        ticker = self.ticker
        
        self.values = DataProcessor.load_data(start_date,end_date,ticker)

    @Helper.timing
    def load_data(start_date,end_date,ticker):
        date_range = pd.bdate_range(start=start_date,end=(pd.to_datetime(end_date).date() + BDay(-1)).strftime('%Y-%m-%d'))
        values = pd.DataFrame({'Dates': date_range})
        values['Dates']= pd.to_datetime(values['Dates'])

        raw_data = YahooFinancials(ticker)
        raw_data = raw_data.get_historical_price_data(start_date, end_date, "daily")
        df = pd.DataFrame(raw_data[ticker]['prices'])[['formatted_date','open','high','low','close','volume']]
        df.columns = ['Dates1','open','high','low','close','volume']
        df['Dates1']= pd.to_datetime(df['Dates1'])
        values = values.merge(df,how='left',left_on='Dates',right_on='Dates1')
        values = values.drop(labels='Dates1',axis=1)

        values = values.fillna(method="ffill",axis=0)
        values = values.fillna(method="bfill",axis=0)            
        cols = values.columns.drop('Dates')
        values[cols] = values[cols].apply(pd.to_numeric,errors='coerce').round(decimals=3)
        values.set_index('Dates',inplace=True)
        return(values)
