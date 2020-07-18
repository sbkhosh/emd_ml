#!/usr/bin/python3

import backtrader as bt
import csv
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import os
import pandas as pd 
import warnings
import yaml

from datetime import datetime, timedelta
from dt_help import Helper
from dt_model import EMDAnalysis
from dt_read import DataProcessor
from pandas.plotting import register_matplotlib_converters

warnings.filterwarnings('ignore',category=FutureWarning)
pd.options.mode.chained_assignment = None
register_matplotlib_converters()

if __name__ == '__main__':
    obj_helper = Helper('data_in','conf_help.yml')
    obj_helper.read_prm()
    
    fontsize = obj_helper.conf['font_size']
    matplotlib.rcParams['axes.labelsize'] = fontsize
    matplotlib.rcParams['xtick.labelsize'] = fontsize
    matplotlib.rcParams['ytick.labelsize'] = fontsize
    matplotlib.rcParams['legend.fontsize'] = fontsize
    matplotlib.rcParams['axes.titlesize'] = fontsize
    matplotlib.rcParams['text.color'] = 'k'

    obj_reader = DataProcessor('data_in','data_out','conf_model.yml')
    obj_reader.read_prm()   
    obj_reader.process()

    obj_emd = EMDAnalysis(data = obj_reader.values)
    obj_emd.get_vars()
    obj_emd.get_imfs_hilbert_ts()
    print(obj_emd.imfs)
