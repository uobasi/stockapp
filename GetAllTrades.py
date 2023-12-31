# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 00:11:31 2023

@author: UOBASUB
"""


from polygon import RESTClient
client = RESTClient("udC9OULShUppFf4EF9UvLMLgHYW7wyCG")
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
import csv  
import time
import bisect
import json
import urllib3
from polygon import exceptions
from google.cloud.storage import Blob
from google.cloud import storage
from google.auth.exceptions import DefaultCredentialsError
import requests
from google.api_core.exceptions import TooManyRequests
from google.api_core.exceptions import RetryError
import calendar
import sys

def getAllTrades(stkName=str(sys.argv[1])):
    if date(date.today().year, date.today().month, date.today().day).weekday() >= 5:
        lastFriday = date.today()
        oneday = timedelta(days=1)

        while lastFriday.weekday() != calendar.FRIDAY:
            lastFriday -= oneday
        
        year = str(lastFriday.year)
        month = lastFriday.month
        if month < 10:
            month = '0'+str(month)
        else:
            month = str(month)
        day = lastFriday.day
        if day < 10:
            day = '0'+str(day)
        else:
            day = str(day)
    else:
        year = str(date.today().year)
        month = date.today().month
        if month < 10:
            month = '0'+str(month)
        else:
            month = str(month)
        day = date.today().day
        if day < 10:
            day = '0'+str(day)
        else:
            day = str(day)



    '''
    day = '10'
    month = '08'
    year = '2023'
    '''

    #day = '08'
    #month = '08'


    agMins = 2
    #stkName = 'IWM'
    aggs = []  
    global AllTrade
    AllTrade = []
    prevAllTrade = []

    while True: 
        try:
            aggs = [] 
            for vv in client.get_aggs(stkName, agMins, 'minute', year+'-'+month+'-'+day, year+'-'+month+'-'+day):
                hourss = datetime.fromtimestamp(int(vv.timestamp/1000)).hour
                if hourss < 10:
                    hourss = '0'+str(hourss)
                minss = datetime.fromtimestamp(int(vv.timestamp/1000)).minute
                if minss < 10:
                    minss = '0'+str(minss)
                opttimeStamp = str(hourss) + ':' + str(minss) + ':00'
                if int(hourss) < 16:
                    aggs.append([vv.open, vv.high, vv.low, vv.close, vv.volume, opttimeStamp, vv.timestamp, stkName,])
                
            
            df = pd.DataFrame(aggs, columns = ['open', 'high', 'low', 'close', 'volume', 'time', 'timestamp', 'name',]) 
            
            
            bTime = df['timestamp'][df[df['time']==df['time'].iloc[0]].index.values[0]]
            AllTrade = list(client.list_trades(stkName, timestamp_gte=int((str(bTime)  + '000000')), timestamp_lte=int((str(df['timestamp'].iloc[-1]+(60000*agMins)) + '000000')), order='asc', limit=50000))
            final = sorted([[i.price,i.size,i.participant_timestamp, i.exchange] for i in AllTrade], key=lambda d: d[2], reverse=False) 

            gclient = storage.Client(project="stockapp-401615")
            bucket = gclient.get_bucket("stockapp-storage")
            blob = Blob('GetAllTrades'+stkName, bucket) 
            blob.upload_from_string(json.dumps(final))
                
            
        except(exceptions.BadResponse, exceptions.NoResultsError, requests.exceptions.ReadTimeout, DefaultCredentialsError, requests.exceptions.ConnectionError, TooManyRequests, urllib3.exceptions.MaxRetryError, RetryError):
            continue

if __name__ == '__main__':
    getAllTrades()
