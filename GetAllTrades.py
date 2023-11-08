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
from polygon import exceptions
from google.cloud.storage import Blob
from google.cloud import storage
from google.auth.exceptions import DefaultCredentialsError
import requests
from google.api_core.exceptions import TooManyRequests

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

#day = '31'
#month = '08'


agMins = 2
stkName = 'SPY'
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
        final = [[i.price,i.size,i.participant_timestamp, i.exchange] for i in AllTrade]
        '''
        with open('GetAllTrades'+stkName+'.csv', "w", newline='',encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(final)
        '''    
        gclient = storage.Client(project="stockapp-401615")
        bucket = gclient.get_bucket("stockapp-storage")
        blob = Blob('GetAllTrades'+stkName, bucket) 
        blob.upload_from_string(json.dumps(final))
            
        
    except(exceptions.BadResponse, requests.exceptions.ReadTimeout, DefaultCredentialsError, requests.exceptions.ConnectionError, TooManyRequests):
        continue
