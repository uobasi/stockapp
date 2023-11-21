# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 21:16:37 2023

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
from polygon import exceptions
import json
import requests
import urllib3
from google.cloud.storage import Blob
from google.cloud import storage
from google.auth.exceptions import DefaultCredentialsError
import calendar
from google.api_core.exceptions import RetryError
from google.api_core.exceptions import TooManyRequests


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

day = '07'
month = '08'


agMins = 2
stkName = 'QQQ'
aggs = []  
Tmins = datetime.now()
global fullS
fullS = []
fft = []

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
        
        closePrice = int(df['close'][df.index[-1]])

        callPriceList = [closePrice-i for i in range(0,40)]  +  [closePrice+i for i in range(1,40)]
        
        bigCallOrders  = []
        
        for x in callPriceList:
            tempName = 'O:'+stkName+year[2:] + month + day +'C00'+str(x)+'000'
            #print(tempName)
            for i in list(client.list_trades(tempName, timestamp_gte=int((str(bTime)  + '000000')), timestamp_lte=int((str(df['timestamp'].iloc[-1]+(60000*agMins)) + '000000')), order='asc', limit=50000)):
                if round(((i.price*100)*i.size),1) >= 20000:#i.size >= 100:
                    hourss = datetime.fromtimestamp(int(i.sip_timestamp // 1000000000)).hour
                    if hourss < 10:
                        hourss = '0'+str(hourss)
                    minss = datetime.fromtimestamp(int(i.sip_timestamp // 1000000000)).minute
                    if minss < 10:
                        minss = '0'+str(minss)
                    secs = datetime.fromtimestamp(int(i.sip_timestamp // 1000000000)).second
                    if secs < 10:
                        secs = '0'+str(secs)
                    opttimeStamp = str(hourss) + ':' + str(minss) + ':'+ str(secs)
                    
                    tradeType = ''
                    '''
                    for t in client.list_quotes(tempName, timestamp_lte = i.sip_timestamp, limit=1):  
                        #if i.exchange == t.ask_exchange:
                        break
                    
                    #if i.exchange == t.ask_exchange:
                    if i.price == t.ask_price:
                        tradeType = 'A(BUY)'
                    elif i.price > t.ask_price:
                        tradeType = 'AA(BUY)'
                    elif i.price == t.bid_price:
                        tradeType = 'B(SELL)'
                    elif i.price < t.bid_price:
                        tradeType = 'BB(SELL)'
                    elif t.bid_price < i.price < t.ask_price:
                        tradeType = 'BBA'
                    i
                    '''
                    bigCallOrders.append([0, round((i.price*100)*i.size,1), i.sip_timestamp, 'C'+str(x), int(df['time'].searchsorted(opttimeStamp)),opttimeStamp, tradeType, i.exchange])#, t.ask_exchange, t.sip_timestamp])
                    
                            
        total = bigCallOrders
        #total.sort(key=lambda x:int(x[2]))  
        '''
        with open('OptionTrackerCall'+stkName+'.csv', "w", newline='',encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(total)
        '''
            
        gclient = storage.Client(project="stockapp-401615")
        bucket = gclient.get_bucket("stockapp-storage")
        blob = Blob('OptionTrackerCall'+stkName, bucket) 
        blob.upload_from_string(json.dumps(total))
            
        #time.sleep(5)
    except(exceptions.BadResponse, urllib3.exceptions.MaxRetryError, DefaultCredentialsError, RetryError, TooManyRequests):
        continue
        
