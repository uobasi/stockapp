# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 10:14:08 2023

@author: UOBASUB
"""

from polygon import RESTClient
client = RESTClient("udC9OULShUppFf4EF9UvLMLgHYW7wyCG")
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
import bisect
import json
from polygon import exceptions
import urllib3
from google.cloud.storage import Blob
from google.cloud import storage
from google.auth.exceptions import DefaultCredentialsError
import calendar
from google.api_core.exceptions import TooManyRequests
from google.api_core.exceptions import RetryError



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

day = '13'
#month = '08'



agMins = 2
stkName = 'SPY'
aggs = []  
Tmins = datetime.now()
global fullS
fullS = []
fft = []
cstat = []


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
        
        if datetime.now() > (Tmins + timedelta(minutes=.00)):
            if len(fullS) == 0:
                tryy = [i for i in df['time'].values[df['time'].searchsorted('09:00:00'):len(df['time'].values)]]
                tryy2 = [i for i in df['timestamp'].values[df['time'].searchsorted('09:00:00'):len(df['time'].values)]]
                tsm = [[tryy[i],tryy2[i]] for i in range(len(tryy))]
                
            elif stkName not in [i[10] for i in fullS]:
                tryy = [i for i in df['time'].values[df['time'].searchsorted('09:00:00'):len(df['time'].values)]]
                tryy2 = [i for i in df['timestamp'].values[df['time'].searchsorted('09:00:00'):len(df['time'].values)]]
                tsm = [[tryy[i],tryy2[i]] for i in range(len(tryy))]
                
            else:
                indx = max(index for index, item in enumerate(fullS) if item[10] == stkName)    
                tryy = [i for i in df['time'].values[df['time'].searchsorted(fullS[indx][3]):len(df['time'].values)]]
                tryy2 = [i for i in df['timestamp'].values[df['time'].searchsorted(fullS[indx][3]):len(df['time'].values)]]
                tsm = [[tryy[i],tryy2[i]] for i in range(len(tryy))]
            
            for vv in tsm:
                btet = list(client.list_trades(stkName, timestamp_gte=int((str(vv[1]) + '000000')), timestamp_lte=int((str(vv[1]+(60000*agMins)) + '000000')), order='asc', limit=50000))
                qtet = list(client.list_quotes(stkName, timestamp_gte=int((str(vv[1]-18000) + '000000')), timestamp_lte=int((str(vv[1]+(60000*agMins)) + '000000')), order='asc', limit=50000))
                enum = ['Bid(SELL)','BelowBid(SELL)','Ask(BUY)','AboveAsk(BUY)','Between']
                enum2 = ['Ask(BUY)','Bid(SELL)','Mid']
            
                minQut = {}
                Bidd = 0
                BelowBid = 0
                Askk = 0
                AboveAsk = 0
                Between = 0
                tradDic = {}
                
                for i in qtet:
                    if i.participant_timestamp is not None:
                        minQut[i.participant_timestamp] = [i.bid_price, i.ask_price]
                    elif i.participant_timestamp is  None:
                        minQut[i.sip_timestamp] = [i.bid_price, i.ask_price]
                
                quo = [i for i in minQut]   
                
                #minQut = [dict(i.participant_timestamp, elem=[i.bid_price, i.ask_price]) for i in qtet]
                
                
                
                for trades in range(len(btet)):
                    try: 
                        indx = bisect.bisect_left(quo, btet[trades].participant_timestamp)
                        if quo[indx] == btet[trades].participant_timestamp:
                            ff = quo[indx]
                        else:
                            ff = quo[indx-1]
                        #v = minQut[ff]
                    except(IndexError):
                        ff = quo[indx-1]
                    v = minQut[ff]
                    
                    '''
                    if btet[trades].price not in tradDic:
                        tradDic[btet[trades].price] = {'ASK':0,
                                                    'BID':0,
                                                    'Between':0, }
                    '''
                    if btet[trades].price == v[1]:
                        Askk+= btet[trades].size
                        #tradDic[btet[trades].price]['ASK'] += btet[trades].size
                    elif btet[trades].price > v[1]:
                        AboveAsk+= btet[trades].size
                        #tradDic[btet[trades].price]['ASK'] += btet[trades].size
                    elif btet[trades].price == v[0]:
                        Bidd+= btet[trades].size
                        #tradDic[btet[trades].price]['BID'] += btet[trades].size
                    elif btet[trades].price < v[0]:
                        BelowBid+= btet[trades].size 
                        #tradDic[btet[trades].price]['BID'] += btet[trades].size
                    elif v[0] < btet[trades].price < v[1]:
                        Between+= btet[trades].size
                        #tradDic[btet[trades].price]['Between'] += btet[trades].size
                        '''
                        dif = abs(btet[trades].price - v[1])
                        dif1 = abs(btet[trades].price - v[0])
                        
                        if dif < dif1:
                            #Askk+= btet[trades].size
                            tradDic[btet[trades].price]['ASK'] += btet[trades].size
                            
                        elif dif1 < dif:
                            #Bidd+= btet[trades].size
                            tradDic[btet[trades].price]['BID'] += btet[trades].size
                        
                        elif dif1 == dif:
                            #Between+= btet[trades].size
                            tradDic[btet[trades].price]['Between'] += btet[trades].size
                        '''
                lsr = [Bidd,BelowBid,Askk,AboveAsk,Between,]
                ind = lsr.index(max(lsr))
                '''
                pList = [i for i in tradDic]
                pList.sort()
                hist, bin_edges = np.histogram(pList, bins=10)
                
                newTdDict = {}
                
                for i in range(len(hist)):
                    newTdDict[bin_edges[i]] = {'ASK':0,
                                                'BID':0,
                                                'Between':0, }
                    
                    for j in pList:
                        if bin_edges[i] <= j <= bin_edges[i+1]:
                            newTdDict[bin_edges[i]]['ASK'] += tradDic[j]['ASK']
                            newTdDict[bin_edges[i]]['BID'] += tradDic[j]['BID']
                            newTdDict[bin_edges[i]]['Between'] += tradDic[j]['Between']
                        elif j > bin_edges[i+1]:
                            break
                
                temp = []
                dList = [i for i in newTdDict]
                dList.sort()
                for pp in dList:
                    ltemp = [round(pp,3), newTdDict[pp]['ASK'], newTdDict[pp]['BID'], newTdDict[pp]['Between']]
                    try:
                        temp.append(ltemp+[enum2[ltemp[1:].index(max(ltemp[1:]))]]+[round(ltemp[1:][ltemp[1:].index(max(ltemp[1:]))]/sum(ltemp[1:]),2)])
                    except(ZeroDivisionError):
                        temp.append(ltemp)
                    
                askcountp = 0
                bidcountp = 0
                midcountp = 0
                dct = 0
                
                
                
                for bb in temp:
                    askcountp += bb[1]
                    bidcountp += bb[2]
                    midcountp += bb[3]
                    
                    sms = sum([askcountp,bidcountp,midcountp])
                    
                    
                sstr = 'None'
                chk = [round(askcountp/sms,3),round(bidcountp/sms,3),round(midcountp/sms,3)]
                
                #for u in chk:
                    #if u >= 0.57:
                        #sstr = 'Imbalance'
                        
                    
                txt2 = "Ask:{askcount}  Bid:{bidcount}  Mid:{midcount}".format(askcount=round(askcountp/sms,3), bidcount=round(bidcountp/sms,3), midcount=round(midcountp/sms,3))     
                #txt = "Ask:{askcount}  Bid:{bidcount}  Mid:{midcount}".format(askcount=round(askcount/dct,3), bidcount=round(bidcount/dct,3), midcount=round(midcount/dct,3))        
                
                cstat.append(temp[::-1]+[txt2])
                
                #lsr+=vv
                #import json
                #data = json.loads(fft[0][12]) 
                '''
                fullS.append([round(lsr[ind]/sum(lsr),3), enum[ind], lsr[ind],]+vv+lsr+[stkName])
                #fft.append([round(lsr[ind]/sum(lsr),2), enum[ind], lsr[ind],]+vv+lsr+[df['time'].searchsorted(vv[0])]+[stkName])
                print([round(lsr[ind]/sum(lsr),3), enum[ind], lsr[ind],]+vv+lsr+[stkName])
                #if round(lsr[ind]/sum(lsr),3) >= 0.46 and lsr[ind] >= 210000: #or (lsr[ind] >= 160000 and round(lsr[ind]/sum(lsr),3) >= 0.650):
                if [round(lsr[ind]/sum(lsr),2), enum[ind], lsr[ind],]+vv+lsr+[df['time'].searchsorted(vv[0])]+[stkName] not in fft:
                #if int(str(time.time()).replace('.', '')  + '00') >=  int(str(vv[1]+(60000*agMins)) + '000000'):
                    fft.append([round(lsr[ind]/sum(lsr),2), enum[ind], lsr[ind],]+vv+lsr+[df['time'].searchsorted(vv[0])]+[stkName]) #+[cstat[len(cstat)-1]]+[sstr]
                    newFFT = []
                    for i in fft[::-1]:
                        if i[3] not in [i[3] for i in newFFT]:
                            i[4] = int(i[4])
                            i[10] = int(i[10])
                            newFFT.append(i)
                    '''
                    with open('dailyCandle'+stkName+'.csv', "w", newline='',encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerows(newFFT[::-1]) 
                    '''    
                    gclient = storage.Client(project="stockapp-401615")
                    bucket = gclient.get_bucket("stockapp-storage")
                    blob = Blob('dailyCandle'+stkName, bucket) 
                    blob.upload_from_string(json.dumps(newFFT[::-1]))
    except(exceptions.BadResponse, urllib3.exceptions.MaxRetryError, DefaultCredentialsError, TooManyRequests, RetryError):
        continue
        

                    
                    
                    
                
