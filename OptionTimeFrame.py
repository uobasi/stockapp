# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 02:04:41 2023

@author: UOBASUB
"""


from datetime import datetime, timedelta, date
import csv  
import time
import json
from google.cloud.storage import Blob
from google.cloud import storage
from google.auth.exceptions import DefaultCredentialsError
from google.api_core.exceptions import TooManyRequests
from google.api_core.exceptions import RetryError


times = ['09:30:00',
'09:32:00',
'09:34:00',
'09:36:00',
'09:38:00',
'09:40:00',
'09:42:00',
'09:44:00',
'09:46:00',
'09:48:00',
'09:50:00',
'09:52:00',
'09:54:00',
'09:56:00',
'09:58:00',
'10:00:00',
'10:02:00',
'10:04:00',
'10:06:00',
'10:08:00',
'10:10:00',
'10:12:00',
'10:14:00',
'10:16:00',
'10:18:00',
'10:20:00',
'10:22:00',
'10:24:00',
'10:26:00',
'10:28:00',
'10:30:00',
'10:32:00',
'10:34:00',
'10:36:00',
'10:38:00',
'10:40:00',
'10:42:00',
'10:44:00',
'10:46:00',
'10:48:00',
'10:50:00',
'10:52:00',
'10:54:00',
'10:56:00',
'10:58:00',
'11:00:00',
'11:02:00',
'11:04:00',
'11:06:00',
'11:08:00',
'11:10:00',
'11:12:00',
'11:14:00',
'11:16:00',
'11:18:00',
'11:20:00',
'11:22:00',
'11:24:00',
'11:26:00',
'11:28:00',
'11:30:00',
'11:32:00',
'11:34:00',
'11:36:00',
'11:38:00',
'11:40:00',
'11:42:00',
'11:44:00',
'11:46:00',
'11:48:00',
'11:50:00',
'11:52:00',
'11:54:00',
'11:56:00',
'11:58:00',
'12:00:00',
'12:02:00',
'12:04:00',
'12:06:00',
'12:08:00',
'12:10:00',
'12:12:00',
'12:14:00',
'12:16:00',
'12:18:00',
'12:20:00',
'12:22:00',
'12:24:00',
'12:26:00',
'12:28:00',
'12:30:00',
'12:32:00',
'12:34:00',
'12:36:00',
'12:38:00',
'12:40:00',
'12:42:00',
'12:44:00',
'12:46:00',
'12:48:00',
'12:50:00',
'12:52:00',
'12:54:00',
'12:56:00',
'12:58:00',
'13:00:00',
'13:02:00',
'13:04:00',
'13:06:00',
'13:08:00',
'13:10:00',
'13:12:00',
'13:14:00',
'13:16:00',
'13:18:00',
'13:20:00',
'13:22:00',
'13:24:00',
'13:26:00',
'13:28:00',
'13:30:00',
'13:32:00',
'13:34:00',
'13:36:00',
'13:38:00',
'13:40:00',
'13:42:00',
'13:44:00',
'13:46:00',
'13:48:00',
'13:50:00',
'13:52:00',
'13:54:00',
'13:56:00',
'13:58:00',
'14:00:00',
'14:02:00',
'14:04:00',
'14:06:00',
'14:08:00',
'14:10:00',
'14:12:00',
'14:14:00',
'14:16:00',
'14:18:00',
'14:20:00',
'14:22:00',
'14:24:00',
'14:26:00',
'14:28:00',
'14:30:00',
'14:32:00',
'14:34:00',
'14:36:00',
'14:38:00',
'14:40:00',
'14:42:00',
'14:44:00',
'14:46:00',
'14:48:00',
'14:50:00',
'14:52:00',
'14:54:00',
'14:56:00',
'14:58:00',
'15:00:00',
'15:02:00',
'15:04:00',
'15:06:00',
'15:08:00',
'15:10:00',
'15:12:00',
'15:14:00',
'15:16:00',
'15:18:00',
'15:20:00',
'15:22:00',
'15:24:00',
'15:26:00',
'15:28:00',
'15:30:00',
'15:32:00',
'15:34:00',
'15:36:00',
'15:38:00',
'15:40:00',
'15:42:00',
'15:44:00',
'15:46:00',
'15:48:00',
'15:50:00',
'15:52:00',
'15:54:00',
'15:56:00',
'15:58:00',
'16:00:00']

stkName = 'NVDA'
enum = ['MidPut', 'BuyPut', 'SellPut', 'MidCall', 'BuyCall', 'SellCall']

while True:  
    fList = []
    try:
        gclient = storage.Client(project="stockapp-401615")
        bucket = gclient.get_bucket("stockapp-storage")
        blob = Blob('OptionTrackerPut'+stkName, bucket) 
        OptionOrdersPut = json.loads(blob.download_as_text())
        '''
        with open('OptionTrackerPut'+stkName+'.csv', 'r', encoding="utf8") as f:
              reader = csv.reader(f)
              OptionOrdersPut = list(reader)
              
        
        with open('OptionTrackerCall'+stkName+'.csv', 'r', encoding="utf8") as f:
              reader = csv.reader(f)
              OptionOrdersCall = list(reader)
        '''
        gclient = storage.Client(project="stockapp-401615")
        bucket = gclient.get_bucket("stockapp-storage")
        blob = Blob('OptionTrackerCall'+stkName, bucket) 
        OptionOrdersCall = json.loads(blob.download_as_text())
        
        optionOrderList = OptionOrdersPut + OptionOrdersCall
        
              
        if len(optionOrderList) > 0:
            optionOrderList.sort(key=lambda x:int(x[2])) 
            oppDict = {}
            for i in times:
                oppDict[i] = [0,0]#[1,1,1,1,1,1,]#
            for i in optionOrderList:
                for x in range(len(times)-1):
                    try:                 
                        if datetime.strptime(i[5], '%H:%M:%S').time() >= datetime.strptime(times[x], '%H:%M:%S').time() and datetime.strptime(i[5], '%H:%M:%S').time() < datetime.strptime(times[x+1], '%H:%M:%S').time():
                            '''
                            if 'P' in i[3] and i[6] == 'BBA':
                                oppDict[times[x]][0] += int(float(i[1]))
                                
                            elif 'P' in i[3] and (i[6] == 'A(BUY)' or i[6] == 'AA(BUY)'):
                                oppDict[times[x]][1] += int(float(i[1]))
                                
                            elif 'P' in i[3] and (i[6] == 'B(SELL)' or i[6] == 'BB(SELL)'):
                                oppDict[times[x]][2] += int(float(i[1]))
                                
                            elif 'C' in i[3] and i[6] == 'BBA':
                                oppDict[times[x]][3] += int(float(i[1]))
                            
                            elif 'C' in i[3] and (i[6] == 'A(BUY)' or i[6] == 'AA(BUY)'):
                                oppDict[times[x]][4] += int(float(i[1]))
                                
                            elif 'C' in i[3] and (i[6] == 'B(SELL)' or i[6] == 'BB(SELL)'):
                                oppDict[times[x]][5] += int(float(i[1]))    
                            break
                            '''
                            if 'P' in i[3]:
                                oppDict[times[x]][0] += int(float(i[1]))
                            elif 'C' in i[3]:
                                oppDict[times[x]][1] += int(float(i[1]))
                            break
                            
                    except(IndexError):
                        pass
                
            
            for y in oppDict:
                if sum(oppDict[y]) > 0 or datetime.strptime(datetime.now().strftime("%H:%M:%S"), '%H:%M:%S').time() > datetime.strptime(y, '%H:%M:%S').time():
                    try:
                        fList.append([y,' (Put:'+str(round(oppDict[y][0] / sum(oppDict[y]),2))+'('+str(oppDict[y][0])+') | '+'Call:'+str(round(oppDict[y][1] / sum(oppDict[y]),2))+'('+str(oppDict[y][1])+') ',oppDict[y][0],oppDict[y][1]])
                    except(ZeroDivisionError):
                        fList.append([y,' (Put:'+str(0)+'('+str(oppDict[y][0])+') | '+'Call:'+str(0)+'('+str(oppDict[y][1])+') ',oppDict[y][0],oppDict[y][1]])

                        
            #fList = [[y,' (Put:'+str(round(oppDict[y][0] / sum(oppDict[y]),2))+'('+str(oppDict[y][0])+') | '+'Call:'+str(round(oppDict[y][1] / sum(oppDict[y]),2))+'('+str(oppDict[y][1])+') ',oppDict[y][0],oppDict[y][1]] for y in oppDict if sum(oppDict[y]) > 0 or  ]#
#            ,oppDict[y][2],oppDict[y][3],oppDict[y][4],oppDict[y][5]
            '''
            with open('OptionTimeFrame'+stkName+'.csv', "w", newline='',encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(fList) 
            '''
            
            gclient = storage.Client(project="stockapp-401615")
            bucket = gclient.get_bucket("stockapp-storage")
            blob = Blob('OptionTimeFrame'+stkName, bucket) 
            blob.upload_from_string(json.dumps(fList))
            
    except(FileNotFoundError,IndexError,ValueError,DefaultCredentialsError,TooManyRequests,RetryError):
        #print('errr')
        continue

'''
for i in fList:
    lsr = i[2:]
    ind = lsr.index(max(lsr))
    print(i,enum[ind], round(lsr[ind]/sum(lsr),2))   
'''        
