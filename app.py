# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 17:24:16 2023

@author: UOBASUB
"""

from scipy import signal
from scipy.misc import derivative
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta, date
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
np.seterr(divide='ignore', invalid='ignore')
pd.options.mode.chained_assignment = None
from scipy.signal import argrelextrema
from polygon import RESTClient, exceptions
client = RESTClient("udC9OULShUppFf4EF9UvLMLgHYW7wyCG")
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import csv
import bisect 
import subprocess
import sys
import atexit
import json
import calendar
from google.cloud.storage import Blob
from google.cloud import storage
dailyCandle = subprocess.Popen([sys.executable,'dailyCandle.py'])
OptionsTrack = subprocess.Popen([sys.executable,'OptionsTrack.py'])
OptionTrackerCall = subprocess.Popen([sys.executable,'OptionTrackerCall.py'])
GetAllTrades = subprocess.Popen([sys.executable,'GetAllTrades.py'])
OptionTimeFrame = subprocess.Popen([sys.executable,'OptionTimeFrame.py'])
global allProcess 
allProcess = [dailyCandle,OptionsTrack,OptionTrackerCall,GetAllTrades,OptionTimeFrame]#

def killAll():
    for i in allProcess:
        i.kill()
    print('All Process killed')
        
#atexit.register(killAll)


def ema(df):
    df['30ema'] = df['close'].ewm(span=30, adjust=False).mean()
    df['40ema'] = df['close'].ewm(span=40, adjust=False).mean()
    df['28ema'] = df['close'].ewm(span=28, adjust=False).mean()
    df['50ema'] = df['close'].ewm(span=50, adjust=False).mean()
    df['1ema'] = df['close'].ewm(span=1, adjust=False).mean()

def vwap(df):
    v = df['volume'].values
    h = df['high'].values
    l = df['low'].values
    # print(v)
    df['vwap'] = np.cumsum(v*(h+l)/2) / np.cumsum(v)
    #df['disVWAP'] = (abs(df['close'] - df['vwap']) / ((df['close'] + df['vwap']) / 2)) * 100
    #df['disVWAPOpen'] = (abs(df['open'] - df['vwap']) / ((df['open'] + df['vwap']) / 2)) * 100
    #df['disEMAtoVWAP'] = ((df['close'].ewm(span=12, adjust=False).mean() - df['vwap'])/df['vwap']) * 100

    df['volumeSum'] = df['volume'].cumsum()
    df['volume2Sum'] = (v*((h+l)/2)*((h+l)/2)).cumsum()
    #df['myvwap'] = df['volume2Sum'] / df['volumeSum'] - df['vwap'].values * df['vwap']
    #tp = (df['low'] + df['close'] + df['high']).div(3).values
    # return df.assign(vwap=(tp * v).cumsum() / v.cumsum())


def sigma(df):
    try:
        val = df.volume2Sum / df.volumeSum - df.vwap * df.vwap
    except(ZeroDivisionError):
        val = df.volume2Sum / (df.volumeSum+0.000000000001) - df.vwap * df.vwap
    return math.sqrt(val) if val >= 0 else val


def PPP(df):

    df['STDEV_TV'] = df.apply(sigma, axis=1)
    stdev_multiple_0 = 0.50
    stdev_multiple_1 = 1
    stdev_multiple_1_5 = 1.5
    stdev_multiple_2 = 2.00
    stdev_multiple_25 = 2.50

    df['STDEV_0'] = df.vwap + stdev_multiple_0 * df['STDEV_TV']
    df['STDEV_N0'] = df.vwap - stdev_multiple_0 * df['STDEV_TV']

    df['STDEV_1'] = df.vwap + stdev_multiple_1 * df['STDEV_TV']
    df['STDEV_N1'] = df.vwap - stdev_multiple_1 * df['STDEV_TV']
    
    df['STDEV_15'] = df.vwap + stdev_multiple_1_5 * df['STDEV_TV']
    df['STDEV_N15'] = df.vwap - stdev_multiple_1_5 * df['STDEV_TV']

    df['STDEV_2'] = df.vwap + stdev_multiple_2 * df['STDEV_TV']
    df['STDEV_N2'] = df.vwap - stdev_multiple_2 * df['STDEV_TV']
    
    df['STDEV_25'] = df.vwap + stdev_multiple_25 * df['STDEV_TV']
    df['STDEV_N25'] = df.vwap - stdev_multiple_25 * df['STDEV_TV']


    #df['disVWAPUpp'] = abs(((df['close'] - df['STDEV_1'])/df['STDEV_1']) * 100)
    #df['disVWAPUpp'] = (abs(df['close'] - df['STDEV_1']) / ((df['close'] + df['STDEV_1']) / 2)) * 100
    #df['disVWAPLow'] = abs(((df['close'] - df['STDEV_N1'])/df['STDEV_N1']) * 100)
    #df['disVWAPLow'] = (abs(df['close'] - df['STDEV_N1']) /((df['close'] + df['STDEV_N1']) / 2)) * 100

    #return[df['STDEV_1'][len(df['STDEV_1'])-1], df['STDEV_N1'][len(df['STDEV_N1'])-1]]




def VMA(df):
    df['vma'] = df['volume'].rolling(4).mean()
      
            


def plotChart(df, lst2, num1, num2, x_fake, df_dx, optionOrderList, stockName='', prevdtstr:str='', sord:list=[], trends:list=[], lstVwap:list=[], bigOrders:list=[], pea:bool=False, timeStamp:int=None, previousDay:bool=False, OptionTimeFrame:list=[], overall:list=[]):
  
    average = round(np.average(df_dx), 3)
    now = round(df_dx[len(df_dx)-1], 3)
    if average > 0:
        strTrend = "Uptrend"
    elif average < 0:
        strTrend = "Downtrend"
    else:
        strTrend = "No trend!"
        
    sortadlist = lst2[1]
    sortadlist2 = lst2[0]
    
    #Tbid = sum([i[6][0] + i[6][1] for i in sortadlist2])
    #Task = sum([i[6][2] + i[6][3] for i in sortadlist2])
    #Tmid = sum([i[6][4] for i in sortadlist2])
    #TBet = sum([i[6][4] for i in sortadlist2])
    
    #strTbid = str(round(Tbid/(Tbid+Task+Tmid),2))
    #strTask = str(round(Task/(Tbid+Task+Tmid),2))
    #strMid = str(round(Tmid/(Tbid+Task+Tmid),2))
    #strTBet = str(round(TBet/(Tbid+Task+TBet),2))
    
    
    putDec = 0
    CallDec = 0
    NumPut = sum([float(i[1]) for i in optionOrderList if 'P' in i[3]])
    NumCall = sum([float(i[1]) for i in optionOrderList if 'C' in i[3]])
    if len(optionOrderList) > 0:
        putDec = round(NumPut / sum([float(i[1]) for i in optionOrderList]),2)
        CallDec = round(NumCall / sum([float(i[1]) for i in optionOrderList]),2)
        
    
    NumPutHalf = 0
    NumCallHalf = 0
    putDecHalf = 0
    CallDecHalf = 0
    if len(optionOrderList) > 0:
        if datetime.fromtimestamp(int(int(optionOrderList[len(optionOrderList)-1][2]) // 1000000000)).hour >= 12:
            tindex = bisect.bisect_left([i[5] for i in optionOrderList], '12:30')
            
            NumPutHalf = sum([float(i[1]) for i in optionOrderList[tindex:] if 'P' in i[3]])
            NumCallHalf = sum([float(i[1]) for i in optionOrderList[tindex:] if 'C' in i[3]])
            if len(optionOrderList) > 0:
                putDecHalf = round(NumPutHalf / sum([float(i[1]) for i in optionOrderList[tindex-1:]]),2)
                CallDecHalf = round(NumCallHalf / sum([float(i[1]) for i in optionOrderList[tindex-1:]]),2)
        
    
    fig = make_subplots(rows=2, cols=3, shared_xaxes=False, shared_yaxes=False,
                        specs=[[{}, {"colspan": 1}, {"colspan": 1},],
                               [{}, {"colspan": 2,}, {}, ]],
                        horizontal_spacing=0.02, vertical_spacing=0.03, subplot_titles=(stockName + ' '+strTrend+'('+str(average)+')' +' (Put:'+str(putDec)+' ('+str(NumPut)+') | '+'Call:'+str(CallDec)+' ('+str(NumCall)+') '+ '<br>' 
                                                                                        +' ( Put:'+str(putDecHalf)+'('+str(NumPutHalf)+') | '+'Call:'+str(CallDecHalf)+'('+str(NumCallHalf)+') '
                                                                                        , 'Volume Profile', str(datetime.now())),
                         column_widths=[0.65,0.23,0.23], row_width=[0.30, 0.70,]) #row_width=[0.15, 0.85,],

    
            
    for pott in OptionTimeFrame:
        pott.append(df['time'].searchsorted(pott[0]))
        #print(pott)
        
    optColor = [     'red' if float(i[2]) > float(i[3])
                else 'green' if float(i[3]) > float(i[2])
                else 'gray' if float(i[3]) == float(i[2])
                else i for i in OptionTimeFrame]

    fig.add_trace(
        go.Bar(
            x=pd.Series([i[0] for i in OptionTimeFrame]),
            y=pd.Series([float(i[2]) if float(i[2]) > float(i[3]) else float(i[3]) if float(i[3]) > float(i[2]) else 0 for i in OptionTimeFrame]),
            #textposition='auto',
            #orientation='h',
            #width=0.2,
            marker_color=optColor,
            hovertext=pd.Series([i[0]+' '+i[1] for i in OptionTimeFrame]),
            
        ),
        row=2, col=1
    )
        
    fig.add_trace(
        go.Bar(
            x=pd.Series([i[0] for i in OptionTimeFrame]),
            y=pd.Series([float(i[3]) if float(i[2]) > float(i[3]) else float(i[2]) if float(i[3]) > float(i[2]) else 0 for i in OptionTimeFrame]),
            #textposition='auto',
            #orientation='h',
            #width=0.2,
            marker_color= [     'green' if float(i[2]) > float(i[3])
                        else 'red' if float(i[3]) > float(i[2])
                        else 'gray' if float(i[3]) == float(i[2])
                        else i for i in OptionTimeFrame],
            hovertext=pd.Series([i[0]+' '+i[1] for i in OptionTimeFrame]),
            
        ),
        row=2, col=1
    )

    pms = pd.Series([i[2] for i in OptionTimeFrame]).rolling(4).mean()
    cms = pd.Series([i[3] for i in OptionTimeFrame]).rolling(4).mean()
    fig.add_trace(go.Scatter(x=pd.Series([i[0] for i in OptionTimeFrame]), y=pms, line=dict(color='red'), mode='lines', name='Put VMA'), row=2, col=1)
    fig.add_trace(go.Scatter(x=pd.Series([i[0] for i in OptionTimeFrame]), y=cms, line=dict(color='green'), mode='lines', name='Call VMA'), row=2, col=1)
    
    #hovertext = []
    # for i in range(len(df)):
    #strr = df['time'][0]+'\n' +'Open: '+ str(df['open'[0]])+'\n'
    #hovertext.append(str(df.bidAskString[i])+' '+str(df.bidAsk[i]))

    fig.add_trace(go.Candlestick(x=df['time'],
                                 open=df['open'],
                                 high=df['high'],
                                 low=df['low'],
                                 close=df['close'],
                                 # hoverinfo='text',
                                 name="OHLC"),
                  row=1, col=1)
    
    
    
    #fig.add_trace(go.Bar(x=df['time'], y=df['volume'],showlegend=False), row=2, col=1)
    
    #fig.add_trace(go.Scatter(x=df['time'], y=df['vma'], mode='lines', name='VMA'), row=2, col=1)
    
    

    
    localMin = argrelextrema(df.close.values, np.less_equal, order=25)[0] 
    localMax = argrelextrema(df.close.values, np.greater_equal, order=25)[0]
     
    if len(localMin) > 0:
        mcount = 0 
        for p in localMin:
            fig.add_annotation(x=df['time'][p], y=df['close'][p],
                               text='<b>' + str(mcount) +'lMin' +  '</b>',
                               showarrow=True,
                               arrowhead=4,
                               font=dict(
                #family="Courier New, monospace",
                size=10,
                # color="#ffffff"
            ),)
            mcount+=1
    if len(localMax) > 0:
        mcount = 0 
        for b in localMax:
            fig.add_annotation(x=df['time'][b], y=df['close'][b],
                               text='<b>' + str(mcount) + 'lMax' +  '</b>',
                               showarrow=True,
                               arrowhead=4,
                               font=dict(
                #family="Courier New, monospace",
                size=10,
                # color="#ffffff"
            ),)
            mcount+=1
    
    '''
    fig.add_trace(go.Scatter(x=df['time'].iloc[df['time'].searchsorted('09:30:00'):] , y=pd.Series([round(i[2] / (i[2]+i[3]),2) for i in overall]), mode='lines',name='Put Volume'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['time'].iloc[df['time'].searchsorted('09:30:00'):] , y=pd.Series([round(i[3] / (i[2]+i[3]),2) for i in overall]), mode='lines',name='Call Volume'), row=2, col=1)

    
    
    '''
    fig.add_trace(go.Scatter(x=df['time'], y=df_dx, mode='lines',name='Derivative'), row=2, col=2)
    fig.add_hline(y=0, row=2, col=2, line_color='black')
        
    localDevMin = argrelextrema(df_dx, np.less_equal, order=60)[0] 
    localDevMax = argrelextrema(df_dx, np.greater_equal, order=60)[0]
    
    if len(localDevMin) > 0:
        for p in localDevMin:
            fig.add_annotation(x=x_fake[p], y=df_dx[p],
                               text='<b>' + 'lMin' + '</b>',
                               showarrow=True,
                               arrowhead=4,
                               font=dict(
                #family="Courier New, monospace",
                size=10,
                # color="#ffffff"
            ),row=2, col=2)
            
    if len(localDevMax) > 0:
        for b in localDevMax:
            fig.add_annotation(x=x_fake[b], y=df_dx[b],
                               text='<b>' + 'lMax' + '</b>',
                               showarrow=True,
                               arrowhead=4,
                               font=dict(
                #family="Courier New, monospace",
                size=10,
                # color="#ffffff"
            ),row=2, col=2)

    
    if pea:
        peak, _ = signal.find_peaks(df['50ema'])
        bottom, _ = signal.find_peaks(-df['50ema'])
    
        if len(peak) > 0:
            for p in peak:
                fig.add_annotation(x=df['time'][p], y=df['open'][p],
                                   text='<b>' + 'P' + '</b>',
                                   showarrow=True,
                                   arrowhead=4,
                                   font=dict(
                    #family="Courier New, monospace",
                    size=10,
                    # color="#ffffff"
                ),)
        if len(bottom) > 0:
            for b in bottom:
                fig.add_annotation(x=df['time'][b], y=df['open'][b],
                                   text='<b>' + 'T' + '</b>',
                                   showarrow=True,
                                   arrowhead=4,
                                   font=dict(
                    #family="Courier New, monospace",
                    size=10,
                    # color="#ffffff"
                ),)
    #fig.update_layout(title=df['name'][0])
    fig.update(layout_xaxis_rangeslider_visible=False)
    #lst2 = histor(df)

    

    #sPercent = sum([i[1] for i in adlist]) * .70
    #tp = valueAreaV1(lst2[0])
    

    fig.add_shape(type="rect",
                  y0=num1, y1=num2, x0=-1, x1=len(df),
                  fillcolor="crimson",
                  opacity=0.09,
                  )

    

    try:
        colors = ['rgba(255,0,0,'+str(round(i[6][i[6].index(max(i[6]))]/sum(i[6]),2))+')' if i[5] == 'red'  #'rgba(255,0,0,'+str(round(i[6][:4][i[6][:4].index(max(i[6][:4]))]/sum(i[6]),2))+')' if i[5] == 'red' 
                  else 'rgba(139,0,0,'+str(round(i[6][i[6].index(max(i[6]))]/sum(i[6]),2))+')'if i[5] == 'darkRed' 
                  else 'rgba(0,139,139,'+str(round(i[6][i[6].index(max(i[6]))]/sum(i[6]),2))+')'if i[5] == 'green' 
                  else 'rgba(0,104,139,'+str(round(i[6][i[6].index(max(i[6]))]/sum(i[6]),2))+')'if i[5] == 'darkGreen' 
                  else '#778899' if i[5] == 'black' 
                  else i for i in sortadlist2]#['darkcyan', ] * len(sortadlist2)#
    except(ZeroDivisionError):
        colors = [     'rgba(255,0,0)' if i[5] == 'red' 
                  else 'rgba(139,0,0)'if i[5] == 'darkRed' 
                  else 'rgba(0,139,139)'if i[5] == 'green' 
                  else 'rgba(0,104,139,)'if i[5] == 'darkGreen'
                  else '#778899' if i[5] == 'black' 
                  else i for i in sortadlist2]
        
    
    #print(colors)
    #colors[0] = 'crimson'
    fig.add_trace(
        go.Bar(
            x=pd.Series([i[1] for i in sortadlist2]),
            y=pd.Series([i[0] for i in sortadlist2]),
            text=np.around(pd.Series([i[0] for i in sortadlist2]), 2),
            textposition='auto',
            orientation='h',
            width=0.2,
            marker_color=colors,
            hovertext=pd.Series([str(i[6][0]) + ' ' + str(i[6][1]) + ' ' + str(i[6][2]) +' '+ str(i[6][3]) + ' ' + str(i[6][4]) +' '+ str(round(i[6][i[6].index(max(i[6]))]/sum(i[6]),2)) for i in sortadlist2]),
        ),
        row=1, col=2
    )
    
    


    fig.add_trace(go.Scatter(x=[sortadlist2[0][1], sortadlist2[0][1]], y=[
                  num1, num2],  opacity=0.5), row=1, col=2)
    
    
    #fig.add_trace(go.Scatter(x=x_fake, y=df_dx, mode='lines',name='Derivative'), row=2, col=2)
    
    fig.add_trace(go.Scatter(x=df['time'], y=df['vwap'], mode='lines', name='VWAP'))
    
    #if 2 in lstVwap:
    fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_2'], mode='lines', opacity=0.2, name='UPPERVWAP2'))
    fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_N2'], mode='lines', opacity=0.2, name='LOWERVWAP2'))
    #if 0 in lstVwap:
    fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_25'], mode='lines', opacity=0.2, name='UPPERVWAP2.5'))
    fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_N25'], mode='lines', opacity=0.2, name='LOWERVWAP2.5'))
    #if 1 in lstVwap:    
    fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_1'], mode='lines', opacity=0.2, name='UPPERVWAP1'))
    fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_N1'], mode='lines', opacity=0.2, name='LOWERVWAP1'))
        
    #if 1.5 in lstVwap:     
    fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_15'], mode='lines', opacity=0.2, name='UPPERVWAP1.5'))
    fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_N15'], mode='lines', opacity=0.2, name='LOWERVWAP1.5'))

    fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_0'], mode='lines', opacity=0.2, name='UPPERVWAP0.5'))
    fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_N0'], mode='lines', opacity=0.2, name='LOWERVWAP0.5'))

    #fig.add_trace(go.Scatter(x=df['time'], y=df['1ema'], mode='lines', opacity=0.19, name='1ema',marker_color='rgba(0,0,0)'))
    
    #fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_2'], mode='lines', name='UPPERVWAP'))
    
    '''
    fig.add_trace(go.Scatter(x=df['time'], y=df['close'], mode='lines',name='Close',marker_color='rgba(0,0,0)'))
    
    
    
    
    
    fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_15'], mode='lines', name='UPPERVWAP15'))
    fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_N15'], mode='lines', name='LOWERVWAP15'))
    '''
    '''
    fig.add_trace(
    go.Scatter(
        x=pd.Series([i[1] for i in linePlot]),
        y=pd.Series([i[0] for i in linePlot]),
    ),
    row=1, col=2)
    '''
    #fig.add_trace(go.Scatter(x=df['time'], y=df['1ema'], mode='lines', name='1ema') , row=1, col=1)
    #fig.add_trace(go.Scatter(x=df['time'], y=df['1ema'], mode='lines', name='1ema',line=dict(color="#000000")))
    fig.add_hline(y=df['close'][len(df)-1], row=1, col=2)
    
    
    #fig.add_hline(y=0, row=1, col=4)
    
 
    trcount = 0
    #indsBuy = [(len(df)-1,i[1]) if i[0] >= len(df) else i for i in [(i[10],i[1]) for i in sord if i[11] == stockName and i[1] == 'Ask(BUY)']]
    #indsSell = [(len(df)-1,i[1]) if i[0] >= len(df) else i for i in [(i[10],i[1]) for i in sord if i[11] == stockName and i[1] == 'Bid(SELL)']]
    #indsBetw = [(len(df)-1,i[1]) if i[0] >= len(df) else i for i in [(i[10],i[1]) for i in sord if i[11] == stockName and i[1] == 'Between']]
    putCand = [i for i in OptionTimeFrame if int(i[2]) > int(i[3]) if int(i[4]) < len(df)]
    callCand = [i for i in OptionTimeFrame if int(i[3]) > int(i[2]) if int(i[4]) < len(df)]
    #MidCand = [i for i in OptionTimeFrame if int(i[3]) == int(i[2]) if int(i[4]) < len(df)]
    indsAbove = [(len(df)-1,i[1]) if i[0] >= len(df) else i for i in [(int(i[10]),i[1]) for i in sord if i[11] == stockName and i[1] == 'AboveAsk(BUY)']]
    
    indsBelow = [(len(df)-1,i[1]) if i[0] >= len(df) else i for i in [(int(i[10]),i[1]) for i in sord if i[11] == stockName and i[1] == 'BelowBid(SELL)']]
    #imbalance = [(len(df)-1,i[1]) if i[0] >= len(df) else i for i in [(i[10],i[1]) for i in sord if i[11] == stockName and i[13] == 'Imbalance' and i[1] != 'BelowBid(SELL)' and i[1] != 'AboveAsk(BUY)']]
    #indsHAbove = [(len(df)-1,i[1]) if i[0] >= len(df) else i for i in [(i[10],i[1]) for i in sord if i[11] == stockName and i[1] == 'Ask(BUY)' and float(i[0]) >= 0.40 and int(i[2]) > 160000]]
    #indsHBelow  = [(len(df)-1,i[1]) if i[0] >= len(df) else i for i in [(i[10],i[1]) for i in sord if i[11] == stockName and i[1] == 'Bid(SELL)' and float(i[0]) >= 0.40 and int(i[2]) > 160000]]
    

    if len(optionOrderList) > 0:
        oppDict = {}
        for opp in optionOrderList:
            if opp[3] not in oppDict:
                oppDict[opp[3]] = int(float(opp[1]))
            else:
                oppDict[opp[3]] += int(float(opp[1]))
        
        newOpp = [[i,oppDict[i]] for i in oppDict ] #if oppDict[i] >= 100000
        newOpp.sort(key=lambda x:float(x[0][1:])) 

        for i in range(len(newOpp)-1):
            if newOpp[i][0][1:] == newOpp[i+1][0][1:]:
                total = newOpp[i+1][1]+newOpp[i][1]
                newOpp[i+1].append(round(newOpp[i+1][1]/total,3))
                newOpp[i].append(round(newOpp[i][1]/total,3))
            elif newOpp[i][0][1:] != newOpp[i+1][0][1:] : #and len(newOpp[i]) < 3
                newOpp[i].append(1)
        newOpp[len(newOpp)-1].append(1)
        
        '''
        if newOpp[len(newOpp)-1][0][1:] == newOpp[len(newOpp)-2][0][1:]:
            total = newOpp[len(newOpp)-1][1]+newOpp[len(newOpp)-2][1]
            newOpp[len(newOpp)-1].append(round(newOpp[len(newOpp)-1][1]/total,3))
            newOpp[len(newOpp)-2].append(round(newOpp[len(newOpp)-2][1]/total,3))
        '''
        fig.add_trace(
            go.Bar(
                x=pd.Series([i[1] for i in newOpp]),
                y=pd.Series([float(i[0][1:]) for i in newOpp]),
                text=pd.Series([i[0] for i in newOpp]),
                textposition='auto',
                orientation='h',
                #width=0.2,
                marker_color=[     'red' if 'P' in i[0] 
                            else 'green' if 'C' in i[0]
                            else i for i in newOpp],
                hovertext=pd.Series([i[0]  + ' ' + str(i[2]) for i in newOpp]),
            ),
            row=1, col=3
        )
        
    '''       
    if len(MidCand) > 0:
       fig.add_trace(go.Candlestick(
           x=[df['time'][i[4]] for i in MidCand],
           open=[df['open'][i[4]] for i in MidCand],
           high=[df['high'][i[4]] for i in MidCand],
           low=[df['low'][i[4]] for i in MidCand],
           close=[df['close'][i[4]] for i in MidCand],
           increasing={'line': {'color': 'gray'}},
           decreasing={'line': {'color': 'gray'}},
           hovertext=[i[1] for i in MidCand ],
           name='highlight' ),
       row=1, col=1)
       trcount+=1
    '''
    if len(putCand) > 0:
        fig.add_trace(go.Candlestick(
            x=[df['time'][i[4]] for i in putCand],
            open=[df['open'][i[4]] for i in putCand],
            high=[df['high'][i[4]] for i in putCand],
            low=[df['low'][i[4]] for i in putCand],
            close=[df['close'][i[4]] for i in putCand],
            increasing={'line': {'color': 'pink'}},
            decreasing={'line': {'color': 'pink'}},
            hovertext=[i[1] for i in putCand ],
            name='highlight' ),
        row=1, col=1)
        trcount+=1
        
    if len(callCand) > 0:
        fig.add_trace(go.Candlestick(
            x=[df['time'][i[4]] for i in callCand],
            open=[df['open'][i[4]] for i in callCand],
            high=[df['high'][i[4]] for i in callCand],
            low=[df['low'][i[4]] for i in callCand],
            close=[df['close'][i[4]] for i in callCand],
            increasing={'line': {'color': 'teal'}},
            decreasing={'line': {'color': 'teal'}},
            hovertext=[i[1] for i in callCand ],
            name='highlight' ),
        row=1, col=1)
        trcount+=1
    '''
    if len(indsBuy) > 0:
        fig.add_trace(go.Candlestick(
            x=[df['time'][i[0]] for i in indsBuy],
            open=[df['open'][i[0]] for i in indsBuy],
            high=[df['high'][i[0]] for i in indsBuy],
            low=[df['low'][i[0]] for i in indsBuy],
            close=[df['close'][i[0]] for i in indsBuy],
            increasing={'line': {'color': 'teal'}},
            decreasing={'line': {'color': 'teal'}},
            hovertext=[str(i[0])+' '+str(i[1])+' '+str(i[2])+'<br>'+i[12].replace('], ', '],<br>')+'<br>' for i in sord if i[11] == stockName  and i[1] == 'Ask(BUY)'],
            name='highlight' ),
        row=1, col=1)
        trcount+=1
    
    if len(indsSell) > 0:
        fig.add_trace(go.Candlestick(
            x=[df['time'][i[0]] for i in indsSell],
            open=[df['open'][i[0]] for i in indsSell],
            high=[df['high'][i[0]] for i in indsSell],
            low=[df['low'][i[0]] for i in indsSell],
            close=[df['close'][i[0]] for i in indsSell],
            increasing={'line': {'color': 'pink'}},
            decreasing={'line': {'color': 'pink'}},
            hovertext=[str(i[0])+' '+str(i[1])+' '+str(i[2])+'<br>'+i[12].replace('], ', '],<br>')+'<br>'   for i in sord if i[11] == stockName  and i[1] == 'Bid(SELL)'],
            name='highlight' ),
        row=1, col=1)
        trcount+=1
    
    if len(indsBetw) > 0:
        fig.add_trace(go.Candlestick(
            x=[df['time'][i[0]] for i in indsBetw],
            open=[df['open'][i[0]] for i in indsBetw],
            high=[df['high'][i[0]] for i in indsBetw],
            low=[df['low'][i[0]] for i in indsBetw],
            close=[df['close'][i[0]] for i in indsBetw],
            increasing={'line': {'color': '#778899'}, 'fillcolor': '#778899'},
            decreasing={'line': {'color': '#778899'}, 'fillcolor': '#778899'},
            hovertext=[str(i[0])+' '+str(i[1])+' '+str(i[2])+'<br>'+i[12].replace('], ', '],<br>')+'<br>'   for i in sord if i[11] == stockName  and i[1] == 'Between'],
            name='highlight' ),
        row=1, col=1)
        trcount+=1
    '''    
    if len(indsAbove) > 0:
        fig.add_trace(go.Candlestick(
            x=[df['time'][i[0]] for i in indsAbove],
            open=[df['open'][i[0]] for i in indsAbove],
            high=[df['high'][i[0]] for i in indsAbove],
            low=[df['low'][i[0]] for i in indsAbove],
            close=[df['close'][i[0]] for i in indsAbove],
            increasing={'line': {'color': '#00FFFF'}},
            decreasing={'line': {'color': '#00FFFF'}},
            hovertext=[str(i[0])+' '+str(i[1])+' '+str(i[2])+'<br>'  for i in sord if i[11] == stockName and i[1] == 'AboveAsk(BUY)'], #+i[12].replace('], ', '],<br>')+'<br>'
            name='highlight' ),
        row=1, col=1)
        trcount+=1
    
    if len(indsBelow) > 0:
        fig.add_trace(go.Candlestick(
            x=[df['time'][i[0]] for i in indsBelow],
            open=[df['open'][i[0]] for i in indsBelow],
            high=[df['high'][i[0]] for i in indsBelow],
            low=[df['low'][i[0]] for i in indsBelow],
            close=[df['close'][i[0]] for i in indsBelow],
            increasing={'line': {'color': '#FF1493'}},
            decreasing={'line': {'color': '#FF1493'}},
            hovertext=[str(i[0])+' '+str(i[1])+' '+str(i[2])+'<br>' for i in sord if i[11] == stockName and i[1] == 'BelowBid(SELL)'], #+i[12].replace('], ', '],<br>')+'<br>'
            name='highlight' ),
        row=1, col=1)
        trcount+=1
    
    #for ttt in trends[0]:
        #fig.add_shape(ttt, row=1, col=1)
    

    #fig.add_trace(go.Scatter(x=df['time'], y=df['2ema'], mode='lines', name='2ema'))
    

    fig.add_trace(go.Scatter(x=df['time'], y= [sortadlist2[0][0]]*len(df['time']) ,
                             line_color='orange',
                             text = 'Current Day POC',
                             textposition="bottom left",
                             showlegend=False,
                             visible=False,
                             mode= 'lines',
                            
                            ),
                  row=1, col=1
                 )
    '''
    fig.add_shape(type="rect",
                      y0=round(sortadlist2[0][0],2)-.3, y1=round(sortadlist2[0][0],2)+.3, x0=-1, x1=len(df),
                      fillcolor="darkcyan",
                      opacity=0.15)
    '''
    for v in range(len(sortadlist)):
        if pea:
            #res = getBAB(df,sortadlist[v][0],stockName)
            res = [0,0,0]
        else: 
            res = [0,0,0]
        fig.add_trace(go.Scatter(x=df['time'],
                                 y= [sortadlist[v][0]]*len(df['time']) ,
                                 line_color='brown' if (str(sortadlist[v][3]) == 'B(SELL)' or str(sortadlist[v][3]) == 'BB(SELL)') else 'rgb(0,104,139)' if (str(sortadlist[v][3]) == 'A(BUY)' or str(sortadlist[v][3]) == 'AA(BUY)') else 'rgb(0,0,0)',
                                 text = str(sortadlist[v][4]) + ' ' + str(sortadlist[v][1]) + ' ' + str(sortadlist[v][3])  + ' ' + str(sortadlist[v][5])  + ' ' + str(sortadlist[v][6]) + ' ' + str(res[0]) + ' ' + str(res[1]) + ' ' + str(res[2]),
                                 #text='('+str(priceDict[sortadlist[v][0]]['ASKAVG'])+'/'+str(priceDict[sortadlist[v][0]]['BIDAVG']) +')'+ '('+str(priceDict[sortadlist[v][0]]['ASK'])+'/'+str(priceDict[sortadlist[v][0]]['BID']) +')'+  '('+ sortadlist[v][3] +') '+str(sortadlist[v][4]),
                                 textposition="bottom left",
                                 name=str(sortadlist[v][0]),
                                 showlegend=False,
                                 visible=False,
                                 mode= 'lines',
                                
                                ),
                      row=1, col=1
                     )
        
        
    


    for tmr in range(0,len(fig.data)): #3
        fig.data[tmr].visible = True
        
    steps = []
    for i in np.arange(0,len(fig.data)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)}],
            #label=str(pricelist[i-1])
        )
        for u in range(0,i):
            step["args"][0]["visible"][u] = True
            
        
        step["args"][0]["visible"][i] = True
        steps.append(step)
    
    #print(steps)
    #if previousDay:
        #nummber = 6
    #else:
        #nummber = 0
    sliders = [dict(
        active=10,
        currentvalue={"prefix": "Price: "},
        pad={"t": 10},
        steps=steps[19+trcount:]#[8::3]
    )]

    fig.update_layout(
        sliders=sliders
    )
    
    
    
    fig.update_layout(height=800, xaxis_rangeslider_visible=False, showlegend=False)
    #fig.add_trace(go.Scatter(x=df['time'], y=df['BbandsMid'], mode='lines', name='BbandsMid'))
    #fig.add_trace(go.Scatter(x=df['time'], y=df['BbandsUpp'], mode='lines', name='BbandsUpp'))
    #fig.add_trace(go.Scatter(x=df['time'], y=df['BbandsLow'], mode='lines', name='BbandsLow'))


    
    
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=2, col=2)
    fig.update_xaxes(showticklabels=False, row=2, col=1)
    #fig.update_xaxes(showticklabels=False, row=1, col=2)
    #fig.show(config={'modeBarButtonsToAdd': ['drawline']})
    return fig
    
    
 


def valueAreaV1(lst):
    #lst = [i for i in lst if i[1] > 0]
    pocIndex = sorted(lst, key=lambda stock: float(stock[1]), reverse=True)[0][2]
    sPercent = sum([i[1] for i in lst]) * .70
    pocVolume = lst[lst[pocIndex][2]][1]
    #topIndex = pocIndex - 2
    #dwnIndex = pocIndex + 2
    topVol = 0
    dwnVol = 0
    total = pocVolume
    #topBool1 = topBool2 = dwnBool1 = dwnBool2 =True

    if 0 <= pocIndex - 1 and 0 <= pocIndex - 2:
        topVol = lst[lst[pocIndex - 1][2]][1] + lst[lst[pocIndex - 2][2]][1]
        topIndex = pocIndex - 2
        #topBool2 = True
    elif 0 <= pocIndex - 1 and 0 > pocIndex - 2:
        topVol = lst[lst[pocIndex - 1][2]][1]
        topIndex = pocIndex - 1
        #topBool1 = True
    else:
        topVol = 0
        topIndex = pocIndex

    if pocIndex + 1 < len(lst) and pocIndex + 2 < len(lst):
        dwnVol = lst[lst[pocIndex + 1][2]][1] + lst[lst[pocIndex + 2][2]][1]
        dwnIndex = pocIndex + 2
        #dwnBool2 = True
    elif pocIndex + 1 < len(lst) and pocIndex + 2 >= len(lst):
        dwnVol = lst[lst[pocIndex + 1][2]][1]
        dwnIndex = pocIndex + 1
        #dwnBool1 = True
    else:
        dwnVol = 0
        dwnIndex = pocIndex

    # print(pocIndex,topVol,dwnVol,topIndex,dwnIndex)
    while sPercent > total:
        if topVol > dwnVol:
            total += topVol
            if total > sPercent:
                break

            if 0 <= topIndex - 1 and 0 <= topIndex - 2:
                topVol = lst[lst[topIndex - 1][2]][1] + \
                    lst[lst[topIndex - 2][2]][1]
                topIndex = topIndex - 2

            elif 0 <= topIndex - 1 and 0 > topIndex - 2:
                topVol = lst[lst[topIndex - 1][2]][1]
                topIndex = topIndex - 1

            if topIndex == 0:
                topVol = 0

        else:
            total += dwnVol

            if total > sPercent:
                break

            if dwnIndex + 1 < len(lst) and dwnIndex + 2 < len(lst):
                dwnVol = lst[lst[dwnIndex + 1][2]][1] + \
                    lst[lst[dwnIndex + 2][2]][1]
                dwnIndex = dwnIndex + 2

            elif dwnIndex + 1 < len(lst) and dwnIndex + 2 >= len(lst):
                dwnVol = lst[lst[dwnIndex + 1][2]][1]
                dwnIndex = dwnIndex + 1

            if dwnIndex == len(lst)-1:
                dwnVol = 0

        if dwnIndex == len(lst)-1 and topIndex == 0:
            break
        elif topIndex == 0:
            topVol = 0
        elif dwnIndex == len(lst)-1:
            dwnVol = 0

        # print(total,sPercent,topIndex,dwnIndex,topVol,dwnVol)
        # time.sleep(3)

    return [lst[topIndex][0], lst[dwnIndex][0], lst[pocIndex][0]]




def historV1(df, num, quodict, trad:list=[], quot:list=[]):
    '''
    bTime = df['timestamp'][df[df['time']==df['time'].iloc[0]].index.values[0]]
    if len(trad) == 0:
        trad = list(client.list_trades(stkName, timestamp_gte=int((str(bTime)  + '000000')), timestamp_lte=int((str(df['timestamp'].iloc[-1]+(60000*2)) + '000000')), order='asc', limit=50000))
        #AllTrade = trad
    
    else:
        trad = list(client.list_trades(stkName, timestamp_gte=int((str(bTime)  + '000000')), timestamp_lte=int((str(AllTrade[len(AllTrade)].participant_timestamp+(60000*2)) + '000000')), order='asc', limit=50000))
        AllTrade += trad

    if len(quot) == 0:
        quot = list(client.list_quotes(stkName, timestamp_gte=int((str(bTime) + '000000')), timestamp_lte=int((str(df['timestamp'].iloc[-1]+(60000*2)) + '000000')), order='asc', limit=50000))
        #AllQuote = quot
    
    else:
        quot = list(client.list_quotes(stkName, timestamp_gte=int((str(bTime-18000) + '000000')), timestamp_lte=int((str(AllQuote[len(AllQuote)].participant_timestamp+(60000*2)) + '000000')), order='asc', limit=50000))
        AllQuote += quot
    '''

    
    pzie = [(i[0],i[1]) for i in trad]
    dct ={}
    for i in pzie:
        if i[0] not in dct:
            dct[i[0]] =  i[1]
        else:
            dct[i[0]] +=  i[1]
            
    
    pzie = [i for i in dct if dct[i] > 500]#list(set(pzie))
    
    hist, bin_edges = np.histogram(pzie, bins=num)
    
    cptemp = []
    cntt = 0
    for i in range(len(hist)):
        pziCount = 0
        for x in trad:
            if bin_edges[i] <= x[0] < bin_edges[i+1]:
                pziCount += x[1]
        if pziCount > 100:
            cptemp.append([bin_edges[i],pziCount,cntt,bin_edges[i+1]])
            cntt+=1
        
    for i in cptemp:
        i+=countCandle(trad,quot,i[0],i[3],df['name'][0],quodict)

        
    sortadlist = sorted(cptemp, key=lambda stock: float(stock[1]), reverse=True)
    
    #AllTrade = trad
    #AllQuote = quot  
    return [cptemp,sortadlist]



  
def countCandle(trad,quot,num1,num2, stkName, quodict):
    enum = ['Bid(SELL)','BelowBid(SELL)','Ask(BUY)','AboveAsk(BUY)','Between']
    color = ['red','darkRed','green','darkGreen','black']

   
    lsr = splitHun(stkName,trad, quot, num1, num2, quodict)
    ind = lsr.index(max(lsr))   #lsr[:4]
    return [enum[ind],color[ind],lsr]


def splitHun(stkName, trad, quot, num1, num2, quodict):
    Bidd = 0
    belowBid = 0
    Askk = 0
    aboveAsk = 0
    Between = 1
    #qdict = [i.participant_timestamp for i in quot]
    

    #print([Bidd,Askk,Between])    
    return [Bidd,belowBid,Askk,aboveAsk,Between]



from dash import Dash, dcc, html, Input, Output, callback
app = Dash()
app.layout = html.Div([
    dcc.Graph(id='graph'),
    dcc.Interval(
        id='interval',
        interval=60000,
        n_intervals=0,
      )
])


@callback(Output('graph', 'figure'),
          Input('interval', 'n_intervals'))

def update_graph_live(n_intervals):
    print('inFunction')	
    fft = []
    global AllTrade
    AllTrade = []
    AllQuote = []
    quodict = {}
    OptionOrders = []
    OptionOrdersCall = []
    OptionOrdersPut = []
    OptionTimeFrame = []
    stkName = 'TSLA'
    
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

    #day = '08'
    #month = '10'
    agMins = 2

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
    except(exceptions.BadResponse):
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
    
    gclient = storage.Client(project="stockapp-401615")
    bucket = gclient.get_bucket("stockapp-storage")
    blob = Blob('GetAllTrades'+stkName, bucket) 
    AllTrade = json.loads(blob.download_as_text())

    AllTrade = [[float(i[0]), int(float(i[1])), int(i[2]), int(i[3])] for i in AllTrade]
    
    
    '''
    if len(AllTrade) == 0:
        AllTrade = list(client.list_trades(stkName, timestamp_gte=int((str(bTime)  + '000000')), timestamp_lte=int((str(df['timestamp'].iloc[-1]+(60000*agMins)) + '000000')), order='asc', limit=50000))
    else:
        trad = list(client.list_trades(stkName, timestamp_gt=int(str(AllTrade[len(AllTrade)-1].participant_timestamp)), timestamp_lte=int((str(df['timestamp'].iloc[-1]+(60000*agMins)) + '000000')), order='asc', limit=50000))
        AllTrade += trad
    
    
    if len(AllQuote) == 0:
        AllQuote = list(client.list_quotes(stkName, timestamp_gte=int((str(bTime) + '000000')), timestamp_lte=int((str(df['timestamp'].iloc[-1]+(60000*agMins)) + '000000')), order='asc', limit=50000))
        for i in AllQuote:
            quodict[i.participant_timestamp] = [i.bid_price, i.ask_price]
    else:
        quot = list(client.list_quotes(stkName, timestamp_gt=int(str(AllQuote[len(AllQuote)-1].participant_timestamp)), timestamp_lte=int((str(df['timestamp'].iloc[-1]+(60000*agMins)) + '000000')), order='asc', limit=50000))
        for i in quot:
            quodict[i.participant_timestamp] = [i.bid_price, i.ask_price]
        AllQuote += quot 
    '''
    #print(len(AllTrade))
    

    vwap(df)
    ema(df)
    PPP(df)


    hs = historV1(df,50,quodict,AllTrade,AllQuote)
    
    va = valueAreaV1(hs[0])
    
    x = np.array([i for i in range(len(df))])
    y = np.array([i for i in df['28ema']])
    
    

    # Simple interpolation of x and y
    f = interp1d(x, y)
    x_fake = np.arange(0.1, len(df)-1, 1)  #0.10

    # derivative of y with respect to x
    df_dx = derivative(f, x_fake, dx=1e-6)
    
    mTrade = [i for i in AllTrade if i[1] >= 10000]
    
    mTrade = sorted(mTrade, key=lambda d: d[1], reverse=True) 
    
    checkDup = []
    newTList =[ ]
    for i in range(len(mTrade)):
        if mTrade[i][0] not in checkDup:
            checkDup.append(mTrade[i][0])
            newTList.append(mTrade[i])
            
            
    newTList = newTList[:4]
    
    #for i in range(len(newTList)):
        #newTList[i].append(i)
    [newTList[i].append(i) for i in range(len(newTList))]
            
    for i in newTList:  
        hourss = datetime.fromtimestamp(int(i[2] // 1000000000)).hour
        if hourss < 10:
            hourss = '0'+str(hourss)
        minss = datetime.fromtimestamp(int(i[2] // 1000000000)).minute
        if minss < 10:
            minss = '0'+str(minss)
        secs = datetime.fromtimestamp(int(i[2] // 1000000000)).second
        if secs < 10:
            secs = '0'+str(secs)
        opttimeStamp = str(hourss) + ':' + str(minss) + ':'+ str(secs)
                                             
        for t in client.list_quotes(stkName, timestamp_lte = i[2]):        
            break 
        
        if i[0] == t.ask_price:
            #i.append('A (buyer agreed to buy the asset at the price)')
            i+= ['A(BUY)', opttimeStamp]
        elif i[0] > t.ask_price:
            #i.append('AA (buyer was willing to pay more than what sellers were initially asking)')
            i+= ['AA(BUY)', opttimeStamp]
        elif i[0] == t.bid_price:
            #i.append('B (seller agreed to sell at price)')
            i+= ['B(SELL)', opttimeStamp]
        elif i[0] < t.bid_price:
            #i.append('BB (the seller accepted a price lower than what buyers were initially willing to pay)')
            i+= ['BB(SELL)', opttimeStamp]
        elif t.bid_price < i[0] < t.ask_price:
            i+= ['BBA', opttimeStamp]
            
    newwT =[]
    for i in newTList:
        newwT.append([i[0],i[1],i[2],i[5], i[4],i[3],i[6]])
        
    #print(newwT)

    
    try:
   
        gclient = storage.Client(project="stockapp-401615")
        bucket = gclient.get_bucket("stockapp-storage")
        blob = Blob('dailyCandle'+stkName, bucket) 
        fft = json.loads(blob.download_as_text())
        
   
        gclient = storage.Client(project="stockapp-401615")
        bucket = gclient.get_bucket("stockapp-storage")
        blob = Blob('OptionTrackerPut'+stkName, bucket) 
        OptionOrdersPut = json.loads(blob.download_as_text())
              
        
        gclient = storage.Client(project="stockapp-401615")
        bucket = gclient.get_bucket("stockapp-storage")
        blob = Blob('OptionTrackerCall'+stkName, bucket) 
        OptionOrdersCall = json.loads(blob.download_as_text())
              
        OptionOrders = OptionOrdersPut + OptionOrdersCall
        OptionOrders.sort(key=lambda x:int(x[2])) 
         
    
        gclient = storage.Client(project="stockapp-401615")
        bucket = gclient.get_bucket("stockapp-storage")
        blob = Blob('OptionTimeFrame'+stkName, bucket) 
        OptionTimeFrame = json.loads(blob.download_as_text())

        '''
        overall = [[i[0]] for i in OptionTimeFrame]
        for opf in overall:
            tput = sum([int(x[2]) for x in OptionTimeFrame[:[i[0] for i in OptionTimeFrame].index(opf[0])+1]])
            tcall = sum([int(x[3]) for x in OptionTimeFrame[:[i[0] for i in OptionTimeFrame].index(opf[0])+1]])
            ttotal = tput+tcall
            opf+=[' (Put:'+str(round(tput / ttotal,2))+'('+str(tput)+') | '+'Call:'+str(round(tcall / ttotal,2))+'('+str(tcall)+') ',tput,tcall]

            
        for o in overall:
            print(o)
        #print(overall)
        
        import matplotlib.pyplot as plt 
        import numpy as np 
          
        # create data 
        x = [round(i[2] / (i[2]+i[3]),2) for i in overall] 
        y = [round(i[3] / (i[2]+i[3]),2) for i in overall] 
          
        # plot lines 
        plt.plot([i for i in range(len(overall))], x, label = "Put") 
        plt.plot([i for i in range(len(overall))], y, label = "Call")  
        plt.legend() 
        plt.show()
          
        OptionTimeFrame = [[i, '', '0', '0'] for i in list(df['time'][:bisect.bisect_left(list(df['time'].values),'09:30:00')])]+OptionTimeFrame
         
        
        mminss = [10,15,20,25,30,35,40,60][::-1]
        for mint in mminss:
            try:
                checkTime = (datetime.strptime(OptionTimeFrame[len(OptionTimeFrame)-1][0], '%H:%M:%S') - timedelta(minutes=mint)).time()
                checkTime.strftime("%H:%M:%S")
                
                stIndex = bisect.bisect_left([i[0] for i in OptionTimeFrame],checkTime.strftime("%H:%M:%S"))
                allputMins = [float(i[2]) for i in OptionTimeFrame[stIndex:]]
                allcallMins = [float(i[3]) for i in OptionTimeFrame[stIndex:]]
                deputMins = round(sum(allputMins) / sum(allputMins+allcallMins),2)
                decallMins = round(sum(allcallMins) / sum(allputMins+allcallMins),2)
                print('last '+str(mint)+'m: '+ ' (Put:'+str(deputMins)+'('+str(sum(allputMins))+') | '+'Call:'+str(decallMins)+'('+str(sum(allcallMins))+') ')
            except(ZeroDivisionError,IndexError):
                continue
        '''  
        '''
        for i in fft:
            i[10] = int(i[10])
            
            if '.' in i[2]:
                i[2] = i[2][:i[2].index('.')]
            i[2] = int(i[2])
        '''
    except(FileNotFoundError):
        pass#continue
    
    fg = plotChart(df, [hs[1],newwT], va[0], va[1], x_fake, df_dx, bigOrders=[], optionOrderList=OptionOrders, stockName=stkName,previousDay=False, prevdtstr='', pea=False, sord = fft, OptionTimeFrame = OptionTimeFrame, overall=[]) #trends=FindTrends(df,n=10)
    #fg.show(config={'modeBarButtonsToAdd': ['drawline']})
    
    return fg
        

if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=8080)
    #app.run_server(debug=False, use_reloader=False)
    killAll()
    

