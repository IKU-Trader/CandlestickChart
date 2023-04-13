# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 22:03:12 2022

@author: IKU-Trader
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../CandlestickChart'))

import pandas as pd
import numpy as np
import pytz
from candle_chart import CandleChart, candleData2arrays, makeFig, gridFig
from time_utils import TimeUtils


def midpoint(ohlcv):
    op = ohlcv[0]
    cl = ohlcv[3]
    mid = []
    for o, c in zip(op, cl):
        mid.append( (o + c ) / 2)
    return np.array(mid)
    
def momentum(array, distance):
    n = len(array)
    mom = np.full(n, np.nan)
    for i in range(distance, n):
        d = array[i] - array[i - distance]
        mom[i] = d
    return np.array(mom)

def momentumPercent(array, distance):
    n = len(array)
    mom = np.full(n, np.nan)
    for i in range(distance, n):
        d = array[i] - array[i - distance]
        mom[i] = d / array[i - distance] * 100.0
    return np.array(mom)    

def sma(array, period):
    n = len(array)
    out = []
    for i in range(n):
        if i < period - 1:
            out.append(np.nan)
        else:
            a = array[i - period + 1: i + 1]
            out.append(sum(a) / period)
    return np.array(out)

def polarity(array, threshold=0.0, values=(1.0, -1.0)):
    out = [values[0] if v >= threshold else values[-1] for v in array]
    return np.array(out)

def crossPoint(array, threshold=0.0):
    n = len(array)
    up = []
    down = []
    for i in range(1, n):
        if array[i - 1] <= threshold and array[i] > threshold:
            up.append([i, array[i]])
        if array[i - 1] >= threshold and array[i] < threshold:
            down.append([i, array[i]])
    return up, down

def backward(ohlc):
    n = len(ohlc[0])
    out = np.zeros(n)
    op = ohlc[0]
    hi = ohlc[1]
    lo = ohlc[2]
    cl = ohlc[3]
    for i in range(1, n):
        mid = (op[i - 1] + cl[i - 1]) / 2
        if cl[i - 1] >= op[i - 1]:
            #positive
            out[i] = lo[i] - op[i - 1]
        else:
            out[i] = op[i - 1] - hi[i]
    return out

def thrust(ohlc):
    n = len(ohlc[0])
    out = np.zeros(n)
    op = ohlc[0]
    hi = ohlc[1]
    lo = ohlc[2]
    cl = ohlc[3]
    for i in range(1, n):
        if cl[i] > hi[i - 1]:
            out[i] = 1
        elif cl[i] < lo[i - 1]:
            out[i] = -1
    return out

def plot(year, month, days, tohlcv):
    time = tohlcv[0]
    jst_all = TimeUtils.changeTimezone(time, TimeUtils.TIMEZONE_TOKYO)     
    sma5_all = sma(tohlcv[4], 5)
    sma20_all = sma(tohlcv[4], 20)
    thrst_all = thrust(tohlcv[1:])
    mid_all = midpoint(tohlcv[1:])
    back_all = backward(tohlcv[1:])    
    for day in days:
        try:
            t0 = TimeUtils.pyTime(year, month, day, 0, 0, 0, TimeUtils.TIMEZONE_TOKYO)
            t1 = TimeUtils.pyTime(year, month, day, 5, 0, 0, TimeUtils.TIMEZONE_TOKYO)
        except:
            continue
        
        length, begin, end = TimeUtils.sliceTime(jst_all, t0, t1)
        print(f'{year}-{month}-{day} data size:', length)
        if length < 50:
            continue
        
        op = tohlcv[1][begin:end+1]
        hi = tohlcv[2][begin:end+1]
        lo = tohlcv[3][begin:end+1]
        cl = tohlcv[4][begin:end+1]
        vo = tohlcv[5][begin:end+1]
        ohlcv = [op, hi, lo, cl, vo]
    
        jst = jst_all[begin:end+1]
        
        tmp_tohlcv = [jst, op, hi, lo, cl, vo]
        
        sma5 = sma5_all[begin: end + 1]
        sma20 = sma20_all[begin: end + 1]
        mid = mid_all[begin: end + 1]
        thrst = thrst_all[begin: end + 1]
        back = back_all[begin: end + 1]
        back /= cl[0] * 100.0
        dif = (sma5 - sma20) / cl[0] * 100.0
        
        fig, axes = gridFig([8, 3, 2, 1], (20, 8))
        chart1 = CandleChart(fig, axes[0], f'DJI(1min) {year}-{month}-{day}')
        chart1.drawCandle(tmp_tohlcv[0], tmp_tohlcv[1], tmp_tohlcv[2], tmp_tohlcv[3], tmp_tohlcv[4])    
        chart1.drawLine(jst, sma5, color='red')
        chart1.drawLine(jst, sma20, color='blue')
        
        up_points, down_points = crossPoint(dif)
        for i, value in up_points:
            chart1.drawMarker(jst[i], hi[i], '^', 'green')
        for i, _ in down_points:
            chart1.drawMarker(jst[i], hi[i], 'v', 'red')
            
        chart2 = CandleChart(fig, axes[1], '')
        chart2.drawLine(jst, back, color='orange')
        #lim = chart3.getYlimit()
        chart2.ylimit((-4e-5, 4e-5))            
                    
        chart3 = CandleChart(fig, axes[2], '')
        chart3.drawLine(jst, dif)
        chart3.ylimit((-0.3, 0.3))
    
        chart4 = CandleChart(fig, axes[3], '')
        chart4.drawLine(jst, thrst, color='green')

def read(filepath):
    f = open(filepath)
    line = f.readline()
    line = f.readline()
    candles = []
    tohlcv = [[], [], [], [], [], []]
    while line:
        values = line.split(',')
        if len(values) == 6:
            t = TimeUtils.str2pytime(values[0], pytz.utc)
            o = float(values[1])
            h = float(values[2])
            l = float(values[3])
            c = float(values[4])
            v = float(values[5])
            candles.append([t, o, h, l, c, v])
            tohlcv[0].append(t)
            tohlcv[1].append(o)
            tohlcv[2].append(h)
            tohlcv[3].append(l)
            tohlcv[4].append(c)
            tohlcv[5].append(v)
        line = f.readline()
    f.close()
    return (candles, tohlcv)

def test():
    year = 2019
    candles, tohlcv = read(f'../MarketData/DJI/DJI_Feature_{year}.csv')
    plot(year, 8, [1, 31], tohlcv)
    

if __name__ == '__main__':
    test()