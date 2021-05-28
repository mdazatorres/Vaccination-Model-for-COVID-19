from datetime import date, timedelta
import datetime as dt
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['font.size'] = 18


def ReadInfo(filename='Info_data.csv'):
    mapeo = {}
    file1 = open(filename, 'r')
    count = 0

    while True:
        count += 1
        line = file1.readline()
        if not line:
            break
        sline = line.strip()
        data = sline.split(',')
        mapeo[data[0]] = data[1:]
        mapeo[data[0]][1] = int(mapeo[data[0]][1])
        for i in range(3, len(data)):
            dateString = data[i].split(' ')
            mapeo[data[0]][i-1] = date(int(dateString[0]), int(dateString[1]), int(dateString[2]))
        mapeo[data[0]].append("None")
    file1.close()
    return mapeo


def calib_forecast(pred, court, init0, init_day, Data, clave, ty):
    init = init0 + dt.timedelta(days=court * (size_window-1))
    init_index = (init - init_day).days
    data_pred = Data[init_index + num_data: init_index + num_data + pred]
    if ty == 0:
        data_pred = data_pred[:, 1]
        quantiles = read_csv(workdir + 'csv/' + clave + '_' + 'court_' + str(court) + '_I' + '.csv')
    else:
        data_pred = data_pred[:, 0]
        quantiles = read_csv(workdir + 'csv/' + clave + '_' + 'court_' + str(court) + '_D' + '.csv')

    q_75 = np.array(quantiles[' q_75'])[num_data:num_data+pred]
    q_25 = np.array(quantiles[' q_25'])[num_data:num_data+pred]
    q_10 = np.array(quantiles[' q_10'])[num_data:num_data+pred]
    q_90 = np.array(quantiles[' q_90'])[num_data:num_data+pred]
    cov50 = np.zeros(weeks)
    cov90 = np.zeros(weeks)
    for i in range(weeks):
        wk_50 = np.mean((q_25[0:7*(i+1)] < data_pred[0:7*(i+1)]) == (data_pred[0:7*(i+1)] < q_75[0:7*(i+1)]))
        wk_90 = np.mean((q_10[0:7*(i+1)] < data_pred[0:7*(i+1)]) == (data_pred[0:7*(i+1)] < q_90[0:7*(i+1)]))
        cov50[i] = wk_50
        cov90[i] = wk_90
    return cov50, cov90


def calib_forecast_all(pred, court, init0, init_day, Data, clave, weeks, ty):
    Cov50 = np.zeros((court, weeks))
    Cov90 = np.zeros((court, weeks))
    for i in range(court):
        cov50, cov90 = calib_forecast(pred, i, init0, init_day, Data, clave, ty)
        Cov50[i] = cov50
        Cov90[i] = cov90
    return Cov50, Cov90


def run_all_states_latex(clave='CA'):
    init_day = Info[clave][2]
    init_trans = 3
    init0 = init_day + timedelta(days=init_trans)
    init_index0 = (init0 - Info[clave][2]).days
    Data = np.array(read_csv( clave + '.csv'))
    court = (Data[init_index0:].shape[0] - num_data) // (size_window - 1) + 1
    court= court-weeks-1
    res = (Data[init_index0:].shape[0] - num_data) % (size_window - 1)
    init0 = init0 + timedelta(days=res)

    Cov50_I, Cov90_I = calib_forecast_all(pred, court, init0, init_day, Data, clave, weeks, ty=0)
    Cov50_D, Cov90_D= calib_forecast_all(pred, court, init0, init_day, Data, clave, weeks, ty=1)

    Tab50_I = pd.DataFrame(Cov50_I, np.arange(court), columns=['week 1', 'week 2', 'week 3', 'week 4'])
    Tab50_D = pd.DataFrame(Cov50_D, np.arange(court), columns=['week 1', 'week 2', 'week 3', 'week 4'])
    Tab90_I = pd.DataFrame(Cov90_I, np.arange(court), columns=['week 1', 'week 2', 'week 3', 'week 4'])
    Tab90_D = pd.DataFrame(Cov90_D, np.arange(court), columns=['week 1', 'week 2', 'week 3', 'week 4'])
    I_mean = pd.DataFrame({'q25-q75': Tab50_I.mean(axis=0), "q10-q90": Tab90_I.mean(axis=0)})
    D_mean = pd.DataFrame({'q25-q75': Tab50_D.mean(axis=0), "q10-q90": Tab90_D.mean(axis=0)})
    print(I_mean)
    print(D_mean)


Info = ReadInfo()
num_data = 35
size_window = 8
weeks = 4
pred = 28
workdir = "./../"
#--- Uncomment the following line to get Table S1
#run_all_states_latex()

