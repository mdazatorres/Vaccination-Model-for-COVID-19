import sys
from datetime import date, timedelta
from matplotlib.pyplot import subplots, rcParams, close, plot
from pandas import datetime, read_excel, read_csv, Timestamp
import matplotlib.pyplot as plt
from main_mcmc import class_mcmc
import numpy as np

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


def call_MCMC(clave, trim, court, init0, init_index0, size_window, size_window_new, exit_probs,
              R_rates_V2, Pobs_I, Pobs_D, delta_day, num_data, workdir="./../"):
    data_fnam = clave + '.csv'
    if size_window == size_window_new:
        init = init0 + timedelta(days=court * (size_window - 1))
        init_index = (init - Info[clave][2]).days
    else:
        init = init0 + timedelta(days=court * (size_window - 1)) + timedelta(days=size_window_new - size_window)
        init_index = (init - Info[clave][2]).days

    if trim == 0:
        out_fnam = clave
    else:
        out_fnam = clave+"_trim%d" % (trim,)
    exit_probs_copy = exit_probs
    N = Info[clave][1]
    init_v=Info[clave][3]

    zm = class_mcmc(Region=Info[clave][0], init_day=Info[clave][2], data_fnam=data_fnam,\
        N=N, out_fnam=out_fnam, init_index=init_index, init=init, init_index0=init_index0, init0=init0,\
        init_v=init_v, trim=trim, Pobs_I=Pobs_I, Pobs_D=Pobs_D, exit_probs=exit_probs_copy, R_rates=R_rates_V2,
        court=court, size_window=size_window, size_window_new=size_window_new, num_data=num_data,
        delta_day=delta_day, workdir=workdir)
    return zm


def PlotFigs(zm, pred, court, q=[10,25,50,75,90], blue=True, workdir='./../'):
    close('all')
    out_fnam = zm.out_fnam
    #-----------------------------------------------------------
    fig, ax = subplots(num=1, figsize=(8, 6))
    zm.PlotEvolution(pred=pred, cumm=False, log=False, ty=0, ax=ax, q=q, blue=blue, add_MRE=True,\
        label=r'Mediana', csv_fnam= "%s_court_%s_I.csv" % (out_fnam, court))
    fig.tight_layout()
    fig.savefig("%s_court_%s_I.png" % (out_fnam, court))
    plt.close()
    # -----------------------------------------------------------
    fig, ax = subplots(num=2, figsize=(8, 6))
    zm.PlotEvolution(pred=pred, cumm=False, log=False, ty=1, ax=ax, q=q, blue=blue,\
        label=r'Mediana', right_axis=False, csv_fnam="%s_court_%s_D.csv" % (out_fnam,court))
    fig.tight_layout()
    fig.savefig("%s_court_%s_D.png" % (out_fnam, court))
    plt.close()
    #------------------------------------------------------------
    fig, ax = subplots(num=5, figsize=(8, 6))
    zm.plot_cones(pred=pred, ty=0, cumm=False, log=False, ax=ax,
                  q=q, blue=blue, color='red', color_q='black',
                  label=True, right_axis=False)
    fig.tight_layout()
    fig.savefig("%s_court_%s_I_cono.png" % (out_fnam, court))
    plt.close()
    #------------------------------------------------------------
    fig, ax = subplots(num=6, figsize=(8, 6))
    zm.plot_cones(pred=pred, ty=1, cumm=False, log=False, ax=ax,
                  q=q, blue=blue, color='red', color_q='black',
                  label=True, right_axis=False)
    fig.tight_layout()
    fig.savefig("%s_court_%s_D_cono.png" % (out_fnam, court))
    plt.close()
    return ax


def all_predictions(T, clave, num_data, size_window, size_window_new, pred, R_rates, all_pred, forecast, init_trans):
    data_fnam = clave + '.csv'
    Init0 = Info[clave][2]
    init0 = Init0 + timedelta(days=init_trans)
    init_index0 = (init0 - Info[clave][2]).days
    data = np.array(read_csv(data_fnam))
    court = (data[init_index0:].shape[0] - num_data) // (size_window - 1) + 1
    res = (data[init_index0:].shape[0] - num_data) % (size_window - 1)
    init0 = init0 + timedelta(days=res)
    init_index0 = (init0 - Info[clave][2]).days
    delta_day = init_trans + res
    i = 0
    if all_pred==True:
        while i < 3:
            Info[clave][-1] = call_MCMC(clave, trim=0, court=i, init0=init0, init_index0=init_index0, size_window=size_window,
                                       size_window_new=size_window_new, exit_probs=exit_probs, R_rates_V2=R_rates, Pobs_I=Pobs_I,
                                       Pobs_D=Pobs_D, delta_day=delta_day, num_data=num_data)
            Info[clave][-1].clave = clave
            zm = Info[clave][-1]
            x0 = zm.sim_init()
            xp0 = zm.sim_init()
            if zm.support(x0) and zm.support(xp0):
                zm.RunMCMC(T=T, x0=x0, xp0=xp0, pred=pred, plot_fit=False)
                PlotFigs(zm, pred=pred, court=i, blue=True)
                i = i + 1
    else:
        Info[clave][-1] = call_MCMC(clave, trim=0, court=court + forecast, init0=init0, init_index0=init_index0,
                                   size_window=size_window, size_window_new=size_window_new, exit_probs=exit_probs,
                                   R_rates_V2=R_rates, Pobs_I=Pobs_I, Pobs_D=Pobs_D, delta_day=delta_day, num_data=num_data)
        Info[clave][-1].clave = clave
        zm = Info[clave][-1]
        x0 = zm.sim_init()
        xp0 = zm.sim_init()

        while (zm.support(x0) and zm.support(xp0))==False:
            x0 = zm.sim_init()
            xp0 = zm.sim_init()

        zm.RunMCMC(T=T, x0=x0, xp0=xp0, pred=pred, plot_fit=False)
        PlotFigs(zm, pred=pred, court=i, blue=True)

Info = ReadInfo()
num_data = 35
exit_probs = [0.6, 1]
Pobs_I = 0.85
Pobs_D = 0.9
pred = 60
init_trans=3
size_window = 8
size_window_new = 8
T = 50000
R_rates = {'E': [1/5, r'\sigma_1',  1], 'I^S': [1/13, r'\sigma_2',  1], 'I^A': [1/7, r'\gamma_1', 1]}
rcParams.update({'font.size': 12})
q = [10, 25, 50, 75, 90]
workdir = "./../"

clave='CA'
all_predictions(T=T, clave=clave, num_data=num_data, size_window=size_window,
            size_window_new=size_window_new, pred=pred, R_rates=R_rates, all_pred=True, forecast=0, init_trans=init_trans)


