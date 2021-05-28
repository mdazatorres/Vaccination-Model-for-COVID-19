from datetime import date, timedelta
import matplotlib.dates as mdates
import datetime as dt
import pickle
import numpy as np
from pandas import read_csv
from matplotlib.pyplot import subplots, close
import matplotlib.pyplot as plt
from fm_matrix import fm_matrix
plt.rcParams['font.size'] = 25


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


def PlotEvolution(pred, solns, log, ty, ax, q, blue, color, color_q,label, right_axis, label_cases, data):
    every = 1
    loc = 2
    Pobs = (Pobs_I, Pobs_D)

    if ty == 0:
        data = data[:, 1]  # Infected reported
        title = 'cases'
    else:
        data = data[:, 0]  # Deaths reported
        title = 'deaths'

    prevalence = data  # aggregate observed data
    solns = np.diff(np.append(np.zeros((solns[ty].shape[0], 1)), Pobs[ty] * solns[ty], axis=1), axis=1)
    ylabel = 'Confirmed ' + title
    title = 'Incidence ' + title

    length = m
    shift = length + pred
    days = mdates.drange(init, init + dt.timedelta(shift), dt.timedelta(days=1))
    sv = -np.ones((len(days), 11))
    for i, day in enumerate(days):
        d = dt.date.fromordinal(int(day))
        sv[i, 0] = d.year
        sv[i, 1] = d.month
        sv[i, 2] = d.day
    sv[:length, 3] = prevalence
    for i in range(length):
        sv[i, 6:11] = np.quantile(solns[:, i], q=np.array(q) / 100)

    for i in range(length, shift, every):
        sv[i, 6:11] = np.quantile(solns[:, i], q=np.array(q) / 100)

    ax.plot(days[:shift], sv[:shift, 8], '-', linewidth=2, color=color, label=label)
    if blue:
        ax.fill_between(days[length-1:shift], sv[length-1:shift, 6], sv[length-1:shift, 10], color='b', alpha=0.25)
        ax.fill_between(days[length-1:shift], sv[length-1:shift, 7], sv[length-1:shift, 9], color='b', alpha=0.25)

        ax.plot(days[:length], sv[:length, 6], '--', color='b', linewidth=1.5)
        ax.plot(days[:length], sv[:length, 10], '--', color='b', linewidth=1.2)
    else:
        ax.plot(days[:shift], sv[:shift, 6], '--', color=color_q, linewidth=1)
        ax.plot(days[:shift], sv[:shift, 10], '--', color=color_q, linewidth=1)

    if label_cases:
        ax.bar(days[:length], prevalence, color='k', width=0.5, label='Cases', alpha=0.7)
        ax.plot(days[:length], prevalence, 'ko', markersize=2)
    else:
        ax.bar(days[:length], prevalence, color='k', width=0.5, alpha=0.7)
        ax.plot(days[:length], prevalence, 'ko', markersize=2)

    ax.legend(loc=loc, shadow=True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

    if shift < 190:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    else:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=14))
    ax.tick_params(which='major', axis='x')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylabel(ylabel)
    ax.set_ylim((0, max(1.1 * np.max(sv[:, -1]), 1.1 * np.max(prevalence))))
    ax.grid(color='grey', which='both', linestyle='--', linewidth=0.5)

    if right_axis and not (log):
        ax_p = ax.twinx()
        y1, y2 = ax.get_ylim()
        ax_p.set_ylim(y1 * 1e5 / N, y2 * 1e5 / N)
        ax_p.set_ylabel('per 100,000')
    return ax


def PlotEvolution1(pred, solns, court, data_all, ty=0, cumm=False, log=False, ax=None,
                   q=[10, 25, 50, 75, 90], blue=True, color='red', color_q='black', label=True, right_axis=False, plotdata=True,
                   dashed=False,label_cases=False):

    init = init0 + dt.timedelta(days=court * (size_window-1))
    init_index = (init - init_day).days
    num_data_all = data_all[init_index0:].shape[0]
    every = 1
    loc = 1
    length = m
    Pobs = (Pobs_I, Pobs_D)
    shift = length + pred
    data = data_all[init_index: init_index + num_data]
    days = mdates.drange(init, init + dt.timedelta(shift), dt.timedelta(days=1))

    if (init_index0 + shift + court * (size_window - 1)) <= num_data_all:
        data_all = data_all[init_index0: init_index0 + length + court * (size_window - 1)]
    else:
        data_all = data_all[init_index0: init_index0 + length + court * (size_window - 1)]

    if ty == 0:
        data = data[:, 1]  # Infected reported
        title = 'cases'
    else:
        data = data[:, 0]  # Deaths reported
        title = 'deaths'

    prevalence = data
    solns = np.diff(np.append(np.zeros((solns[ty].shape[0], 1)), Pobs[ty] * solns[ty], axis=1),
                    axis=1)
    ylabel = 'Confirmed ' + title
    title = 'Incidence ' + title
    sv = -np.ones((len(days), 11))
    for i, day in enumerate(days):
        d = dt.date.fromordinal(int(day))
        sv[i, 0] = d.year
        sv[i, 1] = d.month
        sv[i, 2] = d.day
    sv[:length, 3] = prevalence

    for i in range(length):
        sv[i, 6:11] = np.quantile(solns[:, i], q=np.array(q) / 100)

    for i in range(length, shift, every):
        sv[i, 6:11] = np.quantile(solns[:, i], q=np.array(q) / 100)

    if label:
        ax.plot(days[:shift], sv[:shift, 8], '-', linewidth=2, color=color, label='Median')
    else:
        ax.plot(days[:shift], sv[:shift, 8], '-', linewidth=2, color=color)

    if blue:  # Blue shaowed quantiles
        dly=0
        ax.fill_between(days[length-dly:shift], sv[length-dly:shift, 6], sv[length-dly:shift, 10], color='b', alpha=0.25)
        ax.fill_between(days[length-dly:shift], sv[length-dly:shift, 7], sv[length-dly:shift, 9], color='b', alpha=0.25)
        if dashed:
            ax.plot(days[:length], sv[:length, 6], '--', color=color_q, linewidth=1)
            ax.plot(days[:length], sv[:length, 10], '--', color=color_q, linewidth=1)
    else:
        ax.plot(days[:shift], sv[:shift, 6], '--', color=color_q, linewidth=1)
        ax.plot(days[:shift], sv[:shift, 10], '--', color=color_q, linewidth=1)

    if plotdata:
        ax.plot(days[:length], prevalence, 'ko', markersize=3)
    if label_cases:
        ax.plot(days[:length], prevalence, 'ko', markersize=3, label='Cases')
    ax.legend(loc=loc, shadow=True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

    if shift < 10:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    else:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=14))

    ax.tick_params(which='major', axis='x')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylabel(ylabel)
    if right_axis and not (log):
        ax_p = ax.twinx()
        y1, y2 = ax.get_ylim()
        ax_p.set_ylim(y1 * 1e5 / N, y2 * 1e5 / N)
        ax_p.set_ylabel('per 100,000')
    ax.grid(color='grey', which='both', linestyle='--', linewidth=0.5)
    return ax


def plot_cones(pred, court, solns, data_all, ty=0, cumm=False, log=False, ax=None,
               q=[10, 25, 50, 75, 90], blue=True, color='red', color_q='black',
               label=True, right_axis=False):
    #i=39 #MS
    i=44 # CA
    solns1 = pickle.load(open(workdir + 'output/' + out_fnam + 'court_' + str(i) + '_solns.pkl', 'rb'))
    PlotEvolution1(pred=pred, solns=solns1, court=i, ty=ty, cumm=cumm, log=log, ax=ax,
                   q=q, blue=blue, color=color, color_q=color_q, label=False, right_axis=right_axis,
                   plotdata=True, data_all=data_all, dashed=True)
    while i<court:
        i = 1+i
        solns1 = pickle.load(open(workdir + 'output/' + out_fnam + 'court_' + str(i) + '_solns.pkl', 'rb'))
        PlotEvolution1(pred=pred, solns=solns1, court=i, ty=ty, cumm=cumm, log=log, ax=ax,
                            q=q, blue=blue, color=color, color_q=color_q, label=False, right_axis=right_axis,
                            plotdata=True, data_all=data_all)
    PlotEvolution1(pred=pred, solns=solns, court=court, ty=ty, cumm=cumm, log=log, ax=ax,
                   q=q, blue=blue, color=color, color_q=color_q, label=label, right_axis=right_axis,
                   plotdata=True, data_all=data_all, label_cases=True)
    ax.grid(color='grey', which='both', linestyle='--', linewidth=0.5)



def PlotFigsZMs(pred, court, blue=True, workdir='./../', save=True):
    close('all')
    q = [10, 25, 50, 75, 90]
    solns = pickle.load(open(workdir + 'output/' + out_fnam + 'court_' + str(court) + '_solns.pkl', 'rb'))
    #-----------------------------------------------------------
    fig, ax = subplots(num=1, figsize=(12, 8))
    PlotEvolution(pred=pred, solns=solns, log=False, ty=0, ax=ax, q=q, blue=blue, color='red',
                  color_q='black', label=r'Median', right_axis=True, label_cases=True, data=data)

    fig.tight_layout()
    if save == True:
        fig.savefig("%s%s_court_%s_I.png" % (workdir + 'figures/', out_fnam, court))

    # -----------------------------------------------------------
    fig, ax = subplots(num=2, figsize=(12, 8))
    PlotEvolution(pred=pred, solns=solns, log=False, ty=1, ax=ax, q=q, blue=blue, color='red',
                  color_q='black', label=r'Median', right_axis=True, label_cases=True, data=data)
    fig.tight_layout()
    if save == True:
        fig.savefig("%s%s_court_%s_D.png" % (workdir + 'figures/', out_fnam, court))

    #------------------------------------------------------------
    fig, ax = subplots(num=3, figsize=(12, 8))
    plot_cones(pred=pred, court=court, solns=solns, ty=0, cumm=False, log=False, ax=ax, q=q, blue=blue, color='red', color_q='b',
               label=True, right_axis=False, data_all=data_all)

    fig.tight_layout()
    if save == True:
        fig.savefig("%s%s_court_%s_I_cono.png" % (workdir + 'figures/', out_fnam, court))

    #-------------------------------------------------------------
    fig, ax = subplots(num=4, figsize=(12, 8))
    plot_cones(pred=pred, court=court, solns=solns, ty=1, cumm=False, log=False, ax=ax, q=q, blue=blue, color='red', color_q='b',
               label=True, right_axis=False, data_all=data_all)

    fig.tight_layout()
    if save == True:
        fig.savefig("%s%s_court_%s_D_cono.png" % (workdir + 'figures/', out_fnam, court))
    return ax


Info = ReadInfo()
exit_probs = [0.1, 1]
Pobs_I = 0.85
Pobs_D = 0.9
pred = 20
size_window = 8
T = 50000
workdir = "./../"
clave = 'CA'
out_fnam = clave
data_fnam = clave+'.csv'
N = Info[clave][1]
init_day = Info[clave][2]
init_trans = 3
num_data = 35
init0 = Info[clave][2] + timedelta(days=init_trans)
init_index0 = (init0 - Info[clave][2]).days
Data = np.array(read_csv(workdir + 'data/' + data_fnam))

res = (Data[init_index0:].shape[0] - num_data) % (size_window - 1)
init0 = init0 + timedelta(days=res)
init_index0 = (init0 - Info[clave][2]).days
court = (Data[init_index0:].shape[0] - num_data) // (size_window - 1) + 1
court = court - 1
init = init0 + timedelta(days=court * (size_window - 1))
init_index = (init - Info[clave][2]).days


data_all = Data
data = Data[init_index: init_index + num_data]
m = data.shape[0]

#--- Uncomment the following line to plot Figure 4S
# PlotFigsZMs(pred=pred, court=court, blue=True)
