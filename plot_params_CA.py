from datetime import date, timedelta
import matplotlib.dates as mdates
import datetime as dt
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pandas import  read_csv
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


def params(court, ax, pred, init0, size_window, label, index, q = [10, 25, 50, 75, 90], color='red'):
    workdir = './../'
    samples = pickle.load(open(workdir + 'output/' + out_fnam + 'court_' + str(court) + '_samples.pkl', 'rb'))
    init_m = 5
    beta = samples[:, init_m + index]
    init = init0 + timedelta(days=court * (size_window - 1))
    every = 1
    length = num_data
    shift = length + pred

    if ax == None:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.gca()
    else:
        fig = None
    days = mdates.drange(init, init + dt.timedelta(shift), dt.timedelta(days=1))  # how often do de plot
    sv = -np.ones((len(days), 9))
    for i, day in enumerate(days):
        d = dt.date.fromordinal(int(day))
        sv[i, 0] = d.year
        sv[i, 1] = d.month
        sv[i, 2] = d.day

    for i in range(length):
        sv[i, 4:9] = np.quantile(beta, q=np.array(q) / 100)

    for i in range(length, shift, every):
        sv[i, 4:9] = np.quantile(beta, q=np.array(q) / 100)


    if label == True:
        ax.plot(days[:shift], sv[:shift, 6], '-', linewidth=2, color=color, label=r'Median')
        ax.legend(loc=0, shadow=True)
        ax.fill_between(days[:shift], sv[:shift, 4], sv[:shift, 8], color='blue', alpha=0.25)
        ax.fill_between(days[:shift], sv[:shift, 5], sv[:shift, 7], color='blue', alpha=0.25)
    else:
        ax.plot(days[:size_window], sv[:size_window, 6], '-', linewidth=2, color=color)
        ax.fill_between(days[:size_window], sv[:size_window, 4], sv[:size_window, 8], color='blue', alpha=0.25)
        ax.fill_between(days[:size_window], sv[:size_window, 5], sv[:size_window, 7], color='blue', alpha=0.25)


    if index==0:
        ax.set_ylabel(r" $\beta$",fontsize=35)
    elif index==1:
        ax.set_ylabel(r" $\omega$", fontsize=35)
    else:
        ax.set_ylabel(r" $g$", fontsize=35)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=27))

    ymax = np.max(sv[:, 4:9])
    if label:
        ax.set_xlim(init0-dt.timedelta(1), init + dt.timedelta(shift))
        ax.legend()
    else:

        ax.set_xlim(init0 - dt.timedelta(1), init + dt.timedelta(size_window))

    ax.tick_params(which='major', axis='x')  # , labelrotation=40)
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right", rotation_mode="anchor")
    ax.grid(color='grey', which='both', linestyle='--', linewidth=0.5)

    return ax


def all_params(ax, index, court):
    params(court=0, ax=ax, pred=pred, init0=init0, size_window=size_window, index=index, label=False)
    for i in range(court-2):
        i=i+1
        params(court=i,  ax=ax, pred=pred, init0=init0, size_window=size_window, index=index, label=False)
    params(court=court-1,  ax=ax, pred=pred, init0=init0, size_window=size_window, index=index, label=True)



Info = ReadInfo()
pred = 60
size_window = 8
num_data = 35
clave = 'CA'  # valle de mexico
Region = Info[clave][0]
out_fnam = clave
data_fnam = clave + '.csv'
init0 = Info[clave][2] + timedelta(days=3)
init_index0 = (init0 - Info[clave][2]).days
Data = np.array(read_csv(data_fnam))
court = (Data[init_index0:].shape[0] - num_data) // (size_window - 1) + 1
res = (Data[init_index0:].shape[0] - num_data) % (size_window - 1)
init0 = init0 + timedelta(days=res)
court = court - 1

#--- Uncomment the following lines to plot Figure 5S

# fig, ax = subplots(num=0, figsize=(12, 8))
# all_params(index=0, court=court, ax=ax)
# fig.tight_layout()
# fig.savefig("%s_beta0.png" % out_fnam)
#
# fig, ax = subplots(num=1, figsize=(12, 8))
# all_params(index=1, court=court, ax=ax)
# fig.tight_layout()
# fig.savefig("%s_omega.png" %out_fnam)
#
# fig, ax = subplots(num=3, figsize=(12, 8))
# all_params(index=2, court=court, ax=ax)
# fig.tight_layout()
# fig.savefig("%s_g.png" %out_fnam)




