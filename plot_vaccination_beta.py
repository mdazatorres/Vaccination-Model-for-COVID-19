from datetime import date, timedelta
import matplotlib.dates as mdates
import datetime as dt
import pickle
import numpy as np
from pandas import read_csv
from matplotlib.pyplot import subplots, close
import matplotlib.pyplot as plt
from fm_matrix_v import fm_matrix_v
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


def Initday(clave, ZMs):
    init_trans = 3
    Init0 = ZMs[clave][2]
    init0 = Init0 + timedelta(days=init_trans)
    init_index0 = (init0 - ZMs[clave][2]).days
    Data = np.array(read_csv(data_fnam))
    res = (Data[init_index0:].shape[0] - num_data) % (size_window - 1)
    init0 = init0 + timedelta(days=res)
    init_index0 = (init0 - ZMs[clave][2]).days
    court = (Data[init_index0:].shape[0] - num_data) // (size_window - 1) + 1
    court = court - 1
    court_v = court
    init = init0 + timedelta(days=court_v * (size_window - 1))
    init_index = (init - ZMs[clave][2]).days
    data_all = Data
    data= Data[init_index: init_index + num_data]
    init_open = date(2021, 6, 15)
    inter_index = (init_open - init).days

    return init, inter_index, data, data_all, court_v


def output_vaccination(N, solns_plain_v, samples, inter_index, params, baseline):
    num_days = inter_index
    lamda_v1, lamda_v2, omega, beta = params
    essize = samples.shape[0]

    fm = fm_matrix_v(num_days, N, lamda_v1, lamda_v2)
    samples_new = np.zeros((essize, fm.q + 4))
    w_estimated = np.quantile(samples[:, -3], q=0.5)
    beta_estimated = np.quantile(samples[:, -4], q=0.5)
    if baseline==True:
        X0 = solns_plain_v[:, 0, :]
        samples_new[:, 0] = X0[:, 0]
        samples_new[:, 1] = 0
        samples_new[:, 2] = 0
        samples_new[:, 3:fm.q] = X0[:, 1:]
        samples_new[:, -4:] = samples[:, -4:]
        samples_new[:, -3] = w_estimated

    else:
        X0 = solns_plain_v[:, -1, :]
        samples_new[:, 0] = X0[:, 0]
        samples_new[:, 1] = X0[:, 1]
        samples_new[:, 2] = X0[:, 2]
        samples_new[:, 3:fm.q] = X0[:, 3:fm.q]
        samples_new[:, -4:] = samples[:, -4:]
        samples_new[:, -3] = w_estimated
        samples_new[:, -4] = beta

    solns_v_1 = [np.zeros((essize, num_days)) for i in range(2)]
    solns_plain_v_1 = np.zeros((essize, num_days, fm.q))

    for index, k in enumerate(samples_new):
        tmp = list(fm.solve(k[:-1]))
        soln_v_1 = fm.solve_plain(k[:-1])
        solns_plain_v_1[index, :, :] = soln_v_1[10::10, :]
        for i, sl in enumerate(tmp):
            solns_v_1[i][index, :] = np.cumsum(sl)
    return samples_new, solns_v_1, solns_plain_v_1



def PlotVacc(init, size_window, solns, ty, ax,q, color, label, right_axis, label_cases, data,
             show_data, value,baseline,label_vac):

    if ty == 0:
        data = data[:, 1]  # Infected reported
        title = 'cases'
    else:
        data = data[:, 0]  # Deaths reported
        title = 'deaths'

    Pobs = (Pobs_I, Pobs_D)
    prevalence = data
    solns = np.diff(np.append(np.zeros((solns[ty].shape[0], 1)), Pobs[ty] * solns[ty], axis=1), axis=1)
    ylabel = 'Confirmed ' + title
    title = 'Incidence ' + title
    length = m
    days = mdates.drange(init, init + dt.timedelta(size_window), dt.timedelta(days=1))
    sv = -np.ones((len(days), 8))
    for i, day in enumerate(days):
        d = dt.date.fromordinal(int(day))
        sv[i, 0] = d.year
        sv[i, 1] = d.month
        sv[i, 2] = d.day

    for i in range(size_window):
        sv[i, 3:8] = np.quantile(solns[:, i], q=np.array(q) / 100)
    if label:
        ax.plot(days, sv[:, 5], '-', linewidth=2, color= color, label=r"$\beta$=" + str(value))
    elif baseline:
        ax.plot(days, sv[:, 5], '-', linewidth=2, color=color, label='Baseline model')
    else:
        ax.plot(days, sv[:, 5], '-', linewidth=2, color=color, label=label_vac)

    if show_data:
        if label_cases:
            ax.axvline(x=days[-1] + 0.5, ymin=0, ymax=1, linewidth=2.5, linestyle='dashed', color='gray')
            ax.bar(days[:length], prevalence, color='k', width=0.5, label='Cases', alpha=0.5)
            ax.plot(days[:length], prevalence, 'ko', markersize=2)
        else:
            ax.bar(days[:length], prevalence, color='k', width=0.5, alpha=0.5)
            ax.plot(days[:length], prevalence, 'ko', markersize=2)

    ax.legend(frameon=False, fontsize=19)
    ax.set_xlabel("YEAR: 2021", fontsize=20)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=12))
    ax.tick_params(which='major', axis='x', labelsize=20)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")
    ax.set_ylabel(ylabel, fontsize=20)
    ax.grid(color='grey', which='both', linestyle='--', linewidth=0.5)

    if right_axis:
        ax_p = ax.twinx()
        y1, y2 = ax.get_ylim()
        ax_p.set_ylim(y1 * 1e5 / N, y2 * 1e5 / N)
        ax_p.set_ylabel('per 100,000')
    return ax



def Plot_scn2(workdir='./../', save=True):
    close('all')
    q = [10, 25, 50, 75, 90]
    loc = 1
    Output_MCMC_file = open("Output_MCMC.pkl", "rb")
    Output_MCMC = pickle.load(Output_MCMC_file)
    mean_post = Output_MCMC['mean_post']
    lamda_v1, lamda_v2 = mean_post
    Solns_v =[]
    omega_b = np.quantile(samples[:, -3], q=0.5)
    beta_b = np.quantile(samples[:, -4], q=0.5)
    Beta = [0.31, 0.4, 0.5]
    beta = 0.2
    pred = 30
    params_baseline = [lamda_v1, lamda_v2, omega_b, beta]
    colors=['k', 'b', 'r']
    colors_rates = ['c', 'magenta', 'k']
    soln_rates=[]
    rates_vac = [0.70, 1.30, 1]
    label_vac =['30% Reduction', '30% Improvement', '']

    label_cases = [False, False, True]
    for i in range(len(rates_vac)):
        samples_new, solns_v, solns_plain_v = output_vaccination(N=N, solns_plain_v=solns_plain, samples=samples, inter_index=inter_index,
                                                                 params=[lamda_v1*rates_vac[i], lamda_v2*rates_vac[i], omega_b, beta_b],baseline=True) # baseline model
        soln_rates.append(solns_v)

    for i in range(len(Beta)):
        _, solns_v1, solns_plain_v1 = output_vaccination(N=N, solns_plain_v=solns_plain_v, samples=samples_new,
                                                     inter_index=inter_index, params=[lamda_v1, lamda_v2, omega_b, Beta[i]], baseline=False)
        Solns_v.append(solns_v1)

    fig, ax = subplots(num=1, figsize=(13, 7))
    init = init0 + timedelta(days=(inter_index))
    PlotVacc(init=init, size_window=pred, solns=Solns_v[0], ty=0, ax=ax, q=q,color=colors[0], label=False,
             right_axis=False, label_cases=False,data=data,  show_data=False, value=Beta[0],
             baseline=True, label_vac=None)

    for i in range(len(rates_vac)):
        PlotVacc(init=init0, size_window=inter_index, solns=soln_rates[i], ty=0, ax=ax, q=q,color=colors_rates[i], label=False, right_axis=label_cases[i],
                 label_cases=label_cases[i], data=data, show_data=label_cases[i], value=beta_b,baseline=False, label_vac=label_vac[i])

    for i in range(len(Beta)-1):
        i=i+1
        PlotVacc(init=init, size_window=pred, solns=Solns_v[i], ty=0, ax=ax, q=q, color=colors[i], label=True, right_axis=False, label_cases=False,
                data=data, show_data=False, value=Beta[i], baseline=False, label_vac=None)
    fig.tight_layout()
    if save == True:
        fig.savefig("%s%s_scn2_I.png" % (workdir + 'figures/', out_fnam))

    #--------------------- Death ---------------------
    fig, ax = subplots(num=2, figsize=(13, 7))
    PlotVacc(init=init, size_window=pred, solns=Solns_v[0], ty=1, ax=ax, q=q, color=colors[0], label=False, right_axis=False, label_cases=False,
              data=data, show_data=False, value=Beta[0], baseline=True, label_vac=None)

    for i in range(len(rates_vac)):
        PlotVacc(init=init0, size_window=inter_index, solns=soln_rates[i], ty=1, ax=ax, q=q, color=colors_rates[i],label=False, right_axis=label_cases[i], label_cases=label_cases[i],
                data=data, show_data=label_cases[i], value=beta_b, baseline=False, label_vac=label_vac[i])

    for i in range(len(Beta)-1):
        i=i+1
        PlotVacc(init=init, size_window=pred, solns=Solns_v[i], ty=1, ax=ax, q=q, color=colors[i], label=True, right_axis=True, label_cases=False,
                data=data, show_data=False, value=Beta[i], baseline=False, label_vac=None)
    fig.tight_layout()
    if save == True:
        fig.savefig("%s%s_scn2_D.png" % (workdir + 'figures/', out_fnam))
    return ax


Info = ReadInfo()
exit_probs = [0.6, 1]
Pobs_I = 0.85
Pobs_D = 0.9
size_window = 8
workdir = "./../"
clave = 'CA'
out_fnam = clave
data_fnam = clave+'.csv'
N = Info[clave][1]
init_day = Info[clave][2]
num_data = 35
Region = Info[clave][0]
m = num_data
init0, inter_index, data, data_all, court = Initday(clave, Info)

#--- Uncomment the following lines to plot Figure 2 c-d
# samples = pickle.load(open(workdir + 'output/' + out_fnam + 'court_' + str(court) + '_samples.pkl', 'rb'))
# solns_plain = pickle.load(open(workdir + 'output/' + out_fnam + 'court_' + str(court) + '_solns_plain.pkl', 'rb'))
# Plot_scn2( workdir='./../', save=True)

