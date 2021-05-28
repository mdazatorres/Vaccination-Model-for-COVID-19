import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import matplotlib.dates as mdates
import datetime as dt
plt.rcParams['font.size'] = 18

# ----    Data  -----------
N=39510000
n = 30
data = pd.read_excel('CA-Vaccine.xlsx')
data = data[data['county']=='All CA Counties']
dates = data['administered_date'][-n:]
Dates = pd.to_datetime(dates)
A_data = np.array(data['cumulative_at_least_one_dose'])[-n:]
U_data = np.array(data['cumulative_fully_vaccinated'])[-n:]
V_data = A_data - U_data
W_data = N - A_data

U0 = U_data[0]
A0 = A_data[0]
W0 = N - A0
V0 = V_data[0]
X0 = [W0, V0, U0, A0]

Output_MCMC_file = open("Output_MCMC.pkl", "rb")
Output_MCMC = pickle.load(Output_MCMC_file)
mean_post = Output_MCMC['mean_post']
quantiles = Output_MCMC['quantiles']
t = np.linspace(0, n-1, n)
init = Dates[125]
days = mdates.drange(init, init + dt.timedelta(n), dt.timedelta(days=1))


def Fc(t, p, X0):
    l1 = p[0]
    l2 = p[1]
    W0, V0, U0, A0 = X0

    C = (l1 * W0 / (l1 - l2)) + V0
    W = W0 * np.exp(-l1 * t)
    V = -l1 * W0 * np.exp(-l1 * t) / (l1 - l2) + C * np.exp(-l2 * t)
    U = np.exp(-l1 * t) * W0 * l2 / (l1 - l2) - C * np.exp(-l2 * t) + V0 + W0 + U0
    A = - W0 * np.exp(-l1 * t) + W0 + A0
    return np.array([W, V, U, A])

baseline_model = Fc(t, mean_post, X0)
X0_new = baseline_model[:, -1]


def plot_fitting(ax, index):
    if index==0:
        ax.plot(days, W_data, 'ko', markersize = 5, label='Data')
        ax.set_title('W')
    elif index==1:
        ax.plot(days, V_data, 'ko', markersize = 5, label='Data')
        ax.set_title('$V_1$')
    elif index==2:
        ax.plot(days, U_data, 'ko', markersize = 5, label='Data')
        ax.set_title('$V_2$')
    else:
        ax.plot(days, A_data, 'ko', markersize = 5, label='Data')
        ax.set_title('$A$')

    model_pred = Fc(t, mean_post, X0)
    ax.plot(days, model_pred[index], lw=3, label='Mean posterior')
    ax.set_xlabel("YEAR: 2021")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=8))
    ax.tick_params(which='major', axis='x')
    ax.tick_params(which='major', axis='y')
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")
    ax.legend(frameon=False)
    ax.set_ylabel('Population')


def Fig_3S():
    out_fnam='CA'
    workdir = "./../"
    fig, ax = plt.subplots(num=0, figsize=(12, 9))
    plot_fitting(ax=ax,index=0)
    fig.tight_layout()
    fig.savefig("%s%s_mean_post_W.png" % (workdir, out_fnam))

    fig, ax = plt.subplots(num=1, figsize=(12, 9))
    plot_fitting(ax=ax,index=1)
    fig.tight_layout()
    fig.savefig("%s%s_mean_post_V1.png" % (workdir, out_fnam))

    fig, ax = plt.subplots(num=2, figsize=(12, 9))
    plot_fitting(ax=ax,index=2)
    fig.tight_layout()
    fig.savefig("%s%s_mean_post_V2.png" % (workdir, out_fnam))

    fig, ax = plt.subplots(num=3, figsize=(12, 9))
    plot_fitting(ax=ax,index=3)
    fig.tight_layout()
    fig.savefig("%s%s_mean_post_A.png" % (workdir, out_fnam))

#--- Uncomment the following line to plot Figure S3
#Fig_3S()
