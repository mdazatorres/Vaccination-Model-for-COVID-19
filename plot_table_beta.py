from datetime import date, timedelta
import pickle
import numpy as np
from pandas import read_csv
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
    Data = np.array(read_csv(workdir + 'data/' + data_fnam))
    res = (Data[init_index0:].shape[0] - num_data) % (size_window - 1)
    init0 = init0 + timedelta(days=res)
    init_index0 = (init0 - ZMs[clave][2]).days
    court = (Data[init_index0:].shape[0] - num_data) // (size_window - 1) + 1
    court = court - 1
    init = init0 + timedelta(days=court * (size_window - 1))
    init_index = (init - ZMs[clave][2]).days
    data_all = Data
    data = Data[init_index: init_index + num_data - trim]
    init_open = date(2021, 6, 15)
    inter_index = (init_open - init).days

    return init, inter_index, data, data_all, court


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
        samples_new[:, -4] = beta_estimated

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
    return samples_new, np.array(solns_v_1), solns_plain_v_1


def Tabla_scn1(index):
    Output_MCMC_file = open("Output_MCMC.pkl", "rb")
    Output_MCMC = pickle.load(Output_MCMC_file)
    mean_post = Output_MCMC['mean_post']
    lamda_v1, lamda_v2 = mean_post
    omega_b = np.quantile(samples[:, -3], q=0.5)
    beta_b = np.quantile(samples[:, -4], q=0.5)
    Beta = [0.31, 0.4, 0.5]
    pred = 15
    rates_vac = [1, 0.70, 1.30]

    Tab= np.zeros((len(rates_vac), len(Beta)))
    for i in range(len(rates_vac)):
        samples_new, solns_v, solns_plain_v = output_vaccination(N=N, solns_plain_v=solns_plain, samples=samples, inter_index=inter_index,
                                                                 params=[lamda_v1*rates_vac[i], lamda_v2*rates_vac[i], omega_b, beta_b], baseline=True)
        for j in range(len(Beta)):
            _, solns_v1, solns_plain_v1 = output_vaccination(N=N, solns_plain_v=solns_plain_v, samples=samples_new,
                                                         inter_index=pred, params=[lamda_v1, lamda_v2, omega_b, Beta[j]], baseline=False)
            median = np.quantile(solns_v1[index], q=0.5, axis=0)
            Tab[j][i] = median[-1]

    return Tab


ZMs = ReadInfo()
trim = 0
exit_probs = [0.6, 1]
Pobs_I = 0.85
Pobs_D = 0.9
size_window = 8
workdir = "./../"
clave = 'CA'
out_fnam = clave
data_fnam = clave+'.csv'
N = ZMs[clave][1]
init_day = ZMs[clave][2]
num_data = 35
Region = ZMs[clave][0]
m = num_data
init0, inter_index, data, data_all, court = Initday(clave, ZMs)

#--- Uncomment the following lines to visualize Table 3
# samples = pickle.load(open(workdir + 'output/' + out_fnam + 'court_' + str(court) + '_samples.pkl', 'rb'))
# solns_plain = pickle.load(open(workdir + 'output/' + out_fnam + 'court_' + str(court) + '_solns_plain.pkl', 'rb'))
#
# Tabla_C = Tabla_scn1(index=0)
# base_value = Tabla_C[0][0]
# beta = ['beta_base', "beta1", "beta2"]
# rates = ['same', '30% Reduction', '30% Improvement']
#
# Tabla_per = (Tabla_C-base_value)/base_value
# pd.DataFrame(Tabla_C, beta, columns=rates)
# pd.DataFrame(Tabla_per, beta, columns=rates)
#
# Tabla_D = Tabla_scn1(index=1)
# base_value_D = Tabla_D[0][0]
# Tabla_per_D = (Tabla_D - base_value_D)/base_value_D *100
# pd.DataFrame(Tabla_D, beta, columns=rates)
# pd.DataFrame(Tabla_per_D, beta, columns=rates)
