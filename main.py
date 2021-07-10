from itertools import product
import matplotlib
from matplotlib import cm, rc
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import mpl_toolkits
from mpl_toolkits.axes_grid1 import Divider, inset_locator, Size
import numpy as np
import pandas as pd
import seaborn as sns
from sympy import nsolve, Symbol
from joblib import Parallel, delayed
import time 

MY_N_JOBS = 62

GAMMA = 1/6
KAPPA = 0.002 

cms = 1 / 2.54  # centimeters in inches
STEP = 0.05
FMT = '.3f'
FONT_SIZE = 8
f_list = np.arange(0, 1 + STEP, STEP)
T_STEP = 0.1
T_MAX = 3730
T_MAX_SCALE = 730
MAX_VAL = 10000
MIN_VAL = -1
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial'], 'size': FONT_SIZE})
fontdict = {'family': 'sans-serif', 'size': FONT_SIZE}
ncolors = 256
color_array = plt.get_cmap('PRGn')(range(ncolors))
color_array[:128, 0] = 0.5 * (1.0 + color_array[:128, 0])
color_array[128:, 1] = 0.5 * (1.0 + color_array[128:, 1])
map_object = LinearSegmentedColormap.from_list(name='PRGn_alpha', colors=color_array)
plt.register_cmap(cmap=map_object)
SIGNATURES_MAPPING = {"-": 0, "-+":1, "+":2, "+-":3, "+-+":4, "-+-": 5}


def flattened_list(nested_list):
    return [item for sublist in nested_list for item in sublist]

def formula_v(a, alpha, v1, v2, t):
    return (1 - a) * v1 * alpha / (v1 + v2) * (1 - np.exp(-(v1 + v2) * t))


def formula_s(a, v1, t):
    return a + (1 - a) * np.exp(-v1 * t)


def formula_sv(a, alpha, v1, v2, t):
    return 1 - formula_s(a, v1, t) - formula_v(a, alpha, v1, v2, t)


def split_periods(array, delta=0.1):
    if len(array) == 0:
        return []
    result = [[array[0]]]
    last_value = array[0]
    group_id = 0
    for elem in array[1:]:
        if elem - last_value < 1.5 * delta:
            result[group_id].append(elem)
        else:
            group_id += 1
            result.append([elem])
        last_value = elem
    return result


def doubling_time(data, pts, R_max, ax=None):
    if ax is None:
        print('no ax given, return!')
        return
    for pt in pts:
        f = pt['f']
        fv = pt['fv']
        color = pt['color']
        signature = pt['signature']
        if signature == '-':
            continue
        data['Reff'] = (0.5 * (R1(f, R_max) * data['S'] + R2(fv, R_max) * data['Sv']) + 0.5 * (
                (R1(f, R_max) * data['S'] - R2(fv, R_max) * data['Sv']
                 ) ** 2 + 4 * data['S'] * data['Sv'] * R1(f, R_max) ** 2) ** 0.5)
        for i, t in data[(data['Reff'] > 1).diff().fillna(False)].iterrows():
            ax.plot([t['time']] * 2, [0, 100], ':', c=color, linewidth=0.5)
        data['doubling_time'] = 6 / (data['Reff'] - 1) * np.log(2)
        doubling_above_100 = data[data['doubling_time'] > 100]['time'].to_numpy()
        above_100_split = split_periods(doubling_above_100, delta=0.1)
        for above_100 in above_100_split:
            ax.plot(above_100, [100] * len(above_100), '--', c=color, linewidth=0.5)
        data['doubling_time'] = data['doubling_time'].apply(lambda x: np.nan if x > 100 else -np.nan if x < 0 else x)
        data.plot('time', 'doubling_time', ax=ax, label=r'$\mathtt{' + f'{signature}' + r'}$', color=color, linewidth=0.5)

    ax.set_ylabel('D', fontdict=fontdict, labelpad=0)
    ax.set_xlabel('time', fontdict=fontdict)
    legend = ax.legend(ncol=2, fontsize=FONT_SIZE, bbox_to_anchor=[1.1, -0.5], title='Scenarios')
    legend.get_frame().set_linewidth(0.5)
    legend.get_frame().set_edgecolor("yellow")
    legend.get_frame().set_facecolor("white")
    ax.set_ylim([0, 105])
    ax.set_xlim([0, 730])
    ax.set_xticks([0, 182, 365, 365 + 182, 730])
    ax.set_xticklabels(['0 d', '6 m', '1 y', '18 m', '2 y'], fontdict=fontdict)
    ax.set_yticks([0, 100])
    ax.set_yticklabels(['0', '100'], fontdict=fontdict)
    ax.xaxis.set_tick_params(width=0.5)
    ax.yaxis.set_tick_params(width=0.5)


def R1(f, R_max):
    return R_max * (1 - f)


def R2(fv, R_max):
    return R_max * (1 - fv)


def generate_data(a, alpha, v1, v2):
    t = np.arange(0, T_MAX + T_STEP, T_STEP)
    data = pd.DataFrame({'time': t})
    data['S'] = data['time'].apply(lambda x: formula_s(a, v1, x))
    data['Sv'] = data['time'].apply(lambda x: formula_sv(a, alpha, v1, v2, x))
    data['V'] = data['time'].apply(lambda x: formula_v(a, alpha, v1, v2, x))
    return data


def set_elem_signature(f, fv, value, signatures, fmt=FMT):
    fmt = '{:' + fmt + '}'
    #signatures[value] = signatures.get(value, len(signatures))
    #value = signatures[value]
    value = SIGNATURES_MAPPING[value]
    return {'f': fmt.format(f), 'fv': fmt.format(fv), 'value': value}


def set_elem_heatmap(f, fv, value, fmt=FMT):
    fmt = '{:' + fmt + '}'
    return {'f': fmt.format(f), 'fv': fmt.format(fv), 'value': value}


def signature_pref(data):
    signature_mapping = dict()
    heatmap_signature = []
    for f, fv in product(f_list, repeat=2):
        S = data['S']
        Sv = data['Sv']
        data['lambda_m'] = S * (1 - f) - 0.5 + (1 - fv) ** 2 / ((1 - f) * S + (1 - fv) * (1 - S)) * Sv + (
                4 * (1 - f) ** 3 / ((1 - f) * S + (1 - fv) * (1 - S)) * Sv * S + (
                (1 - f) * S - (1 - fv) ** 2 / ((1 - f) * S + (1 - fv) * (1 - S)) * Sv) ** 2) ** 0.5
        d1 = ['*'] + list(data['lambda_m'].apply(lambda x: '+' if x > 0 else '-' if x < 0 else 'x'))
        signature = ''.join([elem for elem, prev in zip(d1[1:], d1[:-1]) if elem != prev])
        if signature.startswith('x'):
            signature = signature[1:]
        if signature == '':  # supporting an edge case
            signature = '-'
        heatmap_signature.append(set_elem_signature(f, fv, signature, signature_mapping))
    return heatmap_signature, signature_mapping

def signature_pref2_old(data, R_max, d, gamma=GAMMA):
    signature_mapping = dict()
    heatmap_signature = []
    r_max = R_max
    for f, fv in product(f_list, repeat=2):
        S = data['S']
        Sv = data['Sv']
        beta = gamma * r_max * (1 - f)
        delta = gamma * r_max * (1 - fv)
        if beta == 0 and delta == 0:
            delta_star = 0
            delta_plus = 0
        else:
            delta_star = delta * delta / (beta * d + delta * (1 - d))
            delta_plus = beta * beta / (beta * d + delta * (1 - d))
        a11 = beta * S - gamma
        a12 = delta_plus * S
        a21 = beta * Sv
        a22 = delta_star * Sv - gamma
        t = a11 + a22
        det = a11 * a22 - a12 * a21
        data['lambda_m'] = (t + np.sqrt(t ** 2 - 4 * det)) / 2
        d1 = ['*'] + list(data['lambda_m'].apply(lambda x: '+' if x > 0 else '-' if x < 0 else 'x'))
        signature = ''.join([elem for elem, prev in zip(d1[1:], d1[:-1]) if elem != prev])
        if signature.startswith('x'):
            signature = signature[1:]
        if signature == '':  # supporting an edge case
            signature = '-'
        heatmap_signature.append(set_elem_signature(f, fv, signature, signature_mapping))
    return heatmap_signature, signature_mapping

def signature_pref2(data, R_max, d, gamma=GAMMA):
    signature_mapping = None #dict()
    #heatmap_signature = []
    r_max = R_max
    
    def signature_pref2_parallel(f):
        res = [None] * len(f_list)
        for idx, fv in enumerate(f_list):
            S = data['S']
            Sv = data['Sv']
            beta = gamma * r_max * (1 - f)
            delta = gamma * r_max * (1 - fv)
            if beta == 0 and delta == 0:
                delta_star = 0
                delta_plus = 0
            else:
                delta_star = delta * delta / (beta * d + delta * (1 - d))
                delta_plus = beta * beta / (beta * d + delta * (1 - d))
            a11 = beta * S - gamma
            a12 = delta_plus * S
            a21 = beta * Sv
            a22 = delta_star * Sv - gamma
            t = a11 + a22
            det = a11 * a22 - a12 * a21
            data['lambda_m'] = (t + np.sqrt(t ** 2 - 4 * det)) / 2
            d1 = ['*'] + list(data['lambda_m'].apply(lambda x: '+' if x > 0 else '-' if x < 0 else 'x'))
            signature = ''.join([elem for elem, prev in zip(d1[1:], d1[:-1]) if elem != prev])
            if signature.startswith('x'):
                signature = signature[1:]
            if signature == '':  # supporting an edge case
                signature = '-'
            res[idx] = set_elem_signature(f, fv, signature, signature_mapping)
        return res

    output = Parallel(n_jobs = MY_N_JOBS)(delayed(signature_pref2_parallel)(f) for f in f_list)
    return flattened_list(output), signature_mapping

def signature_prop(data, R_max):
    signature_mapping = None #dict()
    
    def signature_prop_parallel(f):
        res = [None] * len(f_list)
        for idx, fv in enumerate(f_list):
            data['Reff'] = (0.5 * (R1(f, R_max) * data['S'] + R2(fv, R_max) * data['Sv']) + 0.5 * (
                (R1(f, R_max) * data['S'] - R2(fv, R_max) * data['Sv']
                 ) ** 2 + 4 * data['S'] * data['Sv'] * R1(f, R_max) ** 2) ** 0.5)
            d1 = ['*'] + list(data['Reff'].apply(lambda x: '+' if x > 1 else '-' if x < 1 else 'x'))
            signature = ''.join([elem for elem, prev in zip(d1[1:], d1[:-1]) if elem != prev])
            if signature.startswith('x'):
                signature = signature[1:]
            res[idx] = set_elem_signature(f, fv, signature, signature_mapping)
        return res
     
    output = Parallel(n_jobs = MY_N_JOBS)(delayed(signature_prop_parallel)(f) for f in f_list)
    
    return flattened_list(output), signature_mapping
    
def signature_prop_old(data, R_max):
    signature_mapping = dict()
    heatmap_signature = []
    for f, fv in product(f_list, repeat=2):
        data['Reff'] = (0.5 * (R1(f, R_max) * data['S'] + R2(fv, R_max) * data['Sv']) + 0.5 * (
                (R1(f, R_max) * data['S'] - R2(fv, R_max) * data['Sv']
                 ) ** 2 + 4 * data['S'] * data['Sv'] * R1(f, R_max) ** 2) ** 0.5)
        d1 = ['*'] + list(data['Reff'].apply(lambda x: '+' if x > 1 else '-' if x < 1 else 'x'))
        signature = ''.join([elem for elem, prev in zip(d1[1:], d1[:-1]) if elem != prev])
        if signature.startswith('x'):
            signature = signature[1:]
        heatmap_signature.append(set_elem_signature(f, fv, signature, signature_mapping))
    return heatmap_signature, signature_mapping


def signature_calc_old(data, type_, R_max):
    if type_ == 'prop':
        return signature_prop(data, R_max)
    else:
        if R_max != 4:
            print(f'pref. for R_max different than 4 is not supported ({R_max})')
        else:
            return signature_pref(data)

def signature_calc(data, type_, R_max, d, gamma=GAMMA):
    if type_ == 'prop':
        return signature_prop(data=data, R_max=R_max)
    else:
        return signature_pref2(data=data, R_max=R_max, d=d, gamma=gamma)

def heatmap_pref2_old(data, R_max, d, gamma=GAMMA):
    heatmap = []
    r_max = R_max
    for f, fv in product(f_list, repeat=2):
        S = data['S']
        Sv = data['Sv']
        beta = gamma * r_max * (1 - f)
        delta = gamma * r_max * (1 - fv)
        if beta == 0 and delta == 0:
            delta_star = 0
            delta_plus = 0
        else:
            delta_star = delta * delta / (beta * d + delta * (1 - d))
            delta_plus = beta * beta / (beta * d + delta * (1 - d))
        a11 = beta * S - gamma
        a12 = delta_plus * S
        a21 = beta * Sv
        a22 = delta_star * Sv - gamma
        t = a11 + a22
        det = a11 * a22 - a12 * a21
        data['lambda_m'] = (t + np.sqrt(t ** 2 - 4 * det)) / 2
        if len(data[data['lambda_m'] > 0]) > 0:
            first_above_1 = data[data['lambda_m'] > 0].time.to_numpy()[0]
            if first_above_1 > 0:
                # started as subcritical, will be overcritical
                heatmap.append(set_elem_heatmap(f, fv, first_above_1))  # [(f, fv)] = first_above_1
            else:
                # started as overcritical...
                if len(data[data['lambda_m'] < 0]) > 0:
                    first_below_1 = data[data['lambda_m'] < 0].time.to_numpy()[0]
                    first_below_1_id = data[data['lambda_m'] < 0].index[0]
                    data2 = data[first_below_1_id:]
                    if len(data2[data2['lambda_m'] > 0]) > 0:
                        # ...will be overcritical again
                        first_exit_above_1 = data2[data2['lambda_m'] > 0].time.to_numpy()[0]
                        heatmap.append(
                            set_elem_heatmap(f, fv, first_exit_above_1))  # heatmap[key][(f, fv)] = first_exit_above_1
                    else:
                        # ...not overcritical again within 2 years
                        heatmap.append(
                            set_elem_heatmap(f, fv, MAX_VAL + first_below_1))  # np.nan))#heatmap[key][(f, fv)] = np.nan
                else:
                    # ... always overcritical
                    heatmap.append(set_elem_heatmap(f, fv, MIN_VAL))  # heatmap[key][(f, fv)] = np.nan

        else:
            # under-critical
            heatmap.append(set_elem_heatmap(f, fv, MAX_VAL))  # np.nan))#heatmap[key][(f, fv)] = np.nan
    return heatmap


def heatmap_pref(data):
    heatmap = []
    for f, fv in product(f_list, repeat=2):
        S = data['S']
        Sv = data['Sv']
        data['lambda_m'] = S * (1 - f) - 0.5 + (1 - fv) ** 2 / ((1 - f) * S + (1 - fv) * (1 - S)) * Sv + (
                4 * (1 - f) ** 3 / ((1 - f) * S + (1 - fv) * (1 - S)) * Sv * S + (
                (1 - f) * S - (1 - fv) ** 2 / ((1 - f) * S + (1 - fv) * (1 - S)) * Sv) ** 2) ** 0.5
        if len(data[data['lambda_m'] > 0]) > 0:
            first_above_1 = data[data['lambda_m'] > 0].time.to_numpy()[0]
            if first_above_1 > 0:
                # started as subcritical, will be overcritical
                heatmap.append(set_elem_heatmap(f, fv, first_above_1))  # [(f, fv)] = first_above_1
            else:
                # started as overcritical...
                if len(data[data['lambda_m'] < 0]) > 0:
                    first_below_1 = data[data['lambda_m'] < 0].time.to_numpy()[0]
                    first_below_1_id = data[data['lambda_m'] < 0].index[0]
                    data2 = data[first_below_1_id:]
                    if len(data2[data2['lambda_m'] > 0]) > 0:
                        # ...will be overcritical again
                        first_exit_above_1 = data2[data2['lambda_m'] > 0].time.to_numpy()[0]
                        heatmap.append(
                            set_elem_heatmap(f, fv, first_exit_above_1))  # heatmap[key][(f, fv)] = first_exit_above_1
                    else:
                        # ...not overcritical again within 2 years
                        heatmap.append(
                            set_elem_heatmap(f, fv, MAX_VAL + first_below_1))  # np.nan))#heatmap[key][(f, fv)] = np.nan
                else:
                    # ... always overcritical
                    heatmap.append(set_elem_heatmap(f, fv, MIN_VAL))  # heatmap[key][(f, fv)] = np.nan

        else:
            # under-critical
            heatmap.append(set_elem_heatmap(f, fv, MAX_VAL))  # np.nan))#heatmap[key][(f, fv)] = np.nan
    return heatmap


def heatmap_pref2(data, R_max, d, gamma=GAMMA):
    
    #heatmap = []
    r_max = R_max
    
    def heatmap_pref2_parallel(f):

        res = [None] * len(f_list)
        for idx, fv in enumerate(f_list):
            S = data['S']
            Sv = data['Sv']
            beta = gamma * r_max * (1 - f)
            delta = gamma * r_max * (1 - fv)
            if beta == 0 and delta == 0:
                delta_star = 0
                delta_plus = 0
            else:
                delta_star = delta * delta / (beta * d + delta * (1 - d))
                delta_plus = beta * beta / (beta * d + delta * (1 - d))
            a11 = beta * S - gamma
            a12 = delta_plus * S
            a21 = beta * Sv
            a22 = delta_star * Sv - gamma
            t = a11 + a22
            det = a11 * a22 - a12 * a21
            data['lambda_m'] = (t + np.sqrt(t ** 2 - 4 * det)) / 2
            if len(data[data['lambda_m'] > 0]) > 0:
                first_above_1 = data[data['lambda_m'] > 0].time.to_numpy()[0]
                if first_above_1 > 0:
                    # started as subcritical, will be overcritical
                    res[idx] = set_elem_heatmap(f, fv, first_above_1)  # [(f, fv)] = first_above_1
                else:
                    # started as overcritical...
                    if len(data[data['lambda_m'] < 0]) > 0:
                        first_below_1 = data[data['lambda_m'] < 0].time.to_numpy()[0]
                        first_below_1_id = data[data['lambda_m'] < 0].index[0]
                        data2 = data[first_below_1_id:]
                        if len(data2[data2['lambda_m'] > 0]) > 0:
                            # ...will be overcritical again
                            first_exit_above_1 = data2[data2['lambda_m'] > 0].time.to_numpy()[0]
                            res[idx] = set_elem_heatmap(f, fv, first_exit_above_1)  # heatmap[key][(f, fv)] = first_exit_above_1
                        else:
                            # ...not overcritical again within 2 years
                            res[idx] = set_elem_heatmap(f, fv, MAX_VAL + first_below_1)  # np.nan))#heatmap[key][(f, fv)] = np.nan
                    else:
                        # ... always overcritical
                        res[idx] = set_elem_heatmap(f, fv, MIN_VAL)  # heatmap[key][(f, fv)] = np.nan

            else:
                # under-critical
                res[idx] = set_elem_heatmap(f, fv, MAX_VAL)  # np.nan))#heatmap[key][(f, fv)] = np.nan

        return res 

    output = Parallel(n_jobs=MY_N_JOBS)(delayed(heatmap_pref2_parallel)(f) for f in f_list)
    
    return flattened_list(output)



def heatmap_prop(data, R_max):

    def heatmap_prop_parallel(f):
        res = [None] * len(f_list)
        for idx, fv in enumerate(f_list):
            data['Reff'] = (0.5 * (R1(f, R_max) * data['S'] + R2(fv, R_max) * data['Sv']) + 0.5 * (
                    (R1(f, R_max) * data['S'] - R2(fv, R_max) * data['Sv']) ** 2 + 4 * data['S'] * data['Sv'] * R1(f,
                                                                                                                R_max) ** 2) ** 0.5)        
            if len(data[data['Reff'] > 1]) > 0:
                first_above_1 = data[data['Reff'] > 1].time.to_numpy()[0]
                if first_above_1 > 0:
                    # started as subcritical, will be overcritical
                    res[idx] = set_elem_heatmap(f, fv, first_above_1)  # [(f, fv)] = first_above_1
                else:
                    # started as overcritical...
                    if len(data[data['Reff'] < 1]) > 0:
                        first_below_1 = data[data['Reff'] < 1].time.to_numpy()[0]
                        first_below_1_id = data[data['Reff'] < 1].index[0]
                        data2 = data[first_below_1_id:]
                        if len(data2[data2['Reff'] > 1]) > 0:
                            # ...will be overcritical again
                            first_exit_above_1 = data2[data2['Reff'] > 1].time.to_numpy()[0]
                            res[idx] = set_elem_heatmap(f, fv, first_exit_above_1)  # heatmap[key][(f, fv)] = first_exit_above_1
                        else:
                            # ...not overcritical again within 2 years
                            res[idx] = set_elem_heatmap(f, fv, MAX_VAL + first_below_1)  # np.nan))#heatmap[key][(f, fv)] = np.nan
                    else:
                        # ... always overcritical
                        res[idx] = set_elem_heatmap(f, fv, MIN_VAL)  # heatmap[key][(f, fv)] = np.nan

            else:
                # under-critical
                res[idx] = set_elem_heatmap(f, fv, MAX_VAL)  # np.nan))#heatmap[key][(f, fv)] = np.nan
        
        return res 

    output = Parallel(n_jobs=MY_N_JOBS)(delayed(heatmap_prop_parallel)(f) for f in f_list)
    
    return flattened_list(output)


def heatmap_prop_old(data, R_max):
    heatmap = []
    for f, fv in product(f_list, repeat=2):
        data['Reff'] = (0.5 * (R1(f, R_max) * data['S'] + R2(fv, R_max) * data['Sv']) + 0.5 * (
                (R1(f, R_max) * data['S'] - R2(fv, R_max) * data['Sv']) ** 2 + 4 * data['S'] * data['Sv'] * R1(f,
                                                                                                               R_max) ** 2) ** 0.5)
        if len(data[data['Reff'] > 1]) > 0:
            first_above_1 = data[data['Reff'] > 1].time.to_numpy()[0]
            if first_above_1 > 0:
                # started as subcritical, will be overcritical
                heatmap.append(set_elem_heatmap(f, fv, first_above_1))  # [(f, fv)] = first_above_1
            else:
                # started as overcritical...
                if len(data[data['Reff'] < 1]) > 0:
                    first_below_1 = data[data['Reff'] < 1].time.to_numpy()[0]
                    first_below_1_id = data[data['Reff'] < 1].index[0]
                    data2 = data[first_below_1_id:]
                    if len(data2[data2['Reff'] > 1]) > 0:
                        # ...will be overcritical again
                        first_exit_above_1 = data2[data2['Reff'] > 1].time.to_numpy()[0]
                        heatmap.append(
                            set_elem_heatmap(f, fv, first_exit_above_1))  # heatmap[key][(f, fv)] = first_exit_above_1
                    else:
                        # ...not overcritical again within 2 years
                        heatmap.append(
                            set_elem_heatmap(f, fv, MAX_VAL + first_below_1))  # np.nan))#heatmap[key][(f, fv)] = np.nan
                else:
                    # ... always overcritical
                    heatmap.append(set_elem_heatmap(f, fv, MIN_VAL))  # heatmap[key][(f, fv)] = np.nan

        else:
            # under-critical
            heatmap.append(set_elem_heatmap(f, fv, MAX_VAL))  # np.nan))#heatmap[key][(f, fv)] = np.nan
    return heatmap


def heatmap_calc_old(data, type_, R_max):
    if type_ == 'prop':
        return heatmap_prop(data, R_max)
    else:
        if R_max != 4:
            print(f'pref. for R_max different than 4 is not supported ({R_max})')
        else:
            return heatmap_pref(data)

def heatmap_calc(data, type_, R_max, d, gamma=GAMMA):
    if type_ == 'prop':
        return heatmap_prop(data, R_max)
    else:
        return heatmap_pref2(data, R_max=R_max, d=d, gamma=gamma)

def plot_figure(heatmap, reff_heatmap, heatmap_signature, key='', ax=None, add_title=True, pts=None,
                labels=None, cax1=None, cax2=None, cax3=None):
    if ax is None:
        return

    font = matplotlib.font_manager.FontProperties(family='sans-serif', size=FONT_SIZE)
    scale = int(np.round((1 / STEP) // 10))

    df0 = pd.DataFrame(heatmap_signature)
    df0 = df0.pivot(index='fv', columns='f', values='value').astype(float)
    df0.sort_index(level=0, ascending=False, inplace=True)
    np0 = df0.to_numpy()
    z0 = np.zeros_like(np0, dtype=bool)
    z0[:, :-1] = (np0[:, 1:] != np0[:, :-1])
    z0[:-1, :] += (np0[1:, :] != np0[:-1, :])
    df0[:] = z0

    cmap1 = cm.get_cmap('PRGn_alpha')
    cmap2 = cm.get_cmap('spring_r')
    cmap3 = cm.get_cmap('seismic')

    df = pd.DataFrame(heatmap)
    df = df.pivot(index='fv', columns='f', values='value').astype(float)
    df.sort_index(level=0, ascending=False, inplace=True)
    df0.columns = pd.Series([f'{float(c):.1f}' for c in list(df.columns)], name='f')
    df0.index = pd.Series([f'{float(c):.1f}' for c in list(df.index)], name='fv')
    matrix_1 = (df >= MAX_VAL) * (1 - z0)

    sns.set(font_scale=1.0)

    sns.heatmap(df, ax=ax, mask=matrix_1, cmap=cmap1, vmin=0, vmax=T_MAX_SCALE, square=True, xticklabels=scale,
                yticklabels=scale, fmt='.1f', cbar=False, cbar_kws={'label': 'Days until overcriticality',
                                                                    'orientation': 'vertical', 'extend': 'both'})
    if add_title:
        ax.set_title(key, fontdict=fontdict)
    df = df - MAX_VAL
    matrix_2 = (df <= -1) * (1 - z0)

    sns.heatmap(df, ax=ax, mask=matrix_2, cmap=cmap2, vmin=0, vmax=T_MAX_SCALE, square=True, xticklabels=scale,
                yticklabels=scale, fmt='.1f', cbar=False, cbar_kws={'label': 'Days until subcriticality',
                                                                    'orientation': 'vertical', 'extend': 'max'})
    decimals = [f'{i / 10:.1f}' for i in range(11)]
    ticklabels = []
    for decimal in decimals[:-1]:
        ticklabels.extend([decimal] + [''] * 9)
    ticklabels.extend([decimals[-1]])
    sns.heatmap(df0, ax=ax, mask=~z0, vmin=0, vmax=1, cmap='gray_r', square=True, xticklabels=scale, yticklabels=scale,
                cbar=False)

    ax.plot([10 * scale + 1, 0], [0, 10 * scale + 1], 'w--')
    cnfont = {'fontname': 'Courier New', 'size': 8}

    box_props = dict(boxstyle='round', facecolor='white', alpha=0.75)
    arrow_props = dict(arrowstyle = "->", connectionstyle = "arc3, rad=0.1", fc = "white", alpha=0.95)
    if pts is not None:
        for i, pt in enumerate(pts):
            ax.plot((pt['f']) * scale * 10 + 0.5, (1 - (pt['fv'])) * scale * 10 + 0.5, 'k.', mew=0.6)
            ax.plot((pt['f']) * scale * 10 + 0.5, (1 - (pt['fv'])) * scale * 10 + 0.5, '.', mew=0, color=pt['color'])
    if labels is not None:
        for i, label in enumerate(labels):
            #ax.text((label['f']) * scale * 10 + 0.5, (1 - (label['fv'])) * scale * 10 + 0.5, label['signature'],
            #        color=label['color'], bbox = box_props, **cnfont)
            arrow_style = None
            arrow_to = ((pts[i]['f']) * scale * 10 + 0.5, (1 - (pts[i]['fv'])) * scale * 10 + 0.5) 
            if label.get('arrow1'):
                arrow_style = arrow_props
                arrow_to = (label['f1arr'] * scale * 10 + 0.5, (1 - label['fv1arr']) * scale * 10 + 0.5)
            ax.annotate(label['signature'], xy = arrow_to , xycoords = 'data', 
                xytext = ((label['f']) * scale * 10 + 0.5, (1 - (label['fv'])) * scale * 10 + 0.5), textcoords = 'data', **cnfont,
                va = 'center', ha = 'center', bbox = box_props, arrowprops = arrow_style)

    df = pd.DataFrame(reff_heatmap)
    df = df.pivot(index='fv', columns='f', values='value').astype(float)
    df.sort_index(level=0, ascending=False, inplace=True)
    df[:] = np.flipud(np.tril(np.rot90(df.to_numpy(), k=3)))

    df.columns = pd.Series([f'{float(c):.1f}' for c in list(df0.columns)], name='f/fv')
    df.index = pd.Series([f'{float(c):.1f}' for c in list(df0.index)], name='fv/f')
    vmax = 2.0
    h = sns.heatmap(df, cbar=False, ax=ax, mask=(df == 0), cmap=cmap3, vmin=0.0, vmax=vmax, square=True,
                    xticklabels=int(scale), yticklabels=int(scale), fmt='.1f', cbar_kws={
            'label': 'Asymptotic R*',
            'orientation': 'horizontal'})

    cb = h.figure.colorbar(h.collections[3], cax=cax1,
                           label='Asymptotic R*',
                           orientation='horizontal', extend='max', ticks=[0, 0.5, 1, 1.5, 2.0])  # Display colorbar
    cb.ax.set_xticklabels(['0.0', '0.5', '1.0', '1.5', r'$\geq$' + '2.0'])
    cb.ax.tick_params(labelsize=FONT_SIZE, width=0.5, length=4, pad=2)  # Set the colorbar scale font size.
    text = cb.ax.xaxis.label
    text.set_font_properties(font)
    cb = h.figure.colorbar(h.collections[0], cax=cax2,
                           label='Time until overcriticality',
                           orientation='horizontal', extend='both', ticks=[0, 182, 365, 365 + 182, 730])
    cb.ax.set_xticklabels(['0d', '6m', '1y', '18m', r'$\geq$' + '2y'])
    cb.ax.tick_params(labelsize=FONT_SIZE, width=0.5, length=4, pad=2)  # Set the colorbar scale font size.
    text = cb.ax.xaxis.label
    text.set_font_properties(font)
    cb = h.figure.colorbar(h.collections[1], cax=cax3,
                           label='Time until subcriticality',
                           orientation='horizontal', extend='max',
                           ticks=[0, 182, 365, 365 + 182, 730])  # Display colorbar
    cb.ax.set_xticklabels(['0d', '6m', '1y', '18m', r'$\geq$' + '2y'])

    cb.ax.tick_params(labelsize=FONT_SIZE, width=0.5, length=4, pad=2)  # Set the colorbar scale font size.
    text = cb.ax.xaxis.label
    text.set_font_properties(font)

    z0 = np.flipud(np.tril(np.rot90(z0, k=3)))
    df0[:] = z0
    df0.columns = pd.Series([f'{float(c):.1f}' for c in list(df.columns)], name='f')
    df0.index = pd.Series([f'{float(c):.1f}' for c in list(df.index)], name='fv')

    sns.heatmap(df0, ax=ax, mask=~z0, vmin=0, vmax=1, cmap='gray_r',
                square=True, xticklabels=False, yticklabels=False, cbar=False)

    def forward_right(x):
        return 1 - (x - 0.5) / scale / 10

    def forward_top(x):
        return (x - 0.5) / scale / 10

    secax = ax.secondary_xaxis('top')
    secax.set_xlabel(r'$f_v$', fontdict=fontdict, labelpad=14)
    secax.set_xticks([])
    secay = ax.secondary_yaxis('right')
    secay.set_ylabel(r'$f_v$', fontdict=fontdict, labelpad=12)
    secay.set_yticks([])
    ax.set_ylabel('f', fontdict=fontdict, labelpad=0)
    ax.set_xlabel('f', fontdict=fontdict, labelpad=0)
    series = list(np.arange(0.5, scale * 10 + 0.51, 2 * scale))
    xseries = series
    yseries = series
    ax.set_xticks(xseries)
    ax.set_yticks(yseries)
    yticks = list([f'{forward_right(x):.1f}' for x in ax.get_yticks()])
    xticks = list([f'{forward_top(x):.1f}' for x in ax.get_xticks()])
    ax.set_yticklabels(yticks, size=FONT_SIZE)
    ax.set_xticklabels(xticks, size=FONT_SIZE)
    for ax_ in [ax]:
        ax_.tick_params(left=True, labelleft=True, top=True, labeltop=True,
                        right=True, labelright=True, bottom=True, labelbottom=True, pad=0)
    for ax_ in [secax, secay]:
        ax_.tick_params(left=False, labelleft=True, top=False, labeltop=True,
                        right=False, labelright=True, bottom=False, labelbottom=True, pad=0)
    plt.sca(ax)
    plt.xticks(rotation=0)
    plt.yticks(rotation=90)
    ax.xaxis.set_tick_params(width=0.5, length=2, pad=2)
    ax.yaxis.set_tick_params(width=0.5, length=2, pad=2)
    if pts is not None:
        for i, pt in enumerate(pts):
            ax.plot((pt['fv']) * scale * 10 + 0.5, (1 - (pt['f'])) * scale * 10 + 0.5, 'k.', mew=.6)
            ax.plot((pt['fv']) * scale * 10 + 0.5, (1 - (pt['f'])) * scale * 10 + 0.5, '.', mew=0, color=pt['color'])

    if labels is not None:
        for i, label in enumerate(labels):
            fv = label['fv']
            f = label['f']
            if label.get('special', False):
                fv = label['fv2']
                f = label['f2']
            arrow_style = None
            arrow_to = ((pts[i]['fv']) * scale * 10 + 0.5, (1 - (pts[i]['f'])) * scale * 10 + 0.5) 
            if label.get('arrow2'):
                arrow_style = arrow_props
                arrow_to = (label['fv2arr'] * scale * 10 + 0.5, (1 - label['f2arr']) * scale * 10 + 0.5)
            ax.annotate(label['signature'], xy = arrow_to , xycoords = 'data', 
                xytext = (fv * scale * 10 + 0.5, (1 - f) * scale * 10 + 0.5), textcoords = 'data', **cnfont,
                va = 'center', ha = 'center', bbox = box_props, arrowprops = arrow_style)
            
            #ax.text(fv * scale * 10 + 0.5, (1 - f) * scale * 10 + 0.5, label['signature'], color=label['color'],
            #        bbox = box_props, **cnfont)

def asymptotic_data(a, alpha, v1, v2):
    return {'S': a,
            'Sv': (1 - a) * (1 - alpha / (1 + v2 / v1))}


def reff_prop(data, R_max):

    def reff_prop_parallel(f): 
        res = [0] * len(f_list)
        for idx, fv in enumerate(f_list):
            Reff = (0.5 * (R1(f, R_max) * data['S'] + R2(fv, R_max) * data['Sv']) + 0.5 * ((R1(f, R_max) * data['S'] - R2(fv, R_max) * data['Sv']) ** 2 + 4 * data['S'] * data['Sv'] * R1(f, R_max) ** 2) ** 0.5)
            res[idx] = set_elem_heatmap(f, fv, Reff)
        return res

    output = Parallel(n_jobs=MY_N_JOBS)(delayed(reff_prop_parallel)(f) for f in f_list)
    return flattened_list(output)

def reff_prop_old(data, R_max):
    heatmap = []
    for f, fv in product(f_list, repeat=2):
        Reff = (0.5 * (R1(f, R_max) * data['S'] + R2(fv, R_max) * data['Sv']) + 0.5 * (
                (R1(f, R_max) * data['S'] - R2(fv, R_max) * data['Sv']
                 ) ** 2 + 4 * data['S'] * data['Sv'] * R1(f, R_max) ** 2) ** 0.5)
        heatmap.append(set_elem_heatmap(f, fv, Reff))
    return heatmap


def reff_pref(data):
    heatmap = []
    gamma = 1 / 6
    for f, fv in product(f_list, repeat=2):
        S = data['S']
        Sv = data['Sv']
        if f == 1 and fv == 1:
            lambda_m = 0
        else:
            lambda_m = S * (1 - f) - 0.5 + (1 - fv) ** 2 / ((1 - f) * S + (1 - fv) * (1 - S)) * Sv + (
                    4 * (1 - f) ** 3 / ((1 - f) * S + (1 - fv) * (1 - S)) * Sv * S + (
                    (1 - f) * S - (1 - fv) ** 2 / ((1 - f) * S + (1 - fv) * (1 - S)) * Sv) ** 2) ** 0.5
        lambda_m = 1 / 3 * lambda_m
        Reff = lambda_m / gamma + 1
        heatmap.append(set_elem_heatmap(f, fv, Reff))
    return heatmap

def reff_pref2(data, R_max, d, gamma=GAMMA):
    heatmap = []
    r_max = R_max
    for f, fv in product(f_list, repeat=2):
        S = data['S']
        Sv = data['Sv']
        beta = gamma * r_max * (1 - f)
        delta = gamma * r_max * (1 - fv)
        if beta == 0 and delta == 0:
            delta_star = 0
            delta_plus = 0
        else:
            delta_star = delta * delta / (beta * d + delta * (1 - d))
            delta_plus = beta * beta / (beta * d + delta * (1 - d))
        a11 = beta * S - gamma
        a12 = delta_plus * S
        a21 = beta * Sv
        a22 = delta_star * Sv - gamma
        t = a11 + a22
        det = a11 * a22 - a12 * a21
        lambda_m = (t + np.sqrt(t ** 2 - 4 * det)) / 2
        Reff = lambda_m / gamma + 1
        heatmap.append(set_elem_heatmap(f, fv, Reff))
    return heatmap

def reff_both_old(data, R_max, type_):
    if type_ == 'prop':
        return reff_prop(data, R_max)
    else:
        assert R_max == 4, 'Reff does not support R_max different than 4'
        return reff_pref(data)

def reff_both(data, R_max, type_, d, gamma=GAMMA):
    if type_ == 'prop':
        return reff_prop(data=data, R_max=R_max)
    else:
        return reff_pref2(data=data, R_max=R_max, d=d, gamma=gamma)


def plot_pts(data, pts, R_max, ax=None, save_timelines=True):
    if ax is None:
        print('ax none - return')
        return
    for pt in pts:
        f = pt['f']
        fv = pt['fv']
        color = pt['color']
        signature = pt['signature']
        data['Reff'] = (0.5 * (R1(f, R_max) * data['S'] + R2(fv, R_max) * data['Sv']) + 0.5 * (
                (R1(f, R_max) * data['S'] - R2(fv, R_max) * data['Sv']
                 ) ** 2 + 4 * data['S'] * data['Sv'] * R1(f, R_max) ** 2) ** 0.5)
        data.plot('time', 'Reff', ax=ax, label=f'Scenario {signature}', color=color, linewidth=0.5)
        if save_timelines:
            data.to_csv(f'scenario_{signature}__line_{color}.csv', index=False)
    ax.plot([0, 730], [1.0, 1.0], 'k--', label='Reff=1.0', linewidth=0.5)
    ax.set_ylabel('R*', fontdict=fontdict)
    ax.set_xlabel('time', fontdict=fontdict)
    ax.get_legend().remove()
    ax.set_ylim([0.0, 2.5])
    ax.set_xlim([0, 730])
    ax.set_xticks([0, 182, 365, 365 + 182, 730])
    ax.set_xticklabels(['0 d', '6 m', '1 y', '18 m', '2 y'], fontdict=fontdict)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['0', '1', '2'], fontdict=fontdict)

    ax.xaxis.set_tick_params(width=0.5)
    ax.yaxis.set_tick_params(width=0.5)


def fig1(scenarios, pts):
    matplotlib.rcParams['axes.linewidth'] = 0.5  # set the value globally
    fig_width = 8.9
    fig_height = 4.9
    fig = plt.figure(figsize=(fig_width * cms, fig_height * cms), dpi=300)
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial'], 'size': FONT_SIZE})

    h = [1.0, 3.0, 1.5, 3.0, 0.4]
    v = [0.4, 2.0, 0.6, 1.1, 0.8][::-1]
    # Sizes are in inches.
    horiz = [Size.Fixed(h_ * cms) for h_ in h]
    vert = [Size.Fixed(v_ * cms) for v_ in v]

    rect = (0.0, 0.0, 1.0, 1.0)
    # Divide the axes rectangle into a grid with sizes specified by horiz * vert.
    divider = Divider(fig, rect, horiz, vert, aspect=False)
    # The rect parameter will actually be ignored and overridden by axes_locator.
    ax1 = fig.add_axes(rect, axes_locator=divider.new_locator(nx=1, ny=3))
    ax2 = fig.add_axes(rect, axes_locator=divider.new_locator(nx=3, ny=3))
    axa = fig.add_axes(rect, axes_locator=divider.new_locator(nx=0, ny=4), frameon=False)
    axb = fig.add_axes(rect, axes_locator=divider.new_locator(nx=2, ny=4), frameon=False)
    for iax in [axa, axb]:
        iax.set_yticks([]), iax.set_xticks([])
    axa.text(0.1, 0.4, 'a', weight='bold', size=FONT_SIZE, fontdict=fontdict)
    axb.text(0.2, 0.4, 'b', weight='bold', size=FONT_SIZE, fontdict=fontdict)

    (a, alpha, v1, v2, type_, R_max, _, _) = scenarios[0]
    data = generate_data(a, alpha, v1, v2)
    doubling_time(data=data, pts=pts, R_max=R_max, ax=ax2)
    plot_pts(data=data, pts=pts, R_max=R_max, ax=ax1, save_timelines=True)
    fig.savefig('Fig1.png')
    fig.savefig('Fig1.pdf')


def fig1b(scenarios, pts):
    matplotlib.rcParams['axes.linewidth'] = 0.5  # set the value globally
    fig_width = 4.4
    fig_height = 4.9
    fig = plt.figure(figsize=(fig_width * cms, fig_height * cms), dpi=300)
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial'], 'size': FONT_SIZE})

    h = [1.0, 3.0, 0.4]
    v = [0.4, 2.0, 0.6, 1.1, 0.8][::-1]
    # Sizes are in inches.
    horiz = [Size.Fixed(h_ * cms) for h_ in h]
    vert = [Size.Fixed(v_ * cms) for v_ in v]

    rect = (0.0, 0.0, 1.0, 1.0)
    # Divide the axes rectangle into a grid with sizes specified by horiz * vert.
    divider = Divider(fig, rect, horiz, vert, aspect=False)
    # The rect parameter will actually be ignored and overridden by axes_locator.
    ax2 = fig.add_axes(rect, axes_locator=divider.new_locator(nx=1, ny=3))

    (a, alpha, v1, v2, type_, R_max, _, _) = scenarios[0]
    data = generate_data(a, alpha, v1, v2)
    doubling_time(data=data, pts=pts, R_max=R_max, ax=ax2)
    fig.savefig('Fig1b.png')
    fig.savefig('Fig1b.pdf')


def fig2(scenarios, pts0=None, labels0=None, tab1_5 = True):
    fig = plt.figure(figsize=(18.3 * cms, 14.6 * cms), dpi=200)
    h = [0.2, 5.7, 0.15] * 3 + [0.15]
    v = ([0.2] + [0.6, 5.7, 0.2] * 2 + [0.1] + [0.4, 0.9])[::-1]
    horiz = [Size.Fixed(h_ * cms) for h_ in h]
    vert = [Size.Fixed(v_ * cms) for v_ in v]
    rect = (0.0, 0.0, 1.0, 1.0)
    # Divide the axes rectangle into a grid with sizes specified by horiz * vert.
    divider = Divider(fig, rect, horiz, vert, aspect=False)
    # The rect parameter will actually be ignored and overridden by axes_locator.
    ax1 = fig.add_axes(rect, axes_locator=divider.new_locator(nx=1, ny=len(v) - 1 - 2), frameon=False)
    ax2 = fig.add_axes(rect, axes_locator=divider.new_locator(nx=4, ny=len(v) - 1 - 2), frameon=False)
    ax3 = fig.add_axes(rect, axes_locator=divider.new_locator(nx=7, ny=len(v) - 1 - 2), frameon=False)
    ax4 = fig.add_axes(rect, axes_locator=divider.new_locator(nx=1, ny=len(v) - 1 - 5), frameon=False)
    ax5 = fig.add_axes(rect, axes_locator=divider.new_locator(nx=4, ny=len(v) - 1 - 5), frameon=False)
    ax6 = fig.add_axes(rect, axes_locator=divider.new_locator(nx=7, ny=len(v) - 1 - 5), frameon=False)
    cax1_o = fig.add_axes(rect, axes_locator=divider.new_locator(nx=1, ny=len(v) - 1 - 8), frameon=False)
    cax2_o = fig.add_axes(rect, axes_locator=divider.new_locator(nx=4, ny=len(v) - 1 - 8), frameon=False)
    cax3_o = fig.add_axes(rect, axes_locator=divider.new_locator(nx=7, ny=len(v) - 1 - 8), frameon=False)
    axa = fig.add_axes(rect, axes_locator=divider.new_locator(nx=0, ny=len(v) - 1 - 1), frameon=False)
    axb = fig.add_axes(rect, axes_locator=divider.new_locator(nx=3, ny=len(v) - 1 - 1), frameon=False)
    axc = fig.add_axes(rect, axes_locator=divider.new_locator(nx=6, ny=len(v) - 1 - 1), frameon=False)
    axd = fig.add_axes(rect, axes_locator=divider.new_locator(nx=0, ny=len(v) - 1 - 4), frameon=False)
    axe = fig.add_axes(rect, axes_locator=divider.new_locator(nx=3, ny=len(v) - 1 - 4), frameon=False)
    axf = fig.add_axes(rect, axes_locator=divider.new_locator(nx=6, ny=len(v) - 1 - 4), frameon=False)
    cax1 = mpl_toolkits.axes_grid1.inset_locator.zoomed_inset_axes(cax1_o, 0.8 / 2, loc='center')
    cax2 = mpl_toolkits.axes_grid1.inset_locator.zoomed_inset_axes(cax2_o, 0.8 / 730, loc='center')
    cax3 = mpl_toolkits.axes_grid1.inset_locator.zoomed_inset_axes(cax3_o, 0.8 / 730, loc='center')
    caxs = [cax1_o, cax2_o, cax3_o]
    axs = [ax1, ax2, ax3, ax4, ax5, ax6]
    axletters = [axa, axb, axc, axd, axe, axf]
    for iax in caxs + axs + axletters:
        iax.set_yticks([]), iax.set_xticks([])

    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial'], 'size': FONT_SIZE})
    matplotlib.rcParams.update({'font.size': FONT_SIZE})
    for scenario, ax, letter_ax in zip(scenarios, axs, axletters):
        (a, alpha, v1, v2, type_, R_max, scen_, letter) = scenario
        asymptotic_data_point = asymptotic_data(a, alpha, v1, v2)
        #reff_heatmap = reff_both(asymptotic_data_point, R_max, type_) # old version before new preferential
        reff_heatmap = reff_both(data=asymptotic_data_point, R_max=R_max, type_=type_, d=a, gamma=GAMMA)
        data = generate_data(a, alpha, v1, v2)
        #heatmap_signature, _ = signature_calc(data, type_=type_, R_max=R_max)
        #heatmap = heatmap_calc(data, type_=type_, R_max=R_max)
        heatmap_signature, _ = signature_calc(data, type_=type_, R_max=R_max, d=a, gamma=GAMMA)
        heatmap = heatmap_calc(data, type_=type_, R_max=R_max, d=a, gamma=GAMMA)
        pts = None
        labels = None
        if scen_ == '*':
            pts = pts0
            labels = labels0
            if tab1_5:
                key = r'$a$' + f' = {alpha:.2f}, ' + r'$\upsilon$' + f' = {v1}, '
                key += r'$\omega$' + f' = {v2}, ' + r'$d$' + f' = {a:.2f}'
            else:
                key = f'change to reference: ' + r'$a$' + f' = {alpha:.2f}, ' + r'$\upsilon$' + f' = {v1}'
        else:
            change = ''
            if scen_ == 0:
                if tab1_5:
                    change = r'$d$' + f' = {a:.2f}'
                else:
                    change = r'$\upsilon$' + f' = {v1}, ' + r'$\omega$' + f' = {v2}' 
            elif scen_ == 1:
                if tab1_5:
                    change = r'$a$' + f' = {alpha:.2f}'
                else:
                    change = r'$a$' + f' = {alpha:.2f}, ' + r'$d$' + f' = {a:.2f}'
            elif scen_ == 2:
                if tab1_5:
                    change = r'$\upsilon$' + f' = {v1}'
                else:
                    change = r'$a$' + f' = {alpha:.2f}, ' + r'$\omega$' + f' = {v2}'
            elif scen_ == 3:
                if tab1_5:
                    change = r'$\omega$' + f' = {v2}'
                else:
                    change = r'$d$' + f' = {a:.2f}, ' + r'$\omega$' + f' = {v2}' 
            elif scen_ == 4:
                if tab1_5:
                    if R_max == 4:
                        change = 'pref mix'
                    else: 
                        change = r'$\upsilon$' + f' = {v1}, ' + r'$\omega$' + f' = {v2}'
                else:
                    change = r'$\upsilon$' + f' = {v1}, ' + r'$d$' + f' = {a:.2f}' 
            key = f'change to reference: {change}'
        key = f'{key}'
        scale = int(np.round((1 / STEP)))
        inset_axes = mpl_toolkits.axes_grid1.inset_locator.zoomed_inset_axes(ax, 0.7 / scale, loc='center')
        plot_figure(heatmap, reff_heatmap, heatmap_signature, key=key, ax=inset_axes, add_title=True,
                    pts=pts, labels=labels, cax1=cax1, cax2=cax2, cax3=cax3)
        letter_ax.text(1, 0.9, letter, weight='bold', size=FONT_SIZE,
                       fontdict=fontdict)
    from datetime import datetime
    fname_addon = datetime.now().strftime("-%Y_%m_%d_%H_%M")
    
    fig_number = '2'
    if not tab1_5:
        fig_number = '4'

    fig_variant = "_delta_"
    if R_max == 4:
        fig_variant = '_alpha_'

    fig.savefig('Fig'+ fig_number + fig_variant + fname_addon +'.png', format='png')
    fig.savefig('Fig'+ fig_number + fig_variant + fname_addon +'.pdf', format='pdf')


def plot_figure4(bottom_right_heatmap, top_left_heatmap, heatmap_signature, key='', ax=None, add_title=True, pts=None,
                 labels=None, cax2=None):
    if ax is None:
        return

    font = matplotlib.font_manager.FontProperties(family='sans-serif', size=FONT_SIZE)
    scale = int(np.round((1 / STEP) // 10))

    df0 = pd.DataFrame(heatmap_signature)
    df0 = df0.pivot(index='fv', columns='f', values='value').astype(float)
    df0.sort_index(level=0, ascending=False, inplace=True)
    np0 = df0.to_numpy()
    z0 = np.zeros_like(np0, dtype=bool)
    z0[:, :-1] = (np0[:, 1:] != np0[:, :-1])
    z0[:-1, :] += (np0[1:, :] != np0[:-1, :])
    df0[:] = z0

    cmap1 = cm.get_cmap('plasma_r')

    df = pd.DataFrame(bottom_right_heatmap)
    df = df.pivot(index='fv', columns='f', values='value').astype(float)
    df.sort_index(level=0, ascending=False, inplace=True)
    df0.columns = pd.Series([f'{float(c):.1f}' for c in list(df.columns)], name='f')
    df0.index = pd.Series([f'{float(c):.1f}' for c in list(df.index)], name='fv')

    sns.set(font_scale=1.0)
    #print(f'(I) max value: {df.max().max()}')
    # bottom-right picture
    from matplotlib.colors import LogNorm
    df_plus = df.copy() + 1e-8
    vmin = 10 + 1e-8
    vmax = 1000 + 1e-8
    sns.heatmap(df_plus, ax=ax, cmap=cmap1, square=True, xticklabels=scale,
                norm=LogNorm(vmin=vmin, vmax=vmax), vmin=vmin, vmax=vmax,
                yticklabels=scale, fmt='.1f', cbar=False, cbar_kws={'label': 'endemic state - I_V',
                                                                    'orientation': 'vertical'})
    if add_title:
        ax.set_title(key, fontdict=fontdict)
    decimals = [f'{i / 10:.1f}' for i in range(11)]
    ticklabels = []
    for decimal in decimals[:-1]:
        ticklabels.extend([decimal] + [''] * 9)
    ticklabels.extend([decimals[-1]])
    # border between phases, bottom-right side
    sns.heatmap(df0, ax=ax, mask=~z0, vmin=0, vmax=1, cmap='gray_r', square=True, xticklabels=scale, yticklabels=scale,
                cbar=False)

    ax.plot([10 * scale + 1, 0], [0, 10 * scale + 1], 'w--')
    cnfont = {'fontname': 'Courier New', 'size': 8}

    box_props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    arrow_props = dict(arrowstyle = "->", connectionstyle = "arc3, rad=0.1", fc = "white", alpha=0.95)
    
    if pts is not None:
        for i, pt in enumerate(pts):
            ax.plot((pt['f']) * scale * 10 + 0.5, (1 - (pt['fv'])) * scale * 10 + 0.5, 'k.', mew=.6)
            ax.plot((pt['f']) * scale * 10 + 0.5, (1 - (pt['fv'])) * scale * 10 + 0.5, '.', mew=0, color=pt['color'])
    if labels is not None:
    #    for i, label in enumerate(labels):
    #        ax.text((label['f']) * scale * 10 + 0.5, (1 - (label['fv'])) * scale * 10 + 0.5, label['signature'],
    #                color=label['color'],bbox = box_props, **cnfont)
        for i, label in enumerate(labels):
            arrow_style = None
            arrow_to = ((pts[i]['f']) * scale * 10 + 0.5, (1 - (pts[i]['fv'])) * scale * 10 + 0.5) 
            if label.get('arrow1'):
                arrow_style = arrow_props
                arrow_to = (label['f1arr'] * scale * 10 + 0.5, (1 - label['fv1arr']) * scale * 10 + 0.5)
            ax.annotate(label['signature'], xy = arrow_to , xycoords = 'data', 
                xytext = ((label['f']) * scale * 10 + 0.5, (1 - (label['fv'])) * scale * 10 + 0.5), textcoords = 'data', **cnfont,
                va = 'center', ha = 'center', bbox = box_props, arrowprops = arrow_style)


    df = pd.DataFrame(top_left_heatmap)
    df = df.pivot(index='fv', columns='f', values='value').astype(float)
    df.sort_index(level=0, ascending=False, inplace=True)
    df[:] = np.flipud(np.tril(np.rot90(df.to_numpy(), k=3)))

    df.columns = pd.Series([f'{float(c):.1f}' for c in list(df0.columns)], name='f/fv')
    df.index = pd.Series([f'{float(c):.1f}' for c in list(df0.index)], name='fv/f')
    # heatmap,  top-left side of the picture
    #print(f'(I_V) max value: {df.max().max()}')
    df_plus = df.copy() + 1e-8
    h = sns.heatmap(df_plus, cbar=False, ax=ax, mask=(df == 0), cmap=cmap1, square=True,
                    norm=LogNorm(vmin=vmin, vmax=vmax),
                    vmin=vmin, vmax=vmax,
                    xticklabels=int(scale), yticklabels=int(scale), fmt='.1f', cbar_kws={
            'label': 'endemic state - I',
            'orientation': 'horizontal'})
    cb = h.figure.colorbar(h.collections[0], cax=cax2,
                           label='Daily cases in the endemic state',
                           orientation='horizontal', extend='both', ticks=[vmin, 100 + 1e-8, vmax])
    cb.ax.set_xticklabels([r'$\leq$' + '10/mln', '100/mln', r'$\geq$' + '1000/mln'])
    cb.ax.tick_params(labelsize=FONT_SIZE, width=0.5, length=4, pad=2)  # Set the colorbar scale font size.
    text = cb.ax.xaxis.label
    text.set_font_properties(font)

    z0 = np.flipud(np.tril(np.rot90(z0, k=3)))
    df0[:] = z0
    df0.columns = pd.Series([f'{float(c):.1f}' for c in list(df.columns)], name='f')
    df0.index = pd.Series([f'{float(c):.1f}' for c in list(df.index)], name='fv')

    # black separation line between phases - top left side of the picture
    sns.heatmap(df0, ax=ax, mask=~z0, vmin=0, vmax=1, cmap='gray_r', square=True, xticklabels=False, yticklabels=False,
                cbar=False)

    def forward_right(x):
        return 1 - (x - 0.5) / scale / 10

    def forward_top(x):
        return (x - 0.5) / scale / 10

    secax = ax.secondary_xaxis('top')
    secax.set_xlabel(r'$f_v$', fontdict=fontdict, labelpad=14)
    secax.set_xticks([])
    secay = ax.secondary_yaxis('right')
    secay.set_ylabel(r'$f_v$', fontdict=fontdict, labelpad=12)
    secay.set_yticks([])
    ax.set_ylabel('f', fontdict=fontdict, labelpad=0)
    ax.set_xlabel('f', fontdict=fontdict, labelpad=0)
    series = list(np.arange(0.5, scale * 10 + 0.51, 2 * scale))
    xseries = series
    yseries = series
    ax.set_xticks(xseries)
    ax.set_yticks(yseries)
    yticks = list([f'{forward_right(x):.1f}' for x in ax.get_yticks()])
    xticks = list([f'{forward_top(x):.1f}' for x in ax.get_xticks()])
    ax.set_yticklabels(yticks, size=FONT_SIZE)
    ax.set_xticklabels(xticks, size=FONT_SIZE)
    for ax_ in [ax]:
        ax_.tick_params(left=True, labelleft=True, top=True, labeltop=True,
                        right=True, labelright=True, bottom=True, labelbottom=True, pad=0)
    for ax_ in [secax, secay]:
        ax_.tick_params(left=False, labelleft=True, top=False, labeltop=True,
                        right=False, labelright=True, bottom=False, labelbottom=True, pad=0)
    plt.sca(ax)
    plt.xticks(rotation=0)
    plt.yticks(rotation=90)
    ax.xaxis.set_tick_params(width=0.5, length=2, pad=2)
    ax.yaxis.set_tick_params(width=0.5, length=2, pad=2)
    if pts is not None:
        for i, pt in enumerate(pts):
            ax.plot((pt['fv']) * scale * 10 + 0.5, (1 - (pt['f'])) * scale * 10 + 0.5, 'k.', mew=.6)
            ax.plot((pt['fv']) * scale * 10 + 0.5, (1 - (pt['f'])) * scale * 10 + 0.5, '.', mew=0, color=pt['color'])

    if labels is not None:
        for i, label in enumerate(labels):
            fv = label['fv']
            f = label['f']
            if label.get('special', False):
                fv = label['fv2']
                f = label['f2']
            arrow_style = None
            arrow_to = ((pts[i]['fv']) * scale * 10 + 0.5, (1 - (pts[i]['f'])) * scale * 10 + 0.5) 
            if label.get('arrow2'):
                arrow_style = arrow_props
                arrow_to = (label['fv2arr'] * scale * 10 + 0.5, (1 - label['f2arr']) * scale * 10 + 0.5)
            ax.annotate(label['signature'], xy = arrow_to , xycoords = 'data', 
                xytext = (fv * scale * 10 + 0.5, (1 - f) * scale * 10 + 0.5), textcoords = 'data', **cnfont,
                va = 'center', ha = 'center', bbox = box_props, arrowprops = arrow_style)
 
#        for i, label in enumerate(labels):
#            fv = label['fv']
#            f = label['f']
#            if label.get('special', False):
#                fv = label['fv2']
#                f = label['f2']
#            ax.text(fv * scale * 10 + 0.5, (1 - f) * scale * 10 + 0.5, label['signature'], color=label['color'],
#                    bbox = box_props, **cnfont)




def heatmap4_calc(d, upsilon1, upsilon2, r_max, alpha, mixing_type, kappa=0.002, gamma=1/6):
    
    def heatmap4_calc_parallel(f):
        res = [None] * len(f_list)
        for idx, fv in enumerate(f_list):
            i_val, iv_val = solve_i_iv(f=f, fv=fv, d=d, kappa=kappa, upsilon1=upsilon1, upsilon2=upsilon2, r_max=r_max,
                                   mixing_type=mixing_type, alpha=alpha, gamma=gamma)
            res[idx] = ( set_elem_heatmap(f, fv, i_val / 6 * 1000000), set_elem_heatmap(f, fv, iv_val / 6 * 1000000))
        return res
    
    result = Parallel(n_jobs=MY_N_JOBS)(delayed(heatmap4_calc_parallel)(f) for f in f_list)
    
    flattened_result = flattened_list(result)

    return [x for x, y in flattened_result], [y for x, y in flattened_result]


def heatmap4_calc_old(d, upsilon1, upsilon2, r_max, alpha, mixing_type, kappa=0.002, gamma=1/6):
    i = []
    iv = []
    for f, fv in product(f_list, repeat=2):
        i_val, iv_val = solve_i_iv(f=f, fv=fv, d=d, kappa=kappa, upsilon1=upsilon1, upsilon2=upsilon2, r_max=r_max,
                                   mixing_type=mixing_type, alpha=alpha, gamma=gamma)
        i.append(set_elem_heatmap(f, fv, i_val / 6 * 1000000))
        iv.append(set_elem_heatmap(f, fv, iv_val / 6 * 1000000))

    return i, iv


def solve_i_iv(f, fv, d, upsilon1, upsilon2, r_max, alpha, mixing_type, kappa=0.002, gamma=1/6):
    x = Symbol('x')
    y = Symbol('y')
    beta = gamma * r_max * (1 - f)
    delta = gamma * r_max * (1 - fv)
    delta_star = delta
    delta_plus = beta
    if mixing_type == 'pref':
        if beta == 0 and delta == 0:
            delta_star = 0
            delta_plus = 0
        else:
            delta_star = delta * delta / (beta * d + delta * (1 - d))
            delta_plus = beta * beta / (beta * d + delta * (1 - d))

    def f1n(x, y):  # checked
        return upsilon1 * alpha * gamma * y * (upsilon2 + beta * x + delta_star * y)

    def f1d1(x, y):  # checked
        return beta * x + delta_star * y + upsilon2 + upsilon1 * (1 - alpha)

    def f2e1(y):  # checked
        return upsilon2 * (1 - d - y * (1 + gamma / (kappa + upsilon1)))

    def f2e2(y):  # checked
        return upsilon2 * gamma * y

    def f3(y):  # checked
        return upsilon1 * gamma * y / (kappa + upsilon1)

    try:
        sols = nsolve([
            (beta * x + delta_plus * y) * (d - x * (1 + gamma/kappa)) - gamma * x,
            (beta * x + delta_star * y) * (-f2e1(y) + f3(y)) + (f1n(x, y) / f1d1(x, y) + f2e2(y))
               ], [x, y], [0.1, 0.1], verify=False)
    except ZeroDivisionError as e:
        print((f, fv, d, upsilon1, upsilon2, r_max, alpha, mixing_type, kappa, gamma))
        raise e
    assert len(sols) == 2, f"expected unique (x,y), got {len(sols)}: {sols}"
    i, iv = sols
    i = float(i)
    iv = float(iv)
    return i, iv


def fig3(scenarios, pts0=None, labels0=None, tab1_5 = True):
    fig = plt.figure(figsize=(18.3 * cms, 14.6 * cms), dpi=200)
    h = [0.2, 5.7, 0.15] * 3 + [0.15]
    v = ([0.2] + [0.6, 5.7, 0.2] * 2 + [0.1] + [0.4, 0.9])[::-1]
    horiz = [Size.Fixed(h_ * cms) for h_ in h]
    vert = [Size.Fixed(v_ * cms) for v_ in v]
    rect = (0.0, 0.0, 1.0, 1.0)
    # Divide the axes rectangle into a grid with sizes specified by horiz * vert.
    divider = Divider(fig, rect, horiz, vert, aspect=False)
    # The rect parameter will actually be ignored and overridden by axes_locator.
    ax1 = fig.add_axes(rect, axes_locator=divider.new_locator(nx=1, ny=len(v) - 1 - 2), frameon=False)
    ax2 = fig.add_axes(rect, axes_locator=divider.new_locator(nx=4, ny=len(v) - 1 - 2), frameon=False)
    ax3 = fig.add_axes(rect, axes_locator=divider.new_locator(nx=7, ny=len(v) - 1 - 2), frameon=False)
    ax4 = fig.add_axes(rect, axes_locator=divider.new_locator(nx=1, ny=len(v) - 1 - 5), frameon=False)
    ax5 = fig.add_axes(rect, axes_locator=divider.new_locator(nx=4, ny=len(v) - 1 - 5), frameon=False)
    ax6 = fig.add_axes(rect, axes_locator=divider.new_locator(nx=7, ny=len(v) - 1 - 5), frameon=False)
    cax2_o = fig.add_axes(rect, axes_locator=divider.new_locator(nx=4, ny=len(v) - 1 - 8), frameon=False)
    axa = fig.add_axes(rect, axes_locator=divider.new_locator(nx=0, ny=len(v) - 1 - 1), frameon=False)
    axb = fig.add_axes(rect, axes_locator=divider.new_locator(nx=3, ny=len(v) - 1 - 1), frameon=False)
    axc = fig.add_axes(rect, axes_locator=divider.new_locator(nx=6, ny=len(v) - 1 - 1), frameon=False)
    axd = fig.add_axes(rect, axes_locator=divider.new_locator(nx=0, ny=len(v) - 1 - 4), frameon=False)
    axe = fig.add_axes(rect, axes_locator=divider.new_locator(nx=3, ny=len(v) - 1 - 4), frameon=False)
    axf = fig.add_axes(rect, axes_locator=divider.new_locator(nx=6, ny=len(v) - 1 - 4), frameon=False)
    cax2 = cax2_o  # mpl_toolkits.axes_grid1.inset_locator.zoomed_inset_axes(cax2_o, 0.8 / 0.01, loc='center')
    caxs = [cax2_o] # cax1_o, cax2_o, cax3_o]
    axs = [ax1, ax2, ax3, ax4, ax5, ax6]
    axletters = [axa, axb, axc, axd, axe, axf]
    for iax in caxs + axs + axletters:
        iax.set_yticks([]), iax.set_xticks([])

    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial'], 'size': FONT_SIZE})
    matplotlib.rcParams.update({'font.size': FONT_SIZE})
    for scenario, ax, letter_ax in zip(scenarios, axs, axletters):
        (a, alpha, v1, v2, type_, R_max, scen_, letter) = scenario
        data = generate_data(a, alpha, v1, v2)
        #print(scenario)
        #heatmap_signature, _ = signature_calc(data, type_=type_, R_max=R_max)
        heatmap_signature, _ = signature_calc(data, type_=type_, R_max=R_max, d=a, gamma=GAMMA)
        bottom_right, top_left = heatmap4_calc(d=a, upsilon1=v1, upsilon2=v2, r_max=R_max,
                                               alpha=alpha, gamma=1/6, kappa=0.002, mixing_type=type_)
        pts = None
        labels = None
        if scen_ == '*':
            pts = pts0
            labels = labels0
            if tab1_5:
                key = r'$a$' + f' = {alpha:.2f}, ' + r'$\upsilon$' + f' = {v1}, '
                key += r'$\omega$' + f' = {v2}, ' + r'$d$' + f' = {a:.2f}'
            else:
                key = f'change to reference: ' + r'$a$' + f' = {alpha:.2f}, ' + r'$\upsilon$' + f' = {v1}'
        else:
            change = ''
            if scen_ == 0:
                if tab1_5:
                    change = r'$d$' + f' = {a:.2f}'
                else:
                    change = r'$\upsilon$' + f' = {v1}, ' + r'$\omega$' + f' = {v2}' 
            elif scen_ == 1:
                if tab1_5:
                    change = r'$a$' + f' = {alpha:.2f}'
                else:
                    change = r'$a$' + f' = {alpha:.2f}, ' + r'$d$' + f' = {a:.2f}'
            elif scen_ == 2:
                if tab1_5:
                    change = r'$\upsilon$' + f' = {v1}'
                else:
                    change = r'$a$' + f' = {alpha:.2f}, ' + r'$\omega$' + f' = {v2}'
            elif scen_ == 3:
                if tab1_5:
                    change = r'$\omega$' + f' = {v2}'
                else:
                    change = r'$d$' + f' = {a:.2f}, ' + r'$\omega$' + f' = {v2}' 
            elif scen_ == 4:
                if tab1_5:
                    if R_max == 4:
                        change = 'pref mix'
                    else:
                        change = r'$\upsilon$' + f' = {v1}, ' + r'$\omega$' + f' = {v2}'
                else:
                    change = r'$\upsilon$' + f' = {v1}, ' + r'$d$' + f' = {a:.2f}' 

            key = f'change to reference: {change}'
        key = f'{key}'
        scale = int(np.round((1 / STEP)))
        inset_axes = mpl_toolkits.axes_grid1.inset_locator.zoomed_inset_axes(ax, 0.7 / scale, loc='center')
        plot_figure4(bottom_right, top_left,
                     heatmap_signature, key=key, ax=inset_axes, add_title=True,
                     pts=pts, labels=labels, cax2=cax2)
        letter_ax.text(1, 0.9, letter, weight='bold', size=FONT_SIZE,
                       fontdict=fontdict)

    from datetime import datetime
    fname_addon = datetime.now().strftime("-%Y_%m_%d_%H_%M")
    fig_number = '3'
    fig_variant = '_delta_'
    if not tab1_5:
        fig_number = '5'
    if R_max == 4:
        fig_variant = '_alpha_'
    fig.savefig('Fig' + fig_number + fig_variant + fname_addon + '.png', format='png')
    fig.savefig('Fig' + fig_number + fig_variant + fname_addon + '.pdf', format='pdf')


if __name__ == "__main__":
    
    # execute only if run as a script

    # These are HTML symbols of "plus" and "minutes" to be used for signature labels
    import html
    PLUS = html.unescape('&#43;')
    MINUS = html.unescape('&#8722;')

    HSIG={'+': PLUS, '-': MINUS, '+-': PLUS+MINUS, '-+': MINUS+PLUS, '+-+': PLUS+MINUS+PLUS}

    # Some flags I used to create specific heatmaps
    alpha_ = False           # True if I plot the alpha variant
    use_labels = True       # True if I want to add labels on the subpanel a)
    use_points = True       # True if I want to add reference points on the subpanel a)
    tab1_5_ = True           # True if I produce heatmaps corresponding to rows 1-5 of the article 
                            # These are scenarios with one change to the reference setting point
    parallel = True         # True if I want to run the code in parallel
    all_figures = True      # True if I want to run all for figures (for both variants for both changes sets)

    ## These are constants used in the article
    d_low = 0.12    # never-vaccinated low, GB-like
    d_high = 0.3    # never-vaccinated high, France-like
    w_low = 1/500   # omega, natural immunity lasts 500 days
    w_high = 1/200  # omega, natural immunity lasts 200 days
    v_low = 0.004   # upsilon, low vaccination rate 0.004 European-like
    v_high = 0.008  # upsilon, high vaccination rate

    ## variant specific effficacies of vaccines
    if alpha_:
        a_low = 0.73 
        a_high = 0.92
    else:
        a_low = 0.6
        a_high = 0.79



    ## If we don't run the code in parallel set names of the principal functions to the old, one-core version
    if not parallel: 
        signature_prop = signature_prop_old
        heatmap_prop = heatmap_prop_old
        reff_prop = reff_prop_old
        heatmap4_calc = heatmap4_calc_old

    
    # clumsy way to specify all expected figures to be generated
    which_heatmaps = [tab1_5_]
    if all_figures: 
        which_heatmaps = [True, False]
    which_variants = [alpha_]
    if all_figures:
        which_variants = [True, False]


    # iterate over all expected figures and generate them
    for tab1_5, alpha in product(which_heatmaps, which_variants): 
        if tab1_5:
            if alpha:
                scenarios_ = [(0.12, 0.92, 0.004, 1 / 500, 'prop', 4, '*', 'a'),
                    (0.12, 0.73, 0.004, 1 / 500, 'prop', 4, 1, 'b'),
                    (0.12, 0.92, 0.008, 1 / 500, 'prop', 4, 2, 'c'),
                    (0.12, 0.92, 0.004, 1 / 500, 'pref', 4, 4, 'd'),
                    (0.3, 0.92, 0.004, 1 / 500, 'prop', 4, 0, 'e'),
                    (0.12, 0.92, 0.004, 1 / 200, 'prop', 4, 3, 'f')
                    ]
            else:
                scenarios_ = [(0.12, 0.79, 0.004, 1 / 500, 'prop', 6, '*', 'a'),
                    (0.12, 0.6, 0.004, 1 / 500, 'prop', 6, 1, 'b'),
                    (0.12, 0.79, 0.008, 1 / 500, 'prop', 6, 2, 'c'),
                    (0.12, 0.79, 0.004, 1 / 500, 'pref', 6, 4, 'd'),
                    (0.3, 0.79, 0.004, 1 / 500, 'prop', 6, 0, 'e'),
                    (0.12, 0.79, 0.004, 1 / 200, 'prop', 6, 3, 'f')
                    ]
        else:
            if alpha:
                scenarios_ = [(0.12, 0.73, 0.008, 1 / 500, 'prop', 4, '*', 'a'),
                    (0.3, 0.73, 0.004, 1 / 500, 'prop', 4, 1, 'b'),
                    (0.12, 0.73, 0.004, 1 / 200, 'prop', 4, 2, 'c'),
                    (0.3, 0.92, 0.008, 1 / 500, 'prop', 4, 4, 'd'),
                    (0.12, 0.92, 0.008, 1 / 200, 'prop', 4, 0, 'e'),
                    (0.3, 0.92, 0.004, 1 / 200, 'prop', 4, 3, 'f')
                    ]
            else:
                scenarios_ = [(0.12, 0.6, 0.008, 1 / 500, 'prop', 6, '*', 'a'),
                    (0.3, 0.6, 0.004, 1 / 500, 'prop', 6, 1, 'b'),
                    (0.12, 0.6, 0.004, 1 / 200, 'prop', 6, 2, 'c'),
                    (0.3, 0.79, 0.008, 1 / 500, 'prop', 6, 4, 'd'),
                    (0.12, 0.79, 0.008, 1 / 200, 'prop', 6, 0, 'e'),
                    (0.3, 0.79, 0.004, 1 / 200, 'prop', 6, 3, 'f')
                    ]


        if use_labels and tab1_5: # we annotate only on the Fig 2 and 3 (which are rows 1-5 from the Table)
            if alpha:
                labels_ = [
                    {'f': 0.83, 'fv': 0.13, 'signature': HSIG['-+'], 'color': 'black', 'size': 6},
                    {'f': 0.6, 'fv': 0.26, 'signature': HSIG['+-+'], 'color': 'black', 'size': 6, 'special': True, 'f2': 0.62,
                    'fv2': 0.1},
                    {'f': 0.34, 'fv': 0.11, 'signature': HSIG['+'], 'color': 'black', 'size': 6},
                    {'f': 0.86, 'fv': 0.55, 'signature': HSIG['-'], 'color': 'black', 'size': 6},
                    {'f': 0.58, 'fv': 0.45, 'signature': HSIG['+-'], 'color': 'black', 'size': 6}
                ]   
            else:
                labels_ = [
                    {'f': 0.93, 'fv': 0.23, 'signature': HSIG['-+'], 'color': 'black', 'size': 6},
                    {'f': 0.78, 'fv': 0.06, 'signature': HSIG['+-+'], 'color': 'black', 'size': 6, 'special': True, 'f2': 0.77,
                    'fv2': 0.105, 'arrow1': True, 'f1arr' : 0.81, 'fv1arr' : 0.32, 'arrow2': True, 'f2arr' : 0.81, 'fv2arr': 0.35},
                    {'f': 0.3, 'fv': 0.06, 'signature': HSIG['+'], 'color': 'black', 'size': 6},
                    {'f': 0.93, 'fv': 0.81, 'signature': HSIG['-'], 'color': 'black', 'size': 6},
                    {'f': 0.93, 'fv': 0.53, 'signature': HSIG['+-'], 'color': 'black', 'size': 6, 
                    'arrow1': True, 'f1arr': 0.78, 'fv1arr': 0.69, 'arrow2': True, 'f2arr': 0.78, 'fv2arr': 0.69}
                ]   
        else:
            labels_ = None

        if use_points and tab1_5: # we annotate only on the Fig 2 and 3 (which are rows 1-5 from the Table)
            if alpha:
                pts_ = [
                    {'f': 0.92, 'fv': 0.04, 'signature': '-+', 'color': 'violet'},
                    {'f': 0.63, 'fv': 0.05, 'signature': '+-+', 'color': 'orange'},
                    {'f': 0.45, 'fv': 0.32, 'signature': '+', 'color': 'red'},
                    {'f': 0.82, 'fv': 0.4, 'signature': '-', 'color': 'blue'},
                    {'f': 0.7, 'fv': 0.4, 'signature': '+-', 'color': 'deepskyblue'}
                ]
            else:
                pts_ = [
                    {'f': 0.92, 'fv': 0.38, 'signature': '-+', 'color': 'violet'},
                    {'f': 0.77, 'fv': 0.55, 'signature': '+-+', 'color': 'orange'},
                    {'f': 0.77, 'fv': 0.38, 'signature': '+', 'color': 'red'},
                    {'f': 0.92, 'fv': 0.71, 'signature': '-', 'color': 'blue'},
                    {'f': 0.77, 'fv': 0.71, 'signature': '+-', 'color': 'deepskyblue'}
                ]
        else:
            pts_ = None

        
        start_time = time.time()
        #fig1b(scenarios=scenarios_, pts=pts_)
        fig2(scenarios=scenarios_, pts0=pts_, labels0=labels_, tab1_5 = tab1_5)
        diff_time = time.time() - start_time 
        print("---- Fig. generated after %s minutes and %s seconds ----" % (diff_time // 60, round(diff_time % 60)) )
        
        start_time = time.time()
        fig3(scenarios=scenarios_, pts0=pts_, labels0=labels_, tab1_5 = tab1_5)
        diff_time = time.time() - start_time 
        print("---- Fig. generated after %s minutes and %s seconds ----" % (diff_time // 60, round(diff_time % 60)) )
        
