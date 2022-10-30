### File containing RECH model functions

import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as opt
import yfinance as yf

def sigmoid(x):
    try:
        sig = 1 / (1 + math.exp(-x))
    except:
        sig = 0
    return sig

def relu(x, bound = 20):
    return min(max(0,x),bound)

def garch(pars, nun_lin_func, returns):
    (omega, alpha , beta) = pars
    iT = len(returns)
    sigma_2 = np.zeros(iT)
    for i in range(iT):
        if i == 0:
            sigma_2[i] = omega/(1- alpha - beta)
            # w[i] = 0.1/(1- alpha - beta)
        else:
            sigma_2[i] = nun_lin_func(omega) + alpha * returns[i-1]**2 + beta * sigma_2[i-1]
    return sigma_2, 1, 1

def garch_loglike(start_v, nun_lin_func, returns):
    sigma_2 = garch(start_v, nun_lin_func, returns)[0]
    neg_LogL = - np.sum(-np.log(sigma_2) -  returns**2/sigma_2)
    return neg_LogL


def SRN_garch(pars, nun_lin_func, returns):
    (alpha, beta, gamma0, gamma1, v_1, v_2, v_3, b) = pars
    iT = len(returns)
    sigma_2 = np.zeros(iT)
    w = np.zeros(iT)
    h = np.zeros(iT)
    for i in range(iT):
        if i == 0:
            sigma_2[i] = 0.1/(1- alpha - beta)
        else:
            h[i] = nun_lin_func(v_1 * np.sign(returns[i-1]) * returns[i-1]**2 + v_2 * sigma_2[i-1] + v_3 * h[i-1] + b)
            w[i] = gamma0 + gamma1 * h[i]
            sigma_2[i] = w[i] + alpha * returns[i-1]**2 + beta * sigma_2[i-1]
    return sigma_2, w, h

def SRN_garch_loglike(start_v, nun_lin_func, returns):
    sigma_2 = SRN_garch(start_v, nun_lin_func, returns)[0]
    neg_LogL = - np.sum(-np.log(sigma_2) -  returns**2/sigma_2)
    return neg_LogL

def SRN_garch_vec(pars, nun_lin_func, returns):
    (alpha, beta, gamma0, gamma1_1, gamma1_2, v_11, v_12, v_21, v_22, b_1, b_2) = pars 
    iT = len(returns)
    sigma_2 = np.zeros(iT)
    w = np.zeros(iT)
    p = 2
    h = np.zeros(iT*p)
    h = h.reshape(iT,p)
    for i in range(iT):
        if i == 0:
            sigma_2[i] = 0.1/(1- alpha - beta)
        else:
            h[i,0] = nun_lin_func(v_11 * np.sign(returns[i-1]) * returns[i-1]**2 + v_12 * h[i-1,0] + b_1)
            h[i,1] = nun_lin_func(v_21 * sigma_2[i-1] + v_22 * h[i-1,1]  + b_2)
            w[i] = gamma0 + gamma1_1 * h[i,0] + gamma1_2 * h[i,1]
            sigma_2[i] = w[i] + alpha * returns[i-1]**2 + beta * sigma_2[i-1]
    return sigma_2, w, h

def SRN_garch_loglike_vec(start_v, nun_lin_func, returns):
    sigma_2 = SRN_garch_hvec(start_v, nun_lin_func, returns)[0]
    neg_LogL = - np.sum(-np.log(sigma_2) -  returns**2/sigma_2)
    return neg_LogL




def MGU_garch(pars, nun_lin_func, returns):
    (alpha, beta, gamma_0, gamma_1, v_11, v_12, v_21, v_22, mu_1, mu_2, b_h, b_z) = pars
    iT = len(returns)
    sigma_2 = np.zeros(iT)
    w = np.zeros(iT)
    h = np.zeros(iT)
    h_hat = np.zeros(iT)
    z = np.zeros(iT)
    for i in range(iT):
        if i == 0:
            sigma_2[i] = 0.1/(1- alpha - beta)
        else:
            z[i] = sigmoid(v_21 * returns[i-1] + v_22 * sigma_2[i-1] + mu_2 * z[i-1] * h[i-1] + b_z) # here sigmoid instead of ReLU
            h_hat[i] = nun_lin_func(v_11 * returns[i-1] + v_12 * sigma_2[i-1] + mu_1 * z[i] * h[i-1] + b_h)
            h[i] = nun_lin_func(z[i] * h_hat[i] + (1-z[i]) * h[i-i])
            w[i] = gamma_0 + gamma_1 * h[i]
            sigma_2[i] = w[i] + alpha * returns[i-1]**2 + beta * sigma_2[i-1]
    return sigma_2, w, h

def MGU_garch_loglike(start_v, nun_lin_func, returns):  
    sigma_2 = MGU_garch(start_v, nun_lin_func, returns)[0]
    LogL = - np.sum(-np.log(sigma_2) -  returns**2/sigma_2)
    return LogL

def GRU_garch(pars, nun_lin_func, returns):
    (alpha, beta, gamma_0, gamma_1, v_11, v_12, v_21, v_22, v_31, v_32, mu_1, mu_2, mu_3, b_h, b_r, b_z) = pars
    iT = len(returns)
    sigma_2 = np.zeros(iT)
    w = np.zeros(iT)
    h = np.zeros(iT)
    h_hat = np.zeros(iT)
    r = np.zeros(iT)
    z = np.zeros(iT)
    for i in range(iT):
        if i == 0:
            sigma_2[i] = 0.1/(1- alpha - beta)
        else:
            z[i] = sigmoid(v_31 * returns[i-1] + v_32 * sigma_2[i-1] + mu_3 * h[i-1] + b_z) # here sigmoid instead of ReLU
            r[i] = nun_lin_func(v_21 * returns[i-1] + v_22 * sigma_2[i-1] + mu_2 * h[i-1] + b_r)
            h_hat[i] = nun_lin_func(v_11 * returns[i-1] + v_12 * sigma_2[i-1] + mu_1 * r[i] * h[i-1] + b_h)
            h[i] = nun_lin_func(z[i] * h_hat[i] + (1-z[i]) * h[i-i])
            w[i] = gamma_0 + gamma_1 * h[i] 
            sigma_2[i] = w[i] + alpha * returns[i-1]**2 + beta * sigma_2[i-1]
    return sigma_2, w, h

def GRU_garch_loglike(start_v, nun_lin_func, returns):
    sigma_2 = GRU_garch(start_v, nun_lin_func, returns)[0]
    LogL = - np.sum(-np.log(sigma_2) -  returns**2/sigma_2)
    return LogL


def LSTM_garch(pars, nun_lin_func, returns):
    (alpha, beta, gamma_0, gamma_1, v_11, v_12, v_21, v_22, v_31, v_32, v_41, v_42, mu_1, mu_2, mu_3, mu_4, b_f, b_i, b_o, b_c) = pars
    iT = len(returns)
    sigma_2 = np.zeros(iT)
    w = np.zeros(iT)
    h = np.zeros(iT)
    f = np.zeros(iT)
    ij = np.zeros(iT)
    o = np.zeros(iT)
    c = np.zeros(iT)
    c_tilde = np.zeros(iT)
    for i in range(iT):
        if i == 0:
            sigma_2[i] = 0.1/(1- alpha - beta)
        else:
            f[i] = sigmoid(v_41 * np.sign(returns[i-1]) *  returns[i-1]**2 + v_42 * sigma_2[i-1] + mu_4 * h[i-1] + b_f) # here sigmoid instead of ReLU
            ij[i] = sigmoid(v_31 * np.sign(returns[i-1]) *  returns[i-1]**2 + v_32 * sigma_2[i-1] + mu_3 * h[i-1] + b_i)
            o[i] = sigmoid(v_21 * np.sign(returns[i-1]) *  returns[i-1]**2 + v_22 * sigma_2[i-1] + mu_2 * h[i-1] + b_o)
            c_tilde[i] = sigmoid(v_21 * np.sign(returns[i-1]) *  returns[i-1]**2 + v_22 * sigma_2[i-1] + mu_2 * h[i-1] + b_o)
            c[i] = f[i] * c[i-1] + ij[i] * c_tilde[i]
            h[i] = o[i] * sigmoid(c[i])
            w[i] = gamma_0 + gamma_1 * h[i] 
            sigma_2[i] = w[i] + alpha * returns[i-1]**2 + beta * sigma_2[i-1]
    return sigma_2, w, h

def LSTM_garch_loglike(start_v, nun_lin_func, returns):
    sigma_2 = LSTM_garch(start_v, nun_lin_func, returns)[0]
    LogL = - np.sum(-np.log(sigma_2) -  returns**2/sigma_2)
    return LogL