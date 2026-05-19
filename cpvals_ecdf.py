import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## DKW in transductive paper
def DKW_tau(n,m):
    return m*n/(m+n)
def DKW_Psi(x, m, n, delta):
    return np.minimum(1, np.sqrt((np.log(1/delta) + np.log(1+np.sqrt(2*np.pi)*2*x*DKW_tau(n,m)/np.sqrt(n+m)))\
                                 /(2*DKW_tau(n,m))))
def DKW_lambda(r,x,m,n,delta):
    re = x
    for kk in range(r):
        re = DKW_Psi(re,m,n,delta)
    return re

## Asymptotic method
def compute_cn(delta,n):
    cn = -np.log(-np.log(1-delta)) + 2*np.log(np.log(n)) + 0.5 * np.log(np.log(np.log(n))) - 0.5 * np.log(np.pi)
    cn /= np.sqrt(2*np.log(np.log(n)))
    return cn

def betainv_generic(pvals, aseq):
    n = len(aseq)
    idx = np.maximum(1, np.floor((n + 1) * (1 - pvals))).astype(int)
    out = 1 - aseq[idx-1]
    return out

def betainv_asymptotic(pvals, n, k, delta):
    k = int(k)
    iseq = np.arange(1,n+1)
    cn = compute_cn(delta, n)
    aseq = iseq / n + cn * np.sqrt(iseq*(n-iseq)) / (n*np.sqrt(n))
    aseq = 1 - np.minimum(1, aseq[::-1])
    out = betainv_generic(pvals, aseq)
    return out

def ecdf_asymptotic(tvec,n,k,delta):
# sum ind b_i < t
    iseq = np.arange(1,n+1)
    cn = compute_cn(delta, n)
    aseq = iseq / n + cn * np.sqrt(iseq*(n-iseq)) / (n*np.sqrt(n))
    aseq = np.minimum(1, aseq)
    return np.array([np.sum(aseq< t) for t in tvec])/len(aseq)


## Simes method
def compute_aseq(n, k, delta):
    def movingaverage (values, window):
        weights = np.repeat(1.0, window)/window
        sma = np.convolve(values, weights, 'valid')
        return sma

    k = int(k)
    fac1 = np.log(delta) / k - np.mean(np.log(np.arange(n-k+1,n+1)))
    fac2 = movingaverage(np.log(np.arange(1,n+1)), k)
    aseq = np.concatenate([np.zeros((k-1,)), np.exp(fac2 + fac1)])
    return aseq

def betainv_simes(pvals, n, k, delta):
    aseq = compute_aseq(n, k, delta)
    out = betainv_generic(pvals, aseq)
    return out

def ecdf_simes(tvec,n,k,delta):
# sum ind b_i < t
    aseq = 1-compute_aseq(n, k, delta)
    return np.array([np.sum(aseq< t) for t in tvec])/len(aseq)

# Monte Carlo
def compute_hybrid_bound(delta,n,gamma):
    i = np.arange(1,n+1)
    cna = compute_cn(delta-gamma,n)
    bound = i/n + cna * np.sqrt(i*(n-i))/(n*np.sqrt(n))
    k_linear = int(n/2)
    slope = (bound[k_linear-1]-bound[k_linear-2])
    bound[k_linear:] = bound[k_linear-1] + slope * (i[k_linear:]-k_linear)
    k_simes = int(n/2)
    bound_s = 1.0-compute_aseq(n, k_simes, delta)[::-1]
    bound_h = np.minimum(bound_s, bound)
    return bound_h

def betainv_mc(pvals, n, delta, fs_correction=1):
    iseq = np.arange(1,n+1)
    cn = compute_cn(delta, n)
    bound = compute_hybrid_bound(delta,n,fs_correction)
    aseq = 1 - np.minimum(1, bound[::-1])
    out = betainv_generic(pvals, aseq)
    return out

## Simulate a conformal pvals
def sample_cpvals(n,m):
    iid_uniform = np.random.rand(n)
    sorted_uniform = np.sort(iid_uniform)
    p = np.diff(np.concatenate(([0], sorted_uniform, [1])))
    generated_variables = (np.random.choice(np.arange(len(p)), size=m, p=p)+1)/len(p)
    #return generated_variables
    return np.random.rand(m)

def compute_M(pvals):
    m = len(pvals)
    sorted_pvals = np.sort(pvals)
    grid = np.arange(1, m)/m
    r1 = (grid - sorted_pvals[0:m-1]) / np.sqrt(sorted_pvals[0:m-1]) / (np.sqrt(1-sorted_pvals[0:m-1])+1e-8)
    r2 = (grid - sorted_pvals[1:m]) / np.sqrt(sorted_pvals[1:m]) / (np.sqrt(1-sorted_pvals[1:m])+1e-8)
    alpha =0
    beta = 1
    ll = np.ceil(alpha*m).astype(int)
    ll=0
    uu = np.ceil(beta*m).astype(int)-1
    return np.max(np.maximum(r1[ll:uu], r2[ll:uu]))



np.random.seed(123)

m = 2000
n = 1000
k = 2

delta = 0.05

pgrid = np.arange(0, 1.0, 0.001)+1e-8
upper_asymptotic_1 = betainv_asymptotic(pgrid, n, k, delta)
upper_asymptotic = 1 - ecdf_asymptotic(1-upper_asymptotic_1, m, m/2, delta)

upper_DKW = pgrid + DKW_lambda(1000,1,m,n,2*delta)
upper_simes_1 = betainv_simes(pgrid, n, n/2, delta)
upper_mc = betainv_mc(pgrid, n, delta)
upper_simes_asy = 1 - ecdf_asymptotic(1-upper_simes_1, m, m, delta)

T = 1000
all_samples = np.zeros((T,m))
M_vec = np.zeros(T)
for t in range(T):
    pvals_tem = sample_cpvals(n,m)
    pvals_tem = np.random.rand(m) ### Experiments on uniform ecdf!!!!!
    all_samples[t] = pvals_tem
    M_vec[t] = compute_M(pvals_tem)

upper_marginal = np.zeros(len(pgrid))
for k in range(len(pgrid)):
    upper_marginal[k] = np.quantile(np.average(all_samples<pgrid[k], axis = 1), 1-delta)

##
M_quantile = np.quantile(M_vec,1-delta)

upper_simulation = np.zeros(len(pgrid))
for k in range(len(pgrid)):
    c = M_quantile
    t = pgrid[k]
    upper_simulation[k] =t + c*np.sqrt(t*(1-t))
##

fig, ax = plt.subplots(figsize=(6, 6))
# ax.step(pgrid, upper_simes,  where='pre', label='simes')
# ax.step(pgrid, upper_asymptotic,  where='pre', label='asymptotic')

#plt.plot(pgrid, upper_simes_asy, label='simes_asy')
plt.plot(pgrid, upper_simes_1, label='simes')
plt.plot(pgrid, upper_asymptotic_1, label='asymptotic')
plt.plot(pgrid, upper_marginal, label='mariginal')
#plt.plot(pgrid, upper_mc, label='montecarlo')
plt.plot(pgrid, upper_simulation, label='simulation')

# upper_mc_1 = betainv_mc(pgrid, n, delta)
# upper_mc = betainv_mc(upper_mc_1, n, delta)
#plt.plot(pgrid, upper_mc, label='mc')

#plot all ecdf
for _ in range(1000):
    x = np.sort(sample_cpvals(n,m))
    y = np.arange(0, len(x)) / len(x)
    x = np.concatenate(([0], x))
    y = np.concatenate(([0], y))
    ax.step(x, y, where='pre', color='black', alpha=0.1)
plt.xlabel('t')
plt.ylabel('Values')
plt.legend()

plt.xlim(-0.001, 0.07)
plt.ylim(-0.001, 0.07)


plt.show()


## Determine coverage rate

# T = 1000
# r_simes = np.zeros(T)
# r_asy = np.zeros(T)
# r_DKW = np.zeros(T)
# r_mar = np.zeros(T)
# for k in range(T):
#     x = np.sort(sample_cpvals(n,m))
#     y = np.arange(0, len(x)) / len(x)
#     r_s = 1
#     r_a = 1
#     r_d = 1
#     r_m = 1
#     for kk in range(len(x)):
#         if np.interp(x[kk], pgrid, upper_simes_1)<y[kk]:
#             r_s = 0
#         if np.interp(x[kk], pgrid, upper_asymptotic_1)<y[kk]:
#             r_a = 0
#         if np.interp(x[kk], pgrid, upper_simulation)<y[kk]:
#             r_d = 0
#         if np.interp(x[kk], pgrid, upper_marginal)<y[kk]:
#             r_m = 0
#     r_simes[k] = r_s
#     r_asy[k] = r_a
#     r_DKW[k] = r_d
#     r_mar[k] = r_m
# print(n,m)
# print(sum(r_simes)/T, sum(r_asy)/T, sum(r_DKW)/T, sum(r_mar)/T)

