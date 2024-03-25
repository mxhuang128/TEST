from spectralradex import radex
import pandas as pd
from multiprocessing import Pool
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy.random import normal, multivariate_normal
from chainconsumer import ChainConsumer
from scipy.stats import norm
from nautilus import Prior

def get_radex(tkin,dens,col,file):
    params=radex.get_default_parameters()
    #params["fmin"]=80
    params["fmax"]=500
    params["tkin"]=tkin
    params["h2"]=dens
    params["linewidth"]=50
    params["cdmol"]=col
    params["molfile"]=file
    results = radex.run(params)
    if(results is not None): 
        return results["FLUX (K*km/s)"].values
    else: 
        return None

#############################
### Forget Ratios
OBS_S = np.genfromtxt('LineIntensityM3_H2CO.txt', skip_header=2)
SIG_S = np.genfromtxt('Sig_LineIntensityM3_H2CO.txt', skip_header=2)
#REG = int(3) #CND-R1 = 1 (AGN at 0): R1 to R5... SBN, SBS
#LABEL_reg = "CND_R"+str(REG)
LABEL_reg = ['GMC1', 'GMC2', 'GMC3', 'GMC4', 'GMC5', 'GMC6', 'GMC7', 'GMC8', 'GMC9', 'GMC10', 'GMC1a', 'GMC1b', 'GMC2a', 'GMC2b', 'GMC8p', 'GMC9p']
REG = int(14) # GMC1= OBS[:, 0] -- start from GMC3-7, GMC1a & 9p=10 & 15
I_OBS_S = list(OBS_S[:, REG])
I_SIG_S = list(SIG_S[:, REG])
obs = np.asarray(I_OBS_S)
errors = np.asarray(I_SIG_S)

Transition_fullList = np.genfromtxt("./Compiled_H2CO_FreqList.txt", skip_header=1)
mainline_freq = Transition_fullList[:,0]
dat = pd.Series(mainline_freq)
DUP_label = dat.duplicated(keep=False).values
flt_desired_index = Transition_fullList[:,4] - 1 # LAMDA transition count from 1 instead of 0
desired_index = flt_desired_index.astype(int)
isomer_index = np.genfromtxt("./Compiled_H2CO_FreqList.txt", skip_header=1, dtype=str)[:,1]

#Bayesian inference requires prior probability distributions.
#Let's assume the simple case that the probability is 1 within limits
#and 0 elsewhere. So log prior is -infinity or 0
prior = Prior()
prior.add_parameter('a', dist=(+2, +8))
prior.add_parameter('b', dist=(+10, +800))
prior.add_parameter('c', dist=(+12, +18))
prior.add_parameter('d', dist=(+12, +18))
prior.add_parameter('e', dist=(0, +1))

def likelihood(param_dict):
    x = np.array([10.0**param_dict['a'], param_dict['b'], 10.0**param_dict['c'], 10.0**param_dict['d'], param_dict['e']])
    Amodel = get_radex(x[1],x[0],x[2],"oh2co-h2.dat")
    Emodel = get_radex(x[1],x[0],x[3],"ph2co-h2.dat")
    if(Emodel is None or Amodel is None): 
        return -np.inf
    else: 
        N_ind = len(desired_index) #num of total transitions, should be 10 for H2CO from 9 cubes
        mmodel = []
        I_component = 0.0
        for k in range(N_ind):
            if(isomer_index[k] == "o-"): I_component = Amodel[desired_index[k]]
            elif(isomer_index[k] == "p-"): I_component = Emodel[desired_index[k]]
            else: print("SOMETHING IS WRONGGGGG")
            if(DUP_label[k]):
                place_holder += I_component
                if(not DUP_label[k+1]): mmodel.append(place_holder)
            else:
                place_holder = I_component
                mmodel.append(place_holder)
        Mmodel=np.asarray(mmodel)
        Mmodel *= x[4] #multiply by beam ff
        chi=Mmodel-obs
        chi=-0.5*np.sum((chi*chi)/(errors*errors))
        return chi

from nautilus import Sampler

sampler = Sampler(prior, likelihood, filepath='results_NTLS'+LABEL_reg[REG]+'.hdf5', resume=True, pool=8)
sampler.run(verbose=True)
import corner
import matplotlib.pyplot as plt

points, log_w, log_l = sampler.posterior()
ndim = points.shape[1]
fig = corner.corner(points, weights=np.exp(log_w), smooth=1, 
                   labels=["$log_{10}$n[H2]", "Tkin", "$log_{10}$N[o-H2CO]", "$log_{10}$N[p-H2CO]", "Filling Factor"],
                   levels=(0.68, 0.95),show_titles=True,color='orange', #fill_contours=True,
                   range=[(2.0,8.0),(10.0,800.0),(12.0,18.0),(12.0,18.0),(0.0,1.0)])
fig.savefig("H2CO_"+LABEL_reg[REG]+"_cornerNTLS.png")
#fig, axes = plt.subplots(ndim, ndim, figsize=(3.5, 3.5))
#FIG = corner.corner(points, weights=np.exp(log_w), bins=20, labels=prior.keys,
#                    plot_datapoints=False, plot_density=False,
#                    fill_contours=True, levels=(0.68, 0.95),
#                    range=np.ones(ndim) * 0.999, fig=fig)
#FIG.savefig(LABEL_reg[REG]+'_H2CO_NTLS_800K.png', dpi=150)

#sampler = Sampler(prior, likelihood, n_live=1000)
#sampler.run(verbose=True)

#import ultranest
#param_names=["$log_{10}$n[H2]", "Tkin", "$log_{10}$N[o-H2CO]", "$log_{10}$N[p-H2CO]", "Filling factor"]
#DIR_name = 'LOG_opH2CO_'+LABEL_reg[REG]
#sampler = ultranest.ReactiveNestedSampler(param_names, likelihood, prior, log_dir=DIR_name, resume='subfolder')
#result = sampler.run(min_ess=400, min_num_live_points=400)
#from ultranest.plot import cornerplot
#sampler.plot()

