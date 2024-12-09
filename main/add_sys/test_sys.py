import os
import h5py
import time
import random
import numpy as np
import sys
from scipy.special import erfinv
from numpy import log, pi,sqrt, exp,cos,sin,tan,argpartition,copy,trapz,mean,cov,vstack,hstack
random.seed(2143178)

z_dict      = {4:0.0, 3:0.5, 2:1.0, 1:2.0, 0:3.0}
r_dict      = {4:'pk', 5:'pkRU', 6:'QSO_like'}

catalogs    = 'fiducial' #fiducial, Mnu_p, Mnu_ppp
tool        = 'Pypower' # Powspec, Pypower
nb          = np.arange(100,200,1)
boxsize     = 1000
snapnum     = 2 #redshiftt
redshift    = z_dict[snapnum]
r_coord     = 6 #redshift uncertainty
r_pk        = r_dict[r_coord]

# assume LCDM without curvature
Om  = 0.3175
Ode = 1-Om
H   = 100*np.sqrt(Om*(1+redshift)**3+Ode)

def adjust_length(arr):
    if arr.size % 2 != 0:
        arr = arr[:-1]# If odd, remove the last element
    return arr

def inv_trans(uniform, func, sigma=None, maxdv=None, dvcatas=1000, dvcatasmax=10**5.65):
    if func == 'G':
        scathalf = int(len(uniform)/2)
        invfunc = np.append(sigma*sqrt(-2*log(uniform[:scathalf]))*cos(2*pi*uniform[-scathalf:]),sigma*sqrt(-2*log(uniform[:scathalf]))*sin(2*pi*uniform[-scathalf:]))
    elif func == 'L':
        invfunc = tan((2*uniform-1)*pi)*sigma/2
        outliers= abs(invfunc)>maxdv
        while len(outliers)>0:
            uni_tmp = np.random.RandomState(seed=int(time.time())).rand(np.sum(outliers))
            invfunc[outliers] = tan((2*uni_tmp-1)*pi)*sigma/2   
            outliers= abs(invfunc)>maxdv
    elif func == 'lnG':
        invfunc = 6.499-exp((253*erfinv(1-2000/991*uniform)+37*2**3.5)/125/2**2.5)    
        outliers= (invfunc<np.log10(dvcatas))|(invfunc>np.log10(dvcatasmax))|(~np.isfinite(invfunc))
        while np.sum(outliers)>0:
            uni_tmp = np.random.RandomState(seed=int(time.time())).rand(np.sum(outliers))
            invfunc[outliers] = 6.499-exp((253*erfinv(1-2000/991*uni_tmp)+37*2**3.5)/125/2**2.5)   
            outliers= (invfunc<np.log10(dvcatas))|(invfunc>np.log10(dvcatasmax))|(~np.isfinite(invfunc))
    return invfunc

def Z_vsmear(Z_position, uniform, target, redshift, boxsize, smeartype='G'):
    # redshift uncertainty with vsmear model
    if target == 'LRG':
        Lmaxdv  = 400 # if 'L', 400 for LRGs and 2000 for QSOss
        if redshift == 0.5:
            sigma = 37.2
        elif redshift == 1.0:
            sigma = 85.7
    elif target == 'QSO':
        Lmaxdv  = 2000 # if 'L', 400 for LRGs and 2000 for QSOss
        if redshift == 1.0:
            sigma = 200.0
    if   smeartype == 'G':
        vsmear = inv_trans(uniform, func='G', sigma=sigma, maxdv=None)
    elif smeartype == 'L':
        vsmear = inv_trans(uniform, func='L', sigma=sigma, maxdv=Lmaxdv)
    Z_position = (Z_position+vsmear*(1+redshift)/H)%boxsize
    return Z_position

def Z_vcatas(Z_position, uniform, redshift, boxsize):
    # implement the catastrophic failure
    # 1% catastrophics rate
    Nfail     = int(len(uniform)*1/100) if int(len(uniform)*1/100)%2==0 else int(len(uniform)*1/100)+1
    inds      = random.sample(range(0, len(uniform)), Nfail)
    dv_uniform= np.random.rand(Nfail)        
    half      = int(len(dv_uniform)/2)
    dv_pos    = inv_trans(dv_uniform[:half], func='lnG')
    dv_neg    = inv_trans(dv_uniform[half:], func='lnG')
    dv        = np.append(10**dv_pos,-10**dv_neg)
    random.shuffle(dv)
    vcatas    = np.zeros_like(uniform)
    vcatas[inds] = dv.copy()
    Z_position = (Z_position+vcatas*(1+redshift)/H)%boxsize
    return Z_position

def vvsmear(Z_position,redshift,boxsize,target):
    # the Gaussian dispersion of the redshift uncertainty distribution
    if target == 'LRG':
        if redshift == 0.5:
            sigma = 37.2
        elif redshift == 1.0:
            sigma = 85.7
    elif target == 'QSO':
        if redshift == 1.0:
            sigma = 200.0
    # cosmology for RSD
    # the redshift uncertainty effect in redshift space
    vsmear = np.random.normal(loc=1,scale=sigma,size=len(Z_position))*(1+redshift)/H
    Z_RSD_smeared = np.abs((Z_position+vsmear)%boxsize)
    return(Z_RSD_smeared)

import readfof
snapath     = f'/srv/astro/projects/cosmo3d/shhe/QUIJOTE/{catalogs}'
outputpath  = f'/home/astro/shhe/projectNU/main/data/{tool}/{r_pk}_z{redshift}/{catalogs}'
os.makedirs(outputpath, exist_ok=True)
h = 100
snapname = '/{}'.format(h)
FoF = readfof.FoF_catalog(snapath+snapname, snapnum, long_ids=False,
                        swap=False, SFR=False, read_IDs=False)
# get the properties of the halos
pos     = adjust_length(FoF.GroupPos/1e3)            #Halo positions in Mpc/h
mass    = adjust_length(FoF.GroupMass*1e10)          #Halo masses in Msun/h
vel     = adjust_length(FoF.GroupVel*(1.0+redshift)) #Halo peculiar velocities in km/s
#Npart  = FoF.GroupLen                #Number of CDM particles in the halo
pos_RSD = (pos[:,-1] + vel[:,-1]*(1+redshift)/H)%boxsize #pos[:-1] in Z coordinate
# pos_LRG = vvsmear(pos_RSD,redshift,boxsize,'LRG')
# pos_QSO = vvsmear(pos_RSD,redshift,boxsize,'QSO')

# print(pos.shape)
# print(vel[:,-1].shape)
uniform = np.random.RandomState(seed=int(time.time())).rand(len(pos))
print(len(uniform))
# print(uniform.shape)

pos_LRG     = Z_vsmear(pos_RSD, uniform, 'LRG', redshift, boxsize)
# pos_QSO     = Z_vsmear(pos_RSD, uniform, 'QSO', redshift, boxsize)
pos_catas   = Z_vcatas(pos_RSD, uniform, redshift, boxsize)
#import pdb;pdb.set_trace()
print(len(pos_catas))
print('is NAN? length:',np.sum(np.isnan(pos_catas)))
print('is inf? length:',np.sum(np.isinf(pos_catas)))

pos_reduce = pos_catas - pos_RSD
print(pos_reduce[pos_reduce!=0])
print(pos_LRG-pos_RSD)

# print(pos_catas)

