import os
import h5py
import time
import numpy as np
import random
import sys
from scipy.special import erfinv
from numpy import log, pi,sqrt, exp,cos,sin,tan,argpartition,copy,trapz,mean,cov,vstack,hstack

# z_dict      = {4:0.0, 3:0.5, 2:1.0, 1:2.0, 0:3.0}
# r_dict      = {4:'RSD', 5:'LRG', 6:'QSO', 7: '1%CATAS', 8:'5%CATAS'}

z_dict      = {0.0:4, 0.5:3, 1.0:2, 2.0:1, 3.0:0}
r_dict      = {'RSD':4, 'LRG':5, 'QSO':6, '1%CATAS':7, '5%CATAS':8}

catalogue   = 'fiducial' #fiducial, Mnu_p, Mnu_ppp
tool        = 'Pypower' # Powspec, Pypower
nb          = np.arange(100,200,1)
boxsize     = 1000
redshift    = 1.0
r_pk        = 'RSD' # RSD, LRG, QSO, 1%CATAS, 5%CATAS
snapnum     = z_dict[redshift]
r_coord     = r_dict[r_pk]

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
        else:
            sigma = 0.0
    elif target == 'QSO':
        Lmaxdv  = 2000 # if 'L', 400 for LRGs and 2000 for QSOss
        if redshift == 0.5:
            sigma = 100.0
        if redshift == 1.0:
            sigma = 200.0
        else:
            sigma = 0.0
    if smeartype == 'G':
        vsmear = inv_trans(uniform, func='G', sigma=sigma, maxdv=None)
    elif smeartype == 'L':
        vsmear = inv_trans(uniform, func='L', sigma=sigma, maxdv=Lmaxdv)
    Z_position = (Z_position+vsmear*(1+redshift)/H)%boxsize
    return Z_position

def Z_vcatas(Z_position, uniform, rate, redshift, boxsize):
    # implement the catastrophic failure
    if rate == '1%CATAS':
        catas_rate = 1/100
    elif rate == '5%CATAS':
        catas_rate = 5/100
    else:
        catas_rate = 0
    Nfail       = int(len(uniform)*catas_rate) if int(len(uniform)*catas_rate)%2==0 else int(len(uniform)*catas_rate)+1
    inds        = random.sample(range(0, len(uniform)), Nfail)
    dv_uniform  = np.random.rand(Nfail)        
    half        = int(len(dv_uniform)/2)
    dv_pos      = inv_trans(dv_uniform[:half], func='lnG')
    dv_neg      = inv_trans(dv_uniform[half:], func='lnG')
    dv          = np.append(10**dv_pos,-10**dv_neg)
    random.shuffle(dv)
    vcatas    = np.zeros_like(uniform)
    vcatas[inds] = dv.copy()
    Z_position = (Z_position+vcatas*(1+redshift)/H)%boxsize
    return Z_position

def catalog_to_Pk_mupltiples(catalogue, r_pk, r_coord, redshift):
    """""
    this function compute the pk multiples in redshift space and add systematic error reshift uncertainty on it 
    it is based on QUIJOTE catalogs mocks and uses powspec or pypower to compute the pk
    """""
    import readfof
    snapath     = f'/srv/astro/projects/cosmo3d/shhe/QUIJOTE/{catalogue}'
    # outputpath  = f'/home/astro/shhe/projectNU/main/data/{tool}/{catalogue}/{r_pk}_z{redshift}'
    outputpath = f'/home/astro/shhe/projectNU/main/data/Mnu/{catalogue}/{r_pk}_z{redshift}'
    os.makedirs(outputpath, exist_ok=True)
    print(outputpath)
    for h in nb:
        snapname = '/{}'.format(h)
        FoF = readfof.FoF_catalog(snapath+snapname, snapnum, long_ids=False,
                                swap=False, SFR=False, read_IDs=False)
        # get the properties of the halos
        pos     = adjust_length(FoF.GroupPos/1e3)            #Halo positions in Mpc/h
        mass    = adjust_length(FoF.GroupMass*1e10)          #Halo masses in Msun/h
        vel     = adjust_length(FoF.GroupVel*(1.0+redshift)) #Halo peculiar velocities in km/s
        #Npart  = FoF.GroupLen                #Number of CDM particles in the halo
        uniform     = np.random.RandomState(seed=int(time.time())).rand(len(pos))
        pos_rsd     = (pos[:,-1] + vel[:,-1]*(1+redshift)/H)%boxsize #pos[:-1] in Z coordinate
        pos_lrg     = Z_vsmear(pos_rsd, uniform, 'LRG', redshift, boxsize)
        pos_qso     = Z_vsmear(pos_rsd, uniform, 'QSO', redshift, boxsize)
        pos_1catas  = Z_vcatas(pos_rsd, uniform, '1%CATAS' ,redshift, boxsize)
        pos_5catas  = Z_vcatas(pos_rsd, uniform, '5%CATAS' ,redshift, boxsize)
        DATA = np.hstack((mass.reshape(len(pos),1),pos))
        DATA = np.hstack((DATA,pos_rsd.reshape(len(pos),1)))
        DATA = np.hstack((DATA,pos_lrg.reshape(len(pos),1)))
        DATA = np.hstack((DATA,pos_qso.reshape(len(pos),1)))
        DATA = np.hstack((DATA,pos_1catas.reshape(len(pos),1)))
        DATA = np.hstack((DATA,pos_5catas.reshape(len(pos),1)))
        # Mass ,X ,Y, Z, Z_RSD, Z_RSD_vsmear#
        "SELECT HALO"
        SELECT_DATA = []
        for halo in DATA:
            if np.log10(halo[0])>13.1 and np.log10(halo[0])<13.5:
                SELECT_DATA.append(halo)
        SELECT_DATA = np.array(SELECT_DATA)
        if tool == 'Powspec': #Compute Pk multiples with Powspec
            from pypowspec import compute_auto_box
            confpath = '/home/astro/shhe/projectNU/main/data/powspec.conf'
            w = np.ones_like(SELECT_DATA[:,1])
            pk = compute_auto_box(SELECT_DATA[:,1]%1000, SELECT_DATA[:,2]%1000, SELECT_DATA[:,r_coord]%1000, w, 
                                powspec_conf_file = confpath,
                                output_file = outputpath+f'/{catalogue}_{h}_z{redshift}.pk')
        elif tool == 'Pypower': #Compute Pk multiples with Pypower
            from pypower import CatalogFFTPower
            ells = (0,2,4) # monopole, quadruple and hexadrupole
            k_edge = np.arange(0.005, 0.205, 0.006)
            # k_edge = np.arange(0.005, 0.252, 0.006)
            # k_edge = np.arange(0.000, 0.300, 0.005)
            result = CatalogFFTPower(data_positions1=[SELECT_DATA[:,1]%1000, SELECT_DATA[:,2]%1000, SELECT_DATA[:,r_coord]%1000],
                                    edges=k_edge, ells=ells, interlacing=3, boxsize=1000., nmesh=512, resampler='tsc',
                                    los=(0,0,1), position_type='xyz', mpiroot=0)
            result.save(outputpath+f'/npy/{catalogue}_{h}_z{redshift}.npy') # saved in binary files
            poles = result.poles
            poles.save_txt(outputpath+f'/pk/{catalogue}_{h}_z{redshift}.pk', complex=False) # saved in txt files
    return 0
        
for catalogue in ['fiducial','Mnu_p','Mnu_ppp']:
#     redshift    = 1.0
#     r_coord     = r_dict[r_pk]
    catalog_to_Pk_mupltiples(catalogue, r_pk, r_coord, redshift)