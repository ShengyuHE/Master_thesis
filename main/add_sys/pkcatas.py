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

catalogue   = 'fiducial' #fiducial, Mnu_p, Mnu_ppp
nb          = np.arange(100,200,1)
boxsize     = 1000
redshift    = 1.0
catas       = '5%CATAS'
systematic  = catas # RSD, LRG, QSO, 1%CATAS, 5%CATAS

z_dict      = {0.0:4, 0.5:3, 1.0:2, 2.0:1, 3.0:0}
r_dict      = {'RSD':4, catas:5}

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

def Z_vcatas(Z_position, uniform, systematic, redshift, boxsize):
    # implement the catastrophic failure
    global inds
    if systematic == '1%CATAS':
        catas_rate = 1/100
    elif systematic == '5%CATAS':
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

def catalog_to_Pk_mupltiples():
    """""
    this function compute the pk multiples in redshift space and add systematic error reshift uncertainty on it 
    it is based on QUIJOTE catalogs mocks and uses powspec or pypower to compute the pk
    """""
    import readfof
    snapnum     = z_dict[redshift]
    snapath     = f'/srv/astro/projects/cosmo3d/shhe/QUIJOTE/{catalogue}'
    # outputpath  = f'/home/astro/shhe/projectNU/main/data/{tool}/{catalogue}/{systematic}_z{redshift}'
    outputpath = f'/home/astro/shhe/projectNU/main/data/kbin2/auto_pkl/{systematic}_z{redshift}'
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
        pos_catas  = Z_vcatas(pos_rsd, uniform, systematic ,redshift, boxsize)
        pos     = pos.T

        # the position of halos affected by the vcatas
        pos0   = np.array([pos[0][inds], pos[1][inds], pos[2][inds], pos_rsd[inds], pos_catas[inds]]).T
        DATA0  = np.hstack((mass[inds].reshape(len(pos0),1),pos0))

        # the position of halos not affected by the vcatas
        inds1       = [x for x in range(len(pos[0])) if x not in inds]
        pos1        = np.array([pos[0][inds1], pos[1][inds1], pos[2][inds1], pos_rsd[inds1], pos_catas[inds1]]).T
        DATA1       = np.hstack((mass[inds1].reshape(len(pos1),1),pos1))

        # Mass ,X ,Y, Z, Z_RSD, Z_RSD_vsmear#
        "SELECT HALO"
        def Select(data):
            select_data = []      
            for halo in data:
                if np.log10(halo[0])>13.1 and np.log10(halo[0])<13.5:
                    select_data.append(halo) 
            return np.array(select_data)   
        DATA0 = Select(DATA0)
        DATA1 = Select(DATA1)

        for sys in [catas]:
            from pypower import CatalogFFTPower
            r_coord    = r_dict[sys]
            ells = (0,2,4) # monopole, quadruple and hexadrupole
            k_edge = np.arange(0.005, 0.205, 0.006)
            result = CatalogFFTPower(data_positions1 = [DATA0[:,1]%1000, DATA0[:,2]%1000, DATA0[:,r_coord]%1000],
                                    #  data_positions2 = [DATA1[:,1]%1000, DATA1[:,2]%1000, DATA1[:,r_coord]%1000],
                                    edges=k_edge, ells=ells, interlacing=3, boxsize=1000., nmesh=512, resampler='tsc',
                                    los=(0,0,1), position_type='xyz', mpiroot=0)
            # result.save(outputpath+f'/npy/{catalogue}_{h}_z{redshift}.npy') # saved in binary files
            poles = result.poles
            poles.save_txt(outputpath+f'/{catalogue}_{h}_z{redshift}.pk', complex=False) # saved in txt files
    return 0


catalog_to_Pk_mupltiples()
