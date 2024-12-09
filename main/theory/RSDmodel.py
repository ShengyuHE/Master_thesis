import os
import numpy as np
import sys
sys.path.append('./FOLPS-nu/')
import FOLPSnu as FOLPS
import scipy
from scipy.interpolate import CubicSpline
from scipy.special import gamma
from scipy.special import spherical_jn
from scipy.special import eval_legendre

# P_Kaiser = ['h', 'omega_cdm', 'logA', 'b1', 'b2', 'sv']
# p_TNS =  ['h', 'omega_cdm', 'logA', 'b1', 'b2', 'sv']
# p_FOLPS =  ['h', 'omega_cdm', 'logA', 'b1', 'b2', 'alpha0', 'alpha2', 'sn0', 'sn2']


'cosmological constants'
global z_pk, n_s, Mnu, f0
# Omega_b = 0.049
# omega_b = 0.0220684
n_s = 0.9649
Mnu = 0.06

'Cosmological independent matrix'
matrices = FOLPS.Matrices() # 10s

def interp(k, x, y):
        inter = CubicSpline(x, y)
        return inter(k) 

def Pklinear(p, z_pk):
    # T0 = time.time()
    from classy import Class
    # pklinear_fn = f'pkl_cb_z{z_pk}.dat'
    # if not os.path.exists(pklinear_fn): 
    k_min = 0.10000E-03
    k_max = 0.10000E+03
    (h, omega_cdm, omega_b, logA)= p[0:4]
    nuCDM = Class()
    params = {'omega_b':omega_b, 'omega_cdm':omega_cdm, 'h':h, 'ln10^{10}A_s':logA, 'n_s':n_s, 
            'N_eff':3.045998221453431,  'N_ncdm':1, 'm_ncdm':Mnu, 
            #   'tau_reio':0.09,'YHe':0.24,
            'output':'mPk','z_pk':z_pk,'P_k_max_1/Mpc':k_max}
    nuCDM.set(params)
    nuCDM.compute()
    kk = np.logspace(np.log10(k_min), np.log10(k_max), num = 312) #Mpc^-1
    Pk_linear=[]
    for k in kk:
        Pk_linear.append([k, nuCDM.pk(k*nuCDM.h(),z_pk)*nuCDM.h()**3])
    nuCDM.empty()
    Pk_linear = np.array(Pk_linear).T
    # np.savetxt(pklinear_fn, Pk_linear)
    return Pk_linear
    # else:
    #     return np.loadtxt(pklinear_fn)
    

def Kaisermultiples(p,kev):
    (h, omega_cdm, logA)= p[0:3]
    (b1, sv)=p[3:5]
    inputpkT=Pklinear(p) # linear power spectrum
    omega_ncdm = Mnu/93.14
    CosmoParams = [z_pk, omega_b, omega_cdm, omega_ncdm, h]
    nonlinear = FOLPS.NonLinear(inputpkT, CosmoParams, EdSkernels = True)
    def PRSDs(k, mu, Table):
        f0 =FOLPS.f0
        (pkl, Fkoverf0, Ploop_dd, Ploop_dt, Ploop_tt, Pb1b2, Pb1bs2, Pb22, Pb2bs2, 
         Pb2s2, sigma23pkl, Pb2t, Pbs2t, I1udd_1, I2uud_1, I2uud_2, I3uuu_2, I3uuu_3, 
         I2uudd_1D, I2uudd_2D, I3uuud_2D, I3uuud_3D, I4uuuu_2D, I4uuuu_3D, I4uuuu_4D, sigma2w) = Table
        fk = Fkoverf0*f0
        #linear power spectrum
        Pdt_L = pkl*Fkoverf0; Ptt_L = pkl*Fkoverf0**2;
        #Linear Kaiser power spectrum
        def PKaiserLs(mu, b1):
            return (b1 + mu**2 * fk)**2 * pkl
        def FoG(sv):
            return np.exp(-(k*mu*sv*fk)**2)
        P_s = FoG(sv)*PKaiserLs(mu, b1)
        return P_s
    def PIRs(kev, mu, Table, Table_NW):
        f0 =FOLPS.f0
        k = kev; Fkoverf0 = Table[1]; fk = Fkoverf0*f0
        pkl = Table[0]; pkl_NW = Table_NW[0];
        Sigma2T = FOLPS.Sigma2Total(kev, mu, Table_NW)
        return ((b1 + fk * mu**2)**2 * (pkl_NW + np.exp(-k**2 * Sigma2T)*(pkl-pkl_NW)*(1 + k**2 * Sigma2T) )
            + np.exp(-k**2 * Sigma2T)*PRSDs(k, mu, Table) 
            + (1 - np.exp(-k**2 * Sigma2T))*PRSDs(k, mu, Table_NW)) 
    Nx = 8                                         #Points
    xGL, wGL = scipy.special.roots_legendre(Nx)    #x=cosθ and weights
    def ModelPkl0(Table, Table_NW):
        monop = 0;
        for ii in range(Nx):
            monop = monop + 0.5*wGL[ii]*PIRs(kev, xGL[ii], Table, Table_NW)
        return monop
    def ModelPkl2(Table, Table_NW):    
        quadrup = 0;
        for ii in range(Nx):
            quadrup = quadrup + 5/2*wGL[ii]*PIRs(kev, xGL[ii], Table, Table_NW)*eval_legendre(2, xGL[ii])
        return quadrup
    def ModelPkl4(Table, Table_NW):
        hexadecap = 0;
        for ii in range(Nx):
            hexadecap = hexadecap + 9/2*wGL[ii]*PIRs(kev, xGL[ii], Table, Table_NW)*eval_legendre(4, xGL[ii])
        return hexadecap

    Pkl0 = ModelPkl0(FOLPS.TableOut_interp(kev), FOLPS.TableOut_NW_interp(kev));
    Pkl2 = ModelPkl2(FOLPS.TableOut_interp(kev), FOLPS.TableOut_NW_interp(kev));
    return(kev, Pkl0, Pkl2)


def TNSmultiples(p,kev):
    (h, omega_cdm, omega_b, logA)= p[0:3]
    (b1, b2, sv)=p[3:6]
    inputpkT=Pklinear(p) # linear power spectrum
    omega_ncdm = Mnu/93.14
    CosmoParams = [z_pk, omega_b, omega_cdm, omega_ncdm, h]
    bs2 = -4/7*(b1 - 1);        #coevolution
    b3nl = 32/315*(b1 - 1);     #coevolution
    alpha4 = 0.0;  # not considered  --> monopole, quadrupole
    ctilde = 0.0;  # fix --> alphashot2
    PshotP = 1/0.0002118763; # constant --> Pshot = 1/n ̄x = 4719.7 h3Mpc−3
    nonlinear = FOLPS.NonLinear(inputpkT, CosmoParams, EdSkernels = True)

    def PRSDs(k, mu, Table):
        f0 =FOLPS.f0
        (pkl, Fkoverf0, Ploop_dd, Ploop_dt, Ploop_tt, Pb1b2, Pb1bs2, Pb22, Pb2bs2, 
         Pb2s2, sigma23pkl, Pb2t, Pbs2t, I1udd_1, I2uud_1, I2uud_2, I3uuu_2, I3uuu_3, 
         I2uudd_1D, I2uudd_2D, I3uuud_2D, I3uuud_3D, I4uuuu_2D, I4uuuu_3D, I4uuuu_4D, sigma2w) = Table
        fk = Fkoverf0*f0
        #linear power spectrum
        Pdt_L = pkl*Fkoverf0; Ptt_L = pkl*Fkoverf0**2;
        #one-loop power spectrum 
        Pdd = pkl + Ploop_dd; Pdt = Pdt_L + Ploop_dt; Ptt = Ptt_L + Ploop_tt;
        #biasing
        def PddXloop(b1, b2, bs2, b3nl):
            return (b1**2 * Ploop_dd + 2*b1*b2*Pb1b2 + 2*b1*bs2*Pb1bs2 + b2**2 * Pb22
                    + 2*b2*bs2*Pb2bs2 + bs2**2 *Pb2s2 + 2*b1*b3nl*sigma23pkl)
            
        def PdtXloop(b1, b2, bs2, b3nl):
            return b1*Ploop_dt + b2*Pb2t + bs2*Pbs2t + b3nl*Fkoverf0*sigma23pkl
            
        def PttXloop(b1, b2, bs2, b3nl):
            return Ploop_tt
        #RSD functions      
        def Af(mu, f0):
            return (f0*mu**2 * I1udd_1 + f0**2 * (mu**2 * I2uud_1 + mu**4 * I2uud_2)
                        + f0**3 * (mu**4 * I3uuu_2 +  mu**6 * I3uuu_3)) 
            
        def Df(mu, f0):
            return (f0**2 * (mu**2 * I2uudd_1D + mu**4 * I2uudd_2D) 
                        + f0**3 * (mu**4 * I3uuud_2D + mu**6 * I3uuud_3D)
                        + f0**4 * (mu**4 * I4uuuu_2D + mu**6 * I4uuuu_3D + mu**8 * I4uuuu_4D))
        #Introducing bias in RSD functions, eq.~ A.32 & A.33 at arXiv: 2208.02791
        def ATNS(mu, b1):
            return b1**3 * Af(mu, f0/b1)
            
        def DRSD(mu, b1):
            return b1**4 * Df(mu, f0/b1)
        
        def FoG(sv):
            return np.exp(-(k*mu*sv*fk)**2)
            
        P_s = FoG(sv)*(PddXloop(b1, b2, bs2, b3nl) + 2*f0*mu**2 * PdtXloop(b1, b2, bs2, b3nl) 
               + mu**4 * f0**2 * PttXloop(b1, b2, bs2, b3nl)+ ATNS(mu, b1) + DRSD(mu, b1))
        return P_s

    def PIRs(kev, mu, Table, Table_NW):
        f0 =FOLPS.f0
        k = kev; Fkoverf0 = Table[1]; fk = Fkoverf0*f0
        pkl = Table[0]; pkl_NW = Table_NW[0];
        Sigma2T = FOLPS.Sigma2Total(kev, mu, Table_NW)
        
        return ((b1 + fk * mu**2)**2 * (pkl_NW + np.exp(-k**2 * Sigma2T)*(pkl-pkl_NW)*(1 + k**2 * Sigma2T) )
            + np.exp(-k**2 * Sigma2T)*PRSDs(k, mu, Table) 
            + (1 - np.exp(-k**2 * Sigma2T))*PRSDs(k, mu, Table_NW)) 

    Nx = 8                                         #Points
    xGL, wGL = scipy.special.roots_legendre(Nx)    #x=cosθ and weights
    def ModelPkl0(Table, Table_NW):
        monop = 0;
        for ii in range(Nx):
            monop = monop + 0.5*wGL[ii]*PIRs(kev, xGL[ii], Table, Table_NW)
        return monop
    def ModelPkl2(Table, Table_NW):    
        quadrup = 0;
        for ii in range(Nx):
            quadrup = quadrup + 5/2*wGL[ii]*PIRs(kev, xGL[ii], Table, Table_NW)*eval_legendre(2, xGL[ii])
        return quadrup
    def ModelPkl4(Table, Table_NW):
        hexadecap = 0;
        for ii in range(Nx):
            hexadecap = hexadecap + 9/2*wGL[ii]*PIRs(kev, xGL[ii], Table, Table_NW)*eval_legendre(4, xGL[ii])
        return hexadecap

    Pkl0 = ModelPkl0(FOLPS.TableOut_interp(kev), FOLPS.TableOut_NW_interp(kev));
    Pkl2 = ModelPkl2(FOLPS.TableOut_interp(kev), FOLPS.TableOut_NW_interp(kev));
    return(kev, Pkl0, Pkl2)


def FOLPSmultiples(p, kev, z_pk):
    (h, omega_cdm, omega_b, logA)= p[0:4]
    (b1, b2, bs2, b3nl, alpha0, alpha2, sn0, sn2)=p[4:12]
    inputpkT=Pklinear(p, z_pk) # linear power spectrum
    omega_ncdm = Mnu/93.14
    CosmoParams = [z_pk, omega_b, omega_cdm, omega_ncdm, h]
    # bs2 = -4/7*(b1 - 1);        #coevolution
    # b3nl = 32/315*(b1 - 1);     #coevolution
    alpha4 = 0.0;  # not considered  --> monopole, quadrupole
    ctilde = 0.0;  # fix --> alphashot2
    PshotP = 1/0.0002118763; # constant --> Pshot = 1/n ̄x = 4719.7 h3Mpc−3
    NuisanParams = [b1, b2, bs2, b3nl, alpha0, alpha2, alpha4, 
                ctilde, sn0, sn2, PshotP]
    nonlinear = FOLPS.NonLinear(inputpkT, CosmoParams, EdSkernels = True)
    kh, Pkl0, Pkl2, Pkl4 = FOLPS.RSDmultipoles(kev, NuisanParams, AP = False)
    return(kh, Pkl0, Pkl2)


def FOLPSmultipoles_smear(p, kev, z_pk):
    (h, omega_cdm, omega_b, logA)= p[0:4]
    (b1, b2, bs2, b3nl, alpha0, alpha2, sn0, sn2, sv)=p[4:11]
    inputpkT=Pklinear(p,z_pk) # linear power spectrum
    omega_ncdm = Mnu/93.14
    CosmoParams = [z_pk, omega_b, omega_cdm, omega_ncdm, h]
    bs2 = -4/7*(b1 - 1);        #coevolution
    b3nl = 32/315*(b1 - 1);     #coevolution
    alpha4 = 0.0;  # not considered  --> monopole, quadrupole
    ctilde = 0.0;  # fix --> alphashot2
    PshotP = 1/0.0002118763; # constant --> Pshot = 1/n ̄x = 4719.7 h3Mpc−3
    nonlinear = FOLPS.NonLinear(inputpkT, CosmoParams, EdSkernels = True)

    def PEFTs(kev, mu, Table):
        f0 =FOLPS.f0
        #Table
        (pkl, Fkoverf0, Ploop_dd, Ploop_dt, Ploop_tt, Pb1b2, Pb1bs2, Pb22, Pb2bs2, 
            Pb2s2, sigma23pkl, Pb2t, Pbs2t, I1udd_1, I2uud_1, I2uud_2, I3uuu_2, I3uuu_3, 
            I2uudd_1D, I2uudd_2D, I3uuud_2D, I3uuud_3D, I4uuuu_2D, I4uuuu_3D, I4uuuu_4D, sigma2w) = Table
        
        fk = Fkoverf0*f0
        #linear power spectrum
        Pdt_L = pkl*Fkoverf0; Ptt_L = pkl*Fkoverf0**2;   
        #one-loop power spectrum 
        Pdd = pkl + Ploop_dd; Pdt = Pdt_L + Ploop_dt; Ptt = Ptt_L + Ploop_tt;
        #biasing
        def PddXloop(b1, b2, bs2, b3nl):
            return (b1**2 * Ploop_dd + 2*b1*b2*Pb1b2 + 2*b1*bs2*Pb1bs2 + b2**2 * Pb22
                    + 2*b2*bs2*Pb2bs2 + bs2**2 *Pb2s2 + 2*b1*b3nl*sigma23pkl)
            
        def PdtXloop(b1, b2, bs2, b3nl):
            return b1*Ploop_dt + b2*Pb2t + bs2*Pbs2t + b3nl*Fkoverf0*sigma23pkl
            
        def PttXloop(b1, b2, bs2, b3nl):
            return Ploop_tt
            
        #RSD functions       
        def Af(mu, f0):
            return (f0*mu**2 * I1udd_1 + f0**2 * (mu**2 * I2uud_1 + mu**4 * I2uud_2)
                        + f0**3 * (mu**4 * I3uuu_2 +  mu**6 * I3uuu_3)) 
            
        def Df(mu, f0):
            return (f0**2 * (mu**2 * I2uudd_1D + mu**4 * I2uudd_2D) 
                        + f0**3 * (mu**4 * I3uuud_2D + mu**6 * I3uuud_3D)
                        + f0**4 * (mu**4 * I4uuuu_2D + mu**6 * I4uuuu_3D + mu**8 * I4uuuu_4D))
            
        #Introducing bias in RSD functions, eq.~ A.32 & A.33 at arXiv: 2208.02791
        def ATNS(mu, b1):
            return b1**3 * Af(mu, f0/b1)
            
        def DRSD(mu, b1):
            return b1**4 * Df(mu, f0/b1)
            
        def GTNS(mu, b1):
            return -((kev*mu*f0)**2 *sigma2w*(b1**2 * pkl + 2*b1*f0*mu**2 * Pdt_L 
                                    + f0**2 * mu**4 * Ptt_L))
            
            
        #One-loop SPT power spectrum in redshift space
        def PloopSPTs(mu, b1, b2, bs2, b3nl):
            return (PddXloop(b1, b2, bs2, b3nl) + 2*f0*mu**2 * PdtXloop(b1, b2, bs2, b3nl)
                        + mu**4 * f0**2 * PttXloop(b1, b2, bs2, b3nl) + ATNS(mu, b1) + DRSD(mu, b1)
                        + GTNS(mu, b1))
            
            
        #Linear Kaiser power spectrum
        def PKaiserLs(mu, b1):
            return (b1 + mu**2 * fk)**2 * pkl
            
        def PctNLOs(mu, b1, ctilde):
            return ctilde*(mu*kev*f0)**4 * sigma2w**2 * PKaiserLs(mu, b1)
        
        # EFT counterterms
        def Pcts(mu, alpha0, alpha2, alpha4):
            return (alpha0 + alpha2 * mu**2 + alpha4 * mu**4)*kev**2 * pkl
        
        #Stochastics noise
        def Pshot(mu, alphashot0, alphashot2, PshotP):
            return PshotP*(alphashot0 + alphashot2 * (kev*mu)**2)
            
        return (PloopSPTs(mu, b1, b2, bs2, b3nl) + Pcts(mu, alpha0, alpha2, alpha4)
                    + PctNLOs(mu, b1, ctilde) + Pshot(mu, sn0, sn2, PshotP))*np.exp(-(kev*mu*sv)**2)

    def PIRs(kev, mu, Table, Table_NW):
        f0 =FOLPS.f0
        k = kev; Fkoverf0 = Table[1]; fk = Fkoverf0*f0
        pkl = Table[0]; pkl_NW = Table_NW[0];
        Sigma2T = FOLPS.Sigma2Total(kev, mu, Table_NW)
        
        return ((b1 + fk * mu**2)**2 * (pkl_NW + np.exp(-k**2 * Sigma2T)*(pkl-pkl_NW)*(1 + k**2 * Sigma2T) )
            + np.exp(-k**2 * Sigma2T)*PEFTs(k, mu, Table) 
            + (1 - np.exp(-k**2 * Sigma2T))*PEFTs(k, mu, Table_NW))
    
    Nx = 8                                         #Points
    xGL, wGL = scipy.special.roots_legendre(Nx)    #x=cosθ and weights
    def ModelPkl0(Table, Table_NW):
        monop = 0;
        for ii in range(Nx):
            monop = monop + 0.5*wGL[ii]*PIRs(kev, xGL[ii], Table, Table_NW)
        return monop
    def ModelPkl2(Table, Table_NW):    
        quadrup = 0;
        for ii in range(Nx):
            quadrup = quadrup + 5/2*wGL[ii]*PIRs(kev, xGL[ii], Table, Table_NW)*eval_legendre(2, xGL[ii])
        return quadrup
    def ModelPkl4(Table, Table_NW):
        hexadecap = 0;
        for ii in range(Nx):
            hexadecap = hexadecap + 9/2*wGL[ii]*PIRs(kev, xGL[ii], Table, Table_NW)*eval_legendre(4, xGL[ii])
        return hexadecap

    Pkl0 = ModelPkl0(FOLPS.TableOut_interp(kev), FOLPS.TableOut_NW_interp(kev));
    Pkl2 = ModelPkl2(FOLPS.TableOut_interp(kev), FOLPS.TableOut_NW_interp(kev));
    return(kev, Pkl0, Pkl2)





