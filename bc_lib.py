#!/usr/bin/env python

import sys
import numpy as np
import numpy.ma as ma
import inspect
import xarray as xr
np.set_printoptions(suppress=True)
import pandas as pd

# Get matlab style percentile values
def matlab_percentile(x, p):
    p = np.asarray(p, dtype=float)
    n = len(x)
    p = (p-50)*n/(n-1) + 50
    p = np.clip(p, 0, 100)
    return np.percentile(x, p)


def matlab_percentile(x):
    print(x)
    print(len(x))
    p = [x for x in range(0,101)]
    p = np.asarray(p, dtype=float)
    n = len(x)
    p = (p-50)*n/(n-1) + 50
    p = np.clip(p, 0, 100)
    return np.percentile(x, p)

# Helper for percentile formatting
def get_pct(data, i):
    if ma.is_masked(data):
        data = data.compressed()
    return matlab_percentile(data, i)

# Helper for percentile formatting
def get_pctmm(data, i):
    if len(data) == 0:
        return [np.nan, np.nan]
    imin = get_pct(data, i-1)
    imax = get_pct(data, i)
    return [imin, imax]


def apply_coldsnap_bc(obs, his, fut):
    [obs_imin, obs_imax] = get_pctmm(obs, 0.01)
    [his_imin, his_imax] = get_pctmm(his, 0.01)
    [fut_imin, fut_imax] = get_pctmm(fut, 0.01)
    fobs = ma.where((obs <= obs_imax))
    fhis = ma.where((his <= his_imax))
    ffut = ma.where((fut <= fut_imax))
    obs_avg = obs_imin if (len(obs[fobs])==0) | (obs_imin == obs_imax) else obs[fobs].mean()
    his_avg = his_imin if (len(his[fhis])==0) | (his_imin == his_imax) else his[fhis].mean()
    ratio_avg = obs_avg - his_avg
    
    his[fhis] = his[fhis] + ratio_avg
    fut[ffut] = fut[ffut] + ratio_avg
    return (his, fut)


def apply_bc(obsdata, hisdata, futdata, mode, ddthres=0.0996, max_ratio=5, step=1):
    #print("Mode: ", mode)
    obs = ma.fix_invalid(obsdata, fill_value = -9999)
    his = ma.fix_invalid(hisdata, fill_value = -9999)
    fut = ma.fix_invalid(futdata, fill_value = -9999)

    # Check for data
    if (((~his.mask).sum()==0) | ((~fut.mask).sum()==0) | ((~obs.mask).sum()==0)):
        print('nodata')
        #print((~his.mask).sum()==0, (~fut.mask).sum()==0)
        return (np.nan, np.nan, np.nan, np.nan)
        return (hisdata, futdata)

    bins = np.arange(0, 100+step, step)
    
    if (mode == 'prec'):
        # Apply dry threshold        
        obs = ma.where(obs < ddthres, 0, obs)
        
        # Fix lowest quantile if dealing with precip
        pmin_idx = np.argmin(~(get_pct(obs, bins)>0))
        pmin = bins[pmin_idx]
        pmin_zero = get_pct(his, pmin)
        
        bchis = ma.where(his <= pmin_zero, 0, his)
        bcfut = ma.where(fut <= pmin_zero, 0, fut)
        
    else:
        pmin = 0
        bchis = his.copy()
        bcfut = fut.copy()
        
    ratio_avg = np.zeros(len(bins))
    steps = range(int(pmin*100), int((100+step)*100), int(step*100))

    for i in [x/100 for x in steps]:        
        idx = int(i/step)
        [obs_imin, obs_imax] = get_pctmm(obs, i)
        [his_imin, his_imax] = get_pctmm(his, i)
        [fut_imin, fut_imax] = get_pctmm(fut, i)
                
        if (i == pmin):
            fobs = ma.where((obs <= obs_imax))
            fhis = ma.where((his <= his_imax))
            ffut = ma.where((fut <= fut_imax))
            
        elif(i == 100):
            fobs = ma.where(obs > obs_imin)
            fhis = ma.where(his > his_imin)
            ffut = ma.where(fut > fut_imin)

        else:
            fobs = ma.where((obs > obs_imin) & (obs <= obs_imax))
            fhis = ma.where((his > his_imin) & (his <= his_imax))
            ffut = ma.where((fut > fut_imin) & (fut <= fut_imax))
            
        obs_avg = obs_imin if (len(obs[fobs])==0) | (obs_imin == obs_imax) else obs[fobs].mean()
        his_avg = his_imin if (len(his[fhis])==0) | (his_imin == his_imax) else his[fhis].mean()

        if (mode == 'prec'):
            if ((his_avg == 0) & (obs_avg <= ddthres)):                
                ratio_avg[idx-1] = 1
            elif (his_avg == 0):
                ratio_avg[idx-1] = obs_avg / ddthres
            else:
                ratio_avg[idx-1] = obs_avg / his_avg

            ratio_avg[ratio_avg > max_ratio] = max_ratio
            
            bchis[fhis] = his[fhis] * ratio_avg[idx-1]
            bcfut[ffut] = fut[ffut] * ratio_avg[idx-1]
        elif ((mode == 'V10') | (mode == 'U10')):
            if (his_avg == 0):
                if (obs_avg <= ddthres) & (obs_avg > 0):
                    ratio_avg[idx-1] = 1
                elif (obs_avg >= -1*ddthres) & (obs_avg < 0):
                    ratio_avg[idx-1] = -1
                else:
                    ratio_avg[idx-1] = obs_avg / ddthres
                    
            ratio_avg[ratio_avg > max_ratio] = max_ratio
            
            bchis[fhis] = his[fhis] * ratio_avg[idx-1]
            bcfut[ffut] = fut[ffut] * ratio_avg[idx-1]

        else:
            ratio_avg[i-1] = obs_avg - his_avg
            bchis[fhis] = his[fhis] + ratio_avg[idx-1]
            bcfut[ffut] = fut[ffut] + ratio_avg[idx-1]

    if (mode == 'prec'): 
        bchis = ma.where(bchis < ddthres, 0, bchis)
        bcfut = ma.where(bcfut < ddthres, 0, bcfut)

    bchis = bchis.data
    bcfut = bcfut.data
    
    bchis = np.where(bchis == -9999, np.nan, bchis)
    bcfut = np.where(bcfut == -9999, np.nan, bcfut)

    #(bchis, bcfut) = apply_coldsnap_bc(obs, bchis, bcfut)
    return(bchis, bcfut, pmin_zero, pmin)



# ddthres: minimum precip dry threshold. Values below this is treated as 0
def calc_ratio(obsD, hisD, obsQ, hisQ, minQ, mode, ddthres=0.0996, max_ratio=5):

    # Loop through each bin and calculate mean for each bin
    obsAvg = obsQ.copy() * 0
    hisAvg = hisQ.copy() * 0
    ratio = hisQ.copy() * 0

    # Calculate averages for each bin
    for i in range(1,len(obsQ['quantile'])):
        print(i)
        obsAvg[i,:,:] = obsD.where((obsD > obsQ[i-1,:,:]) & (obsD <= obsQ[i,:,:]) & (i >= minQ[:,:])).mean(dim='time').fillna(0)
        hisAvg[i,:,:] = hisD.where((hisD > hisQ[i-1,:,:]) & (hisD <= hisQ[i,:,:]) & (i >= minQ[:,:])).mean(dim='time').fillna(0)
    
    # Calculate ratio from the obs/his means
    if mode == 'prec':
        hisAvg = hisAvg.where(hisAvg > 0, ddthres)        # Deal with small divisions in precip
        obsAvg = obsAvg.where(obsAvg > ddthres, ddthres)
        ratio = obsAvg / hisAvg
        ratio = ratio.where(ratio < max_ratio, max_ratio) # Cap maximum ratio
    else:
        ratio = obsAvg - hisAvg

    # Fix ratios with null data
    na = obsD.isnull().sum(dim='time')
    na = xr.where(na>0, np.nan, 1)
    na,_ = xr.broadcast(na, ratio)
    na = na.transpose('quantile', 'x', 'y')
    ratio = ratio * na
    
    return ratio



# Apply bias-correction to netcdf dataset
def apply_bc2daily(obsdata, simdata, mode, st_prd, ed_prd, ddthres=0.0996, max_ratio=5, step=1):
    obsD = obsdata.sel(time=slice(st_prd, ed_prd))
    simD = simdata.resample(time='D').sum()
    print('Get Daily Data')
    if len(simD.time)*24 != len(simdata.time):
        simD = simD.sel(time = simD.time[~((simD.time.dt.month == 2) & (simD.time.dt.day == 29))])
    hisD = simD.sel(time=slice(st_prd, ed_prd))

    # Calculate percentile bins
    p = [x/100 for x in range(0, int((100+step)*100), int(step*100))]
    p = np.asarray(p, dtype=float)
    n = len(obsD.time)
    p = (p-50)*n/(n-1) + 50
    p = np.clip(p, 0, 100) / 100
    
    obsQ = obsD.quantile(p, dim='time')
    hisQ = hisD.quantile(p, dim='time')
    minQ = obsQ.where(obsQ>0).fillna(99999).argmin(dim='quantile') # Minimum non-zero bin number

    print('Calculate Ratio')
    ratio = calc_ratio(obsD, hisD, obsQ, hisQ, minQ, mode, ddthres, max_ratio)
    print(ratio)

    bc = simdata.copy()
    scaleD = simD.copy() * 0
    # Loop through each grid point and calculate the scaling to apply for each bin of that grid
    for (i,j) in [(x,y) for x in range(len(simD.x)) for y in range(len(simD.y))]:
        print(i, j)
        for k in range(minQ[i,j].values, len(p)):
            imin = hisQ[k-1, i, j]
            imax = hisQ[k, i, j]
            if k == minQ[i,j].values:
                fidx = simD[:,i,j] <= imax
            elif (k == 100):
                fidx = simD[:,i,j] > imin
            else:
                fidx = (simD[:,i,j] > imin) & (simD[:,i,j] <= imax)
            scaleD[fidx, i,j] = ratio[k, i, j]

    # Expand scale to hourly
    scaleH = scaleD.reindex({'time':simdata.time})
    scaleH = scaleH.ffill('time')
    if mode == 'prec':
        bc = bc * scaleH
        print('Applying minimum threshold')
        bc = bc.where(bc > ddthres, 0)
        print(bc[:,20,20])
        
    else:
        bc = bc + scaleH
        
    return (bc, ratio)
