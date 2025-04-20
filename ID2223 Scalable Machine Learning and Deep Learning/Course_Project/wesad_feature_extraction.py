# The functions here are modified from https://github.com/dzisandy/

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy import signal


def msrs(data, name, f):
    '''
    Returns dictionary of mean, std, dynamic range and slope
    of data, using sliding window with window shift of 0.25 sec
    and window size of 60 sec.
    
    Arguments:
    data: 1D array
    name: name of data used in dictionary keys
    f: sampling frequency of data
    '''
    
    i_025 = int(f*0.25)
    i_60 = int(f*60)
    i = 0
    start = 0
    end = i_60
    
    mean = np.empty(int((len(data) - i_60)/i_025))
    std = np.empty(int((len(data) - i_60)/i_025))
    dynamic_range = np.empty(int((len(data) - i_60)/i_025))
    slope = np.empty(int((len(data) - i_60)/i_025))
    
    while i*i_025 + i_60 < int(len(data)):
        mean[i] = data[start:end+1].mean()
        std[i] = data[start:end+1].std()
        dynamic_range[i] = 20*np.log10(data[start:end+1].max()/data[start:end+1].min())
        slope[i] = (float(data[end]) - float(data[start]))/i_60
        i += 1
        start = i*i_025
        end = i*i_025 + i_60
        
    return {
        'mean_{}'.format(name): mean,
        'std_{}'.format(name): std,
        'dynamic_range_{}'.format(name): dynamic_range,
        'slope_{}'.format(name):slope
    }


def decompose_eda(eda):
        '''Returns lists of SCR and SCL data'''
        
        scr_list  = []
        b,a = signal.butter(4,0.5/2)
        gsr_filt = signal.filtfilt(b,a,eda,axis=0)
        b,a = signal.butter(4,0.5/2,'highpass')
        scr = signal.filtfilt(b,a,gsr_filt,axis=0)
        scl = [float(x-y) for x,y in zip(gsr_filt,scr)]
        
        for i in range(len(scr)):
            scr_list.append(scr[i][0])
            
        return scr_list,scl
    
    
def scrl(scr, scl, f):
    '''
    Returns dictionary of mean, std and number of peaks
    in SCR and SCL data, using sliding window with
    window shift of 0.25 sec and window size of 60 sec.
    
    Arguments:
    scr: scr series
    scl: scl series
    f: frequency
    '''
    
    scr = np.array(scr)
    scl = np.array(scl)
    i_025 = int(f*0.25)
    i_60 = int(f*60)
    i = 0
    start = 0
    end = i_60
    mean_l = np.empty(int((len(scr) - i_60)/i_025))
    mean_r = np.empty(int((len(scr) - i_60)/i_025))
    std_l = np.empty(int((len(scr) - i_60)/i_025))
    std_r = np.empty(int((len(scr) - i_60)/i_025))
    peaks = np.empty(int((len(scr) - i_60)/i_025))
    peak = np.empty(int((len(scr) - i_60)/i_025))
    out = {}
    
    while i*i_025 + i_60 < int(len(scr)):
        mean_r[i] = scr[start:end+1].mean()
        std_r[i] = scr[start:end+1].std()
        mean_l[i] = scl[start:end+1].mean()
        std_l[i] = scl[start:end+1].std()
        peaks[i] = len(signal.find_peaks(scr[start:end+1],height = 0 ,distance=5)[0])
        i += 1
        start = i*i_025
        end = i*i_025 + i_60
        
    return {
        'mean_r': mean_r,
        'mean_l': mean_l,
        'std_r':std_r,
        'std_l':std_l,
        'peaks':peaks
    }


def emg(data, f):
    '''
    Returns dictionary of mean, std and dynamic range
    of EMG data, using sliding window with
    window shift of 0.25 sec and window size of 60 sec.
    
    Arguments:
    data: EMG data series
    f: frequency
    '''
    
    i_025 = int(f*0.25)
    i_5 = int(f*5)
    i = 0
    start = 0
    end = i_5
    dynamic_range = np.empty(int((len(data) - i_5)/i_025))
    mean = np.empty(int((len(data) - i_5)/i_025))
    std = np.empty(int((len(data) - i_5)/i_025))
    
    while i*i_025 + i_5 < int(len(data)):
        mean[i] = data[start:end+1].mean()
        std[i] = data[start:end+1].std()
        dynamic_range[i] = 20*np.log(abs(data[start:end+1].max())/abs(data[start:end+1].min()))
        i += 1
        start = i*i_025
        end = i*i_025 + i_5
        
    return {
        'mean': mean, 
        'std':std, 
        'dynamic_range': dynamic_range
    }


def intgr(data, timestep):
    '''
    Returns the absolute integral XXXXX
    
    '''
    
    return (data[:len(data) - 1].sum() + data[1:].sum())/2*timestep 


def ACC_features(data,window_size,num_0_25_sec,timestep_data, label ):
    '''
    Returns a dictionary of XXXXX
    
    '''
    
    ACC_X_mean = []
    ACC_Y_mean = []
    ACC_Z_mean = []
    ACC_X_std = []
    ACC_Y_std = []
    ACC_Z_std = []
    ACC_X_max = []
    ACC_Y_max = []
    ACC_Z_max = []
    ACC_3D_mean =[]
    ACC_3D_std =[]
    ACC_X_iabs = [] 
    ACC_Y_iabs = [] 
    ACC_Z_iabs = [] 
    ACC_3D_iabs =[]
    
    for i in range(window_size,len(data), num_0_25_sec):
        ACC_X_mean.append( data[:,0][i - window_size:i].mean())
        ACC_X_std.append(data[:,0][i - window_size:i].std())
        ACC_X_max.append(np.fabs(data[:,0][i - window_size:i]).max())
        ACC_X_iabs.append(intgr(np.fabs(data[:,0][i - window_size:i]),timestep_data))
        ACC_Y_mean.append(data[:,1][i - window_size:i].mean())
        ACC_Y_std.append(data[:,1][i - window_size:i].std())
        ACC_Y_max.append(np.fabs(data[:,1][i - window_size:i]).max())
        ACC_Y_iabs.append(intgr(np.fabs(data[:,1][i - window_size:i]),timestep_data))
        ACC_Z_mean.append(data[:,2][i - window_size:i].mean())
        ACC_Z_std.append(data[:,2][i - window_size:i].std())
        ACC_Z_max.append(np.fabs(data[:,2][i - window_size:i]).max())
        ACC_Z_iabs.append(intgr(np.fabs(data[:,2][i - window_size:i]),timestep_data))
        ACC_3D = np.sqrt(data[:,0][i - window_size:i]**2 + data[:,1][i - window_size:i]**2 +data[:,2][i - window_size:i]**2)
        ACC_3D_mean.append(ACC_3D.mean())
        ACC_3D_std.append( ACC_3D.std())
        ACC_3D_iabs.append(intgr(ACC_3D,timestep_data))
        
    return {
        'ACC_X_mean' + label : ACC_X_mean,
        'ACC_Y_mean' + label: ACC_Y_mean,
        'ACC_Z_mean' + label: ACC_Z_mean,
        'ACC_X_std' + label: ACC_X_std,
        'ACC_Y_std' + label: ACC_Y_std,
        'ACC_Z_std' + label: ACC_Z_std,
        'ACC_X_max' + label: ACC_X_max,
        'ACC_Y_max' + label: ACC_Y_max,
        'ACC_Z_max' + label: ACC_Z_max,
        'ACC_X_iabs' + label: ACC_X_iabs,
        'ACC_Y_iabs' + label: ACC_Y_iabs,
        'ACC_Z_iabs' + label: ACC_Z_iabs,
        'ACC_3D_mean' + label: ACC_3D_mean,
        'ACC_3D_std' + label: ACC_3D_std,
        'ACC_3D_iabs' + label: ACC_3D_iabs
    }
        


def f_fr_n(freq, max_freq, l ):
    '''
    
    '''
    if freq < max_freq:
        return int(freq * l/max_freq)
    else:
        return l - 1
    
def detect_peaks_ecg(data,window_size,num_0_25_sec,timestep_data,distance,label): #150 - respiban
    '''
    
    '''
    HR_mean =[]
    HR_std = []
    HR_max = []
    HR_min = []
    N_HRV_50 = []
    P_HRV_50 = []
    rmssd = []
    rr_mean = []
    rr_std = []
    ULF =[]
    HF = []
    LF = []
    UHF = []
    rate_L_H = [] 
    
    for i in range(window_size,len(data), num_0_25_sec):
        f_p = find_peaks(data[i - window_size:i ], height = 0.4, distance = distance)
        #time features
        f_p_diff = np.diff(f_p[0]) * timestep_data
        
        # heart rate mean std min max 
        HR_mean.append((60/f_p_diff).mean())  
        HR_std.append((60/f_p_diff).std()) 
        HR_max.append((60/f_p_diff).max())
        HR_min.append((60/f_p_diff).max())
        #NN50
        #pNN50
        NN50 = sum(np.abs(np.diff(f_p_diff)) > 0.050)
        N_HRV_50.append(NN50)
        P_HRV_50.append(NN50/len(f_p_diff))
        #rr_features
        rmssd.append(np.sqrt(np.mean(np.square(np.diff(f_p_diff)))))
        rr_mean.append(f_p_diff.mean())
        rr_std.append(f_p_diff.std())
        # freq features
        f_p_diff_fft = savgol_filter(np.diff(f_p_diff), 5,2)
        
        T = window_size * timestep_data
        k = np.arange(len(f_p_diff_fft))
        freqs = k/T
        m = freqs.max()/2
        l = int(len(freqs)/2)
        ffts = abs(np.fft.fft(f_p_diff_fft)*np.hamming(len(k)))**2
        ULF.append(sum( ffts[ f_fr_n(0.01,m,l):f_fr_n(0.04,m,l) ] ) )
        HF.append(sum( ffts[ f_fr_n(0.15,m,l):f_fr_n(0.4,m,l) ] ) )
        LF.append(sum( ffts[ f_fr_n(0.04,m,l):f_fr_n(0.15,m,l) ] ) )
        UHF.append(sum( ffts[ f_fr_n(0.4,m,l):f_fr_n(1,m,l) ] ) )
        
        rate_L_H.append(LF[-1]/HF[-1])
        
    return {
        'HR_mean' + label: np.array(HR_mean),
        'HR_std' + label: np.array(HR_std),
        'HR_max'+ label: np.array(HR_max),
        'HR_min'+ label : np.array(HR_min),
        'NN50' + label: np.array(N_HRV_50),
        'pNN50' + label: np.array(P_HRV_50),
        'rmssd' + label: np.array(rmssd),
        'rr_mean' + label: np.array(rr_mean),
        'rr_std' + label: np.array(rr_std),
        'ULF' + label: np.array(ULF),
        'HF'+ label:np.array(HF),
        'LF'+ label:np.array(LF),
        'UHF'+ label:np.array(UHF),
        'rate_L_H'+ label:np.array(rate_L_H)
    }
        
        
    

def f_fr_n(freq, max_freq, l ):
    '''
    
    '''
    if freq < max_freq:
        return int(freq * l/max_freq)
    else:
        return l - 1
    
def detect_peaks_ecg(data,window_size,num_0_25_sec,timestep_data,distance,label): #150 - respiban
    '''
    
    '''
    HR_mean =[]
    HR_std = []
    HR_max = []
    HR_min = []
    N_HRV_50 = []
    P_HRV_50 = []
    rmssd = []
    rr_mean = []
    rr_std = []
    ULF =[]
    HF = []
    LF = []
    UHF = []
    rate_L_H = [] 
    
    for i in range(window_size,len(data), num_0_25_sec):
        f_p = find_peaks(data[i - window_size:i ], height = 0.4, distance = distance)
        #time features
        f_p_diff = np.diff(f_p[0]) * timestep_data
        
        # heart rate mean std min max 
        HR_mean.append((60/f_p_diff).mean())  
        HR_std.append((60/f_p_diff).std()) 
        HR_max.append((60/f_p_diff).max())
        HR_min.append((60/f_p_diff).max())
        #NN50
        #pNN50
        NN50 = sum(np.abs(np.diff(f_p_diff)) > 0.050)
        N_HRV_50.append(NN50)
        P_HRV_50.append(NN50/len(f_p_diff))
        #rr_features
        rmssd.append(np.sqrt(np.mean(np.square(np.diff(f_p_diff)))))
        rr_mean.append(f_p_diff.mean())
        rr_std.append(f_p_diff.std())
        # freq features
        f_p_diff_fft = savgol_filter(np.diff(f_p_diff), 5,2)
        
        T = window_size * timestep_data
        k = np.arange(len(f_p_diff_fft))
        freqs = k/T
        m = freqs.max()/2
        l = int(len(freqs)/2)
        ffts = abs(np.fft.fft(f_p_diff_fft)*np.hamming(len(k)))**2
        ULF.append(sum( ffts[ f_fr_n(0.01,m,l):f_fr_n(0.04,m,l) ] ) )
        HF.append(sum( ffts[ f_fr_n(0.15,m,l):f_fr_n(0.4,m,l) ] ) )
        LF.append(sum( ffts[ f_fr_n(0.04,m,l):f_fr_n(0.15,m,l) ] ) )
        UHF.append(sum( ffts[ f_fr_n(0.4,m,l):f_fr_n(1,m,l) ] ) )
        
        rate_L_H.append(LF[-1]/HF[-1])
        
    return {
        'HR_mean' + label: np.array(HR_mean),
        'HR_std' + label: np.array(HR_std),
        'HR_max'+ label: np.array(HR_max),
        'HR_min'+ label : np.array(HR_min),
        'NN50' + label: np.array(N_HRV_50),
        'pNN50' + label: np.array(P_HRV_50),
        'rmssd' + label: np.array(rmssd),
        'rr_mean' + label: np.array(rr_mean),
        'rr_std' + label: np.array(rr_std),
        'ULF' + label: np.array(ULF),
        'HF'+ label:np.array(HF),
        'LF'+ label:np.array(LF),
        'UHF'+ label:np.array(UHF),
        'rate_L_H'+ label:np.array(rate_L_H)
    }
        
        
def make_target(data_new):
    #TODO: what does this do? Why 175?
    target = []
    for i in range(175, len(data_new[b'label']),175 ):
        target.append(int(data_new[b'label'][i - 175:i].mean()))
    return np.array(target)



def data_collection(data_new):
    numb_of_measures_4_HZ = data_new[b'signal'][b'wrist'][b'EDA'].shape[0]
    experiment_time = numb_of_measures_4_HZ * 0.25
       
    # BVP_wrist
    print('Processing BVP')
    BVP = data_new[b'signal'][b'wrist'][b'BVP']
    BVP = BVP.reshape((BVP.shape[0],))
    window_size_ts = 60
    number_o_in_0_25_sec_wr = int(data_new[b'signal'][b'wrist'][b'BVP'].shape[0]/data_new[b'signal'][b'wrist'][b'EDA'].shape[0])
    window_size_o_wr = 4 * window_size_ts * number_o_in_0_25_sec_wr
    timestep_wr = numb_of_measures_4_HZ * 0.25/data_new[b'signal'][b'wrist'][b'BVP'].shape[0]
    BVP_feature = pd.DataFrame(detect_peaks_ecg(BVP, window_size_o_wr ,number_o_in_0_25_sec_wr,timestep_wr, distance = 25, label = '_BVP'))
    
    #ACC_wrist
    print('Processing ACC')
    ACC = data_new[b'signal'][b'wrist'][b'ACC']
    window_size_ts = 5
    number_o_in_0_25_sec_wr = int(data_new[b'signal'][b'wrist'][b'ACC'].shape[0]/data_new[b'signal'][b'wrist'][b'EDA'].shape[0])
    window_size_o_wr = 4 * window_size_ts * number_o_in_0_25_sec_wr
    timestep_wr = numb_of_measures_4_HZ * 0.25/data_new[b'signal'][b'wrist'][b'ACC'].shape[0]

    ACC_feature = pd.DataFrame(ACC_features(ACC, window_size_o_wr ,number_o_in_0_25_sec_wr,timestep_wr, label = '_wrist'))
    
    #TEMP_WRIST
    print('Processing Temperature')
    Temp_wrist = pd.DataFrame(msrs(data_new[b'signal'][b'wrist'][b'TEMP'],'Temp_wrist',4))
    
    #EDA_wrist
    print('Processing EDA')
    EDA_wrist = pd.DataFrame(msrs(data_new[b'signal'][b'wrist'][b'EDA'],'EDA_wrist',4))
    
    #SCRL_wrist
    print('Processing SCR and SCL')
    scr_wrist, scl_wrist = decompose_eda(data_new[b'signal'][b'wrist'][b'EDA'])
    SCRL_wrist = pd.DataFrame(scrl(scr_wrist,scl_wrist,4))
    data_wrist = pd.concat([ACC_feature[:EDA_wrist.shape[0]],EDA_wrist,Temp_wrist,SCRL_wrist,BVP_feature],axis = 1)

    return data_wrist