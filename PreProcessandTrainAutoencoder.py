

import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.dates as dt
import numpy as np
import scipy.signal as sgnl
import scipy as sp
import seaborn as sns
import pandas as pd
import scipy.signal as scp  
import pickle
from scipy.signal import butter, filtfilt

import sklearn
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.utils import register_keras_serializable
from sklearn.model_selection import train_test_split
import keras

#%%
plt.rcParams['figure.dpi'] = 300


#%%

#we build a dictionnary where each flight file is linked with its flight number
#the other variables are set manually (in order to align starting and ending time of gps / pressure records and to avoid ascent and descent of the balloons)
#flight number i => data_full[i] 


suffixes = {
    "01": {'key' : "TTL5", "align_temps" : 419, "tf_ascent" : 180, "t_end_p" : 29, "t_end_gps" : 4},
    "02": {'key' : "STR1", "align_temps" : 2369, "tf_ascent" :210, "t_end_p" : 3800+9, "t_end_gps" : 0},
    "03": {'key' : "TTL4", "align_temps" : 3809+52, "tf_ascent" :180, "t_end_p" :31-2, "t_end_gps" : 1},
    "04": {'key' : "STR2", "align_temps" : 59+120, "tf_ascent" :420, "t_end_p" : 3800+9, "t_end_gps" : 154},
    "05": {'key' : "TTL3", "align_temps" : 329, "tf_ascent" :180, "t_end_p" :5000+9, "t_end_gps" : 189},
    "06": {'key' : "TTL5", "align_temps" : 299, "tf_ascent" :180, "t_end_p" : 29, "t_end_gps" : 1},
    "07": {'key' : "TTL4", "align_temps" : 269+60, "tf_ascent" :150, "t_end_p" :23000+9, "t_end_gps" : 318},
    "08": {'key' : "STR1", "align_temps" : 389+60, "tf_ascent" :210, "t_end_p" :1829, "t_end_gps" : 0},
    "09": {'key' : "STR2", "align_temps" : 269, "tf_ascent" :210, "t_end_p" :29, "t_end_gps" : 0},
    "10": {'key' : "TTL3", "align_temps" : 299+60, "tf_ascent" :150, "t_end_p" :1700+9, "t_end_gps" : 141},
    "11": {'key' : "TTL1", "align_temps" : 269, "tf_ascent" :180, "t_end_p" :29, "t_end_gps" : 1},
    "12": {'key' : "STR4", "align_temps" : 239, "tf_ascent" :270, "t_end_p" :29, "t_end_gps" : 1},
    "13": {'key' : "STR1", "align_temps" : 299, "tf_ascent" :210, "t_end_p" :29, "t_end_gps" : 1},
    "15": {'key' : "TTL4", "align_temps" : 7199, "tf_ascent" :180, "t_end_p" :1500+29, "t_end_gps" : 486},
    "16": {'key' : "TTL5", "align_temps" : 7199, "tf_ascent" :180, "t_end_p" :29, "t_end_gps" : 1},
    "17": {'key' : "TTL3", "align_temps" : 479, "tf_ascent" :270, "t_end_p" :29, "t_end_gps" : 1}
}

STR_flights = ["02","04","09","12","13","14","08"]
TTL_flights = ["03", "05", "07","10", "01", "06", "17", "15", "16", "11"]

data_full = {}
data_P1s = {}
for v, suff in suffixes.items():
    data_full[v] = nc.Dataset(f"C:/PATHtoDATA/ST2_C1_{v}_{suff['key']}_TSEN_v01.nc")
    data_P1s[v] = nc.Dataset(f"C:/PATHtoDATA/ST2_C1_{v}_{suff['key']}_TSEN_P1s_v01.nc")

#%%

#in the result dictionnary, the time series are managed in order to make them adequate (supressing masked values, using meaningful names for variables, etc.)

results = {}
for v, suff in suffixes.items():
    t=suff["align_temps"]
    tend=suff["t_end_p"]
    fin_ascent_gps=suff["tf_ascent"]
    # P1s
    time_P1s = data_P1s[v]['time'][:]
    time_P1s = np.where(time_P1s.mask, 1, time_P1s)
    press = data_P1s[v]['pressure'][:]
    press = np.where(press.mask, 1, press)

    # full
    time_gps = data_full[v]['time'][:]
    alt = data_full[v]['alt'][:]
    temp = data_full[v]['thermistor'][:]
    corrected_temp = data_full[v]['corrected_thermistor'][:]
    u_wind = data_full[v]['u'][:]
    v_wind = data_full[v]['v'][:]
    wind_speed = data_full[v]['ff'][:]
    sza = data_full[v]['sza'][:]
    lat = data_full[v]['lat'][:]
    lon = data_full[v]['lon'][:]

    
    data_masked = alt.mask + wind_speed.mask +temp.mask
    fin_ascent_press=fin_ascent_gps*30
    add = t+fin_ascent_press
    


    tend_gps=suff["t_end_gps"]
    
    last_idx_P=len(time_P1s)-1
    last_idx_gps=len(time_gps)-1
    #here are all of the variables available in the results dictionnary (for now, time_P1s is aligned with press, and all of the other variables are aligned with time_gps)
    results[v] = {
        "time_P1s": time_P1s[add:last_idx_P-tend],
        "press": press[add:last_idx_P-tend],
        "time_gps": time_gps[~data_masked][fin_ascent_gps:last_idx_gps-tend_gps],
        "alt": alt[~data_masked][fin_ascent_gps:last_idx_gps-tend_gps],
        "temp": temp[~data_masked][fin_ascent_gps:last_idx_gps-tend_gps],
        "corrected_temp": corrected_temp[~data_masked][fin_ascent_gps:last_idx_gps-tend_gps],
        "u_wind": u_wind[~data_masked][fin_ascent_gps:last_idx_gps-tend_gps],
        "v_wind": v_wind[~data_masked][fin_ascent_gps:last_idx_gps-tend_gps],
        "wind_speed": wind_speed[~data_masked][fin_ascent_gps:last_idx_gps-tend_gps],
        "sza": sza[~data_masked][fin_ascent_gps:last_idx_gps-tend_gps],
        "lat": lat[~data_masked][fin_ascent_gps:last_idx_gps-tend_gps],
        "lon": lon[~data_masked][fin_ascent_gps:last_idx_gps-tend_gps]
    }


#%%
#this is the starting time of our times array (seconds since 2019-01-01)
start_time = dt.date2num(np.datetime64("2019-01-01"))

#%%
#Here is a debugging cell to verify if times and variables are actually aligned

for v, data in results.items():
    # time_P1s and pressure
    len_p1s = len(data["time_P1s"])
    len_press = len(data["press"])
    check_p1s = (len_p1s == len_press)
    check_start_time = (data["time_P1s"][0] == data["time_gps"][0])
    check_end_time = (data["time_P1s"][-1] == data["time_gps"][-1])

    # other variables and time_gps
    keys_full = ["time_gps","alt","temp","corrected_temp",
                 "u_wind","v_wind","wind_speed","sza","lat","lon"]
    lengths_full = [len(data[k]) for k in keys_full]
    check_full = (len(set(lengths_full)) == 1)

    if check_p1s and check_full :
        print(f"flight {v}: vectors of same length")
    else:
        print(f"flight {v}: issue")
        print(f"   time_P1s={len_p1s}, press={len_press}")
        print("   others:", {k: len(data[k]) for k in keys_full})
      

    if check_start_time : 
        print("starting times aligned")
    else : 
        print("start press : "  , data["time_P1s"][0] )
        print("start gps : ", data["time_gps"][0] )

    if check_end_time : 
        print("ending times aligned" )
    else : 
        print("fin press :" , data["time_P1s"][-1])
        print("fin gps", data["time_gps"][-1])
        print("time gap : " , data["time_P1s"][-1]-data["time_gps"][-1])
        
#%%

import os
import numpy as np
import scipy as sp
import matplotlib.dates as dt
#Running this cell might take a lot of time (15minutes) the first time. But the time conversion file is saved, so the next times this file will be used, it will run in a short amount of time
savefile = "times_converted.npz"
#in this cell, we interpolate our time series to have them at the same rate as our pressure variable
#we also convert the time vectors (seconds since 2019-01-01) to adequate format (meaningful)
if os.path.exists(savefile):
    
    data_time = np.load(savefile, allow_pickle=True)
    for v, data in results.items():
        results[v]["time_P1s_todate"] = data_time[f"{v}_time_P1s_todate"]
        results[v]["time_gps_todate"] = data_time[f"{v}_time_gps_todate"]
        print(v)
        print(np.any(~np.isfinite(data["alt"])))
        print(np.any(~np.isfinite(data["temp"])))
        print(np.any(~np.isfinite(data["u_wind"])))
        print(np.any(~np.isfinite(data["v_wind"])))
        print(np.any(~np.isfinite(data["wind_speed"])))
        
        print(data["time_P1s"].dtype)
        time = data["time_P1s"]
        print(data["time_gps"].dtype)
        print(data["alt"].dtype)
        alt_interp = sp.interpolate.CubicSpline(data["time_gps"], data["alt"])(time)
        print(data["temp"].dtype)
        temp_interp  = sp.interpolate.CubicSpline(data["time_gps"], data["temp"])(time)
        print(data["u_wind"].dtype)
        u_wind_interp = sp.interpolate.CubicSpline(data["time_gps"], data["u_wind"])(time)
        print(data["v_wind"].dtype)
        v_wind_interp = sp.interpolate.CubicSpline(data["time_gps"], data["v_wind"])(time)
        print(data["wind_speed"].dtype)
        wind_speed_interp = sp.interpolate.CubicSpline(data["time_gps"], data["wind_speed"])(time)

        results[v]["alt_interp"] = alt_interp
        results[v]["temp_interp"] = temp_interp
        results[v]["u_wind_interp"] = u_wind_interp
        results[v]["v_wind_interp"] = v_wind_interp
        results[v]["wind_speed_interp"] = wind_speed_interp

else:
    
    for v, data in results.items():
        print(v)
        print(np.any(~np.isfinite(data["alt"])))
        print(np.any(~np.isfinite(data["temp"])))
        print(np.any(~np.isfinite(data["u_wind"])))
        print(np.any(~np.isfinite(data["v_wind"])))
        print(np.any(~np.isfinite(data["wind_speed"])))
        
        print(data["time_P1s"].dtype)
        time = data["time_P1s"]
        print(data["time_gps"].dtype)
        print(data["alt"].dtype)
        alt_interp = sp.interpolate.CubicSpline(data["time_gps"], data["alt"])(time)
        print(data["temp"].dtype)
        temp_interp  = sp.interpolate.CubicSpline(data["time_gps"], data["temp"])(time)
        print(data["u_wind"].dtype)
        u_wind_interp = sp.interpolate.CubicSpline(data["time_gps"], data["u_wind"])(time)
        print(data["v_wind"].dtype)
        v_wind_interp = sp.interpolate.CubicSpline(data["time_gps"], data["v_wind"])(time)
        print(data["wind_speed"].dtype)
        wind_speed_interp = sp.interpolate.CubicSpline(data["time_gps"], data["wind_speed"])(time)

        results[v]["alt_interp"] = alt_interp
        results[v]["temp_interp"] = temp_interp
        results[v]["u_wind_interp"] = u_wind_interp
        results[v]["v_wind_interp"] = v_wind_interp
        results[v]["wind_speed_interp"] = wind_speed_interp

        # to datetime
        time_P1s_todate = start_time + results[v]["time_P1s"] / 86400
        time_gps_todate = start_time + results[v]["time_gps"] / 86400
        time_P1s_todate = dt.num2date(time_P1s_todate)
        time_gps_todate = dt.num2date(time_gps_todate)

        results[v]["time_gps_todate"] = time_gps_todate
        results[v]["time_P1s_todate"] = time_P1s_todate

    # we save the time file
    np.savez(
        savefile,
        **{f"{v}_time_P1s_todate": results[v]["time_P1s_todate"] for v in results},
        **{f"{v}_time_gps_todate": results[v]["time_gps_todate"] for v in results}
    )

#%%
#pressure correction

g=9.81
Rd=287
R=8.314
M=29*10**-3
T0 = 293.15
p0 = 101325

for v, data in results.items() : 
    alt_interp = data["alt_interp"]
    temp_interp = data["temp_interp"]
    h_anomalies_full = alt_interp - np.mean(alt_interp)
    var_full = g*h_anomalies_full/(Rd*temp_interp)
    var_full = np.array(var_full, dtype = float)
    press_adjusted_ = data["press"] *np.exp(var_full)
    results[v]["press_adjusted"] = press_adjusted_


#%%

#this function takes as input a time vector and a pressure vector and returns all of the blocks that do not contain any missing data

def blocks_extraction(time: np.ndarray, press : np.ndarray, wind : np.ndarray, freq_hz=1):
    
    dt = np.diff(time).astype('timedelta64[s]').astype(int)
    standard_gap = int(1 / freq_hz)

    
    trous_indices = np.where(dt > standard_gap)[0]
    nb=0
    blocs = []
    debut = 0

    for idx_trou in trous_indices:
        fin = idx_trou + 1  
        blocs.append({
            "time": time[debut:fin],
            "pressure": press[debut:fin],
            "wind_speed_interp": wind[debut:fin]
        })
        debut = fin
        nb=nb+1

    
    blocs.append({
        "time": time[debut:],
        "pressure": press[debut:],
        "wind_speed_interp": wind[debut:fin]
        
    })
    nb=nb+1
    print(nb)

    return blocs
#%%

for v, data in results.items(): 
    blocs = blocks_extraction(results[v]["time_P1s_todate"], results[v]["press_adjusted"], results[v]["wind_speed_interp"], freq_hz=1)
    results[v]["blocs"]=blocs

    
#%%

#this function takes blocks (return of the previous function) and calculate all of the spectrograms
#user has to set the duration of the spectrograms (duree_segment_sec, in seconds), NFFT and an overlap ratio. Spectrograms are in dB scale. If you want natural scale, just comment the "Pxx=10np.log10(Pxx)" line

def spectro_from_blocks(blocs, fs=1, duree_segment_sec=20*60, NFFT=600, overlap_ratio=0.5):
    Specs = []
    Times = []
    freqs_ref = None
    colonnes_par_segment = None
    noverlap=int(overlap_ratio*NFFT)

    for bloc in blocs:
        press = bloc["pressure"]
        time_vect = bloc["time"]
        sos = butter(4, 0.1, 'lowpass', fs=1, output='sos')
        
        p_filt = scp.sosfiltfilt(sos, press)
        
        if len(press) < NFFT:
            continue

        freqs, bins, Pxx= scp.spectrogram(
            press-p_filt,
            nperseg=NFFT,
            fs=fs,
            noverlap=noverlap,
            mode='psd', 
            scaling="density"
        )

        Pxx=10*np.log10(Pxx)
    
    
        if freqs_ref is None:
            freqs_ref = freqs

        step_sec = (NFFT - noverlap) / fs
        colonnes_par_segment = int(duree_segment_sec / step_sec)
        if colonnes_par_segment == 0:
            continue

        
        for start_col in range(0, Pxx.shape[1] - colonnes_par_segment + 1, colonnes_par_segment):
            seg_Pxx = Pxx[:, start_col:start_col + colonnes_par_segment]

            
            seg_bins_rel = bins[start_col:start_col + colonnes_par_segment]
            seg_bins_abs = np.datetime64(time_vect[0]) + seg_bins_rel.astype('timedelta64[s]')

            Specs.append(seg_Pxx)
            Times.append(seg_bins_abs)

    if len(Specs) == 0:
        return np.empty((0, len(freqs_ref), colonnes_par_segment)), freqs_ref, []

    Specs = np.stack(Specs)

    return Specs, freqs_ref, Times

#%%

#here, we calculate all of the spectrograms for all of our flights, using the same parameters so our spectrograms are constant input shape
for v, data in results.items() : 
    Specs, freqs_ref, Times = spectro_from_blocks(results[v]["blocs"], fs = 1, duree_segment_sec=20*60, NFFT = 100, overlap_ratio=0.9)
    results[v]["Specs"] = Specs
    results[v]["freqs_ref"]=freqs_ref
    results[v]["Times"]=Times

#%%

spectrograms_TTL = []
spectrograms_STR = []
metadata_TTL = []
metadata_STR = []
#here, we just build two datasets : one for ttl flights and one for STR flights. The two metadata datasets are buit this way :
#we store spectrograms in chronological order. For each spectrogram, we add the following informations : id (global id of the spectrogram), flight number, time_start and time_end (starting and ending time of the recording for the spectrogram)
#lat(start and end), lon (start and end) =>position of the baloon at the start/end of the spectrogram
#alt start / end => altitude of the baloon at start/end of the spectrogram
ref_time = np.datetime64("2019-01-01") 

for v, data in results.items():
    specs = data["Specs"] 
    times = data["time_gps"]       # seconds since 2019-01-01
    lats = data["lat"]
    lons = data["lon"]
    bounds = data["Times"]    
    alt = data["alt_interp"]
    wind = data["wind_speed_interp"]
    
    print(np.size(specs[0]))      

    for i, spec in enumerate(specs):
        
        t0 = bounds[i][0]
        t1 = bounds[i][-1]

        if isinstance(t0, np.datetime64):
            t_start = (t0 - ref_time) / np.timedelta64(1, 's')
        elif isinstance(t0, pd.Timestamp):
            t_start = (t0.to_datetime64() - ref_time) / np.timedelta64(1, 's')
        else:  # datetime python
            t_start = (np.datetime64(t0) - ref_time) / np.timedelta64(1, 's')

        if isinstance(t1, np.datetime64):
            t_end = (t1 - ref_time) / np.timedelta64(1, 's')
        elif isinstance(t1, pd.Timestamp):
            t_end = (t1.to_datetime64() - ref_time) / np.timedelta64(1, 's')
        else:  # datetime python
            t_end = (np.datetime64(t1) - ref_time) / np.timedelta64(1, 's')

        # find starting and ending times of spectrograms
        i_start = np.argmin(np.abs(times - t_start))
        i_end   = np.argmin(np.abs(times - t_end))

        lat_start, lon_start = lats[i_start], lons[i_start]
        lat_end, lon_end     = lats[i_end], lons[i_end]
        alt_start, alt_end   = alt[i_start], alt[i_end]
        
        try: 
            first_quart, third_quart =  np.percentile(wind[i_start:i_end],[25,75])#+1.5*np.std(entrop_valid)
        except:
            first_quart= np.nan
            third_quart = np.nan


        
        if v in STR_flights:
            spectrograms_STR.append(spec)
            metadata_STR.append({
                "id": f"{v}_{i}",
                "vol": v,
                "t_start": pd.to_datetime(t_start, unit='s', origin='2019-01-01'),
                "t_end": pd.to_datetime(t_end, unit='s', origin='2019-01-01'),
                "lat_start": lat_start,
                "lon_start": lon_start,
                "lat_end": lat_end,
                "lon_end": lon_end,
                "alt_start": alt_start,
                "alt_end": alt_end
            })
        else:
            spectrograms_TTL.append(spec)
            metadata_TTL.append({
                "id": f"{v}_{i}",
                "vol": v,
                "t_start": pd.to_datetime(t_start, unit='s', origin='2019-01-01'),
                "t_end": pd.to_datetime(t_end, unit='s', origin='2019-01-01'),
                "lat_start": lat_start,
                "lon_start": lon_start,
                "lat_end": lat_end,
                "lon_end": lon_end,
                "alt_start": alt_start,
                "alt_end": alt_end,
                "mean_wind": np.nanmean(wind[i_start:i_end]),
                "first_quart": first_quart,
                "third_quart": third_quart,
                "Gustiness": np.nanstd(wind[i_start:i_end])/np.nanmean(wind[i_start:i_end])
            })

df_meta_STR = pd.DataFrame(metadata_STR)
df_meta_TTL = pd.DataFrame(metadata_TTL)

#these two arrays only contains the spectrograms, these are input for the machine learning part
X_STR = spectrograms_STR
X_TTL = spectrograms_TTL

#%%

print(np.shape(X_TTL))
print(np.shape(X_STR))

#%%
#normalizing our inputs :
X_TTL= np.array(X_TTL).astype(np.float32)[..., np.newaxis]
X_STR=np.array(X_STR).astype(np.float32)[..., np.newaxis]
min_STR, max_STR = X_STR.min(), X_STR.max()
min_TTL, max_TTL =X_TTL.min(), X_TTL.max()

X_TTL=(X_TTL-min_TTL)/(max_TTL-min_TTL)
X_STR=(X_STR-min_STR)/(max_STR-min_STR)



#%%



def build_autoencoder(input_shape):
    H, W = input_shape[0], input_shape[1]

    # dimension calculation
    H_encoded, W_encoded = H, W
    for _ in range(4):
        H_encoded = (H_encoded + 1) // 2
        W_encoded = (W_encoded + 1) // 2

    print((H_encoded))
    print((W_encoded))
    # Encoder
    encoder_inputs = keras.Input(shape=input_shape)

    print(encoder_inputs.shape)
    x = keras.layers.ZeroPadding2D(padding=1)(encoder_inputs)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    print(x.shape)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    print(x.shape)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    print(x.shape)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(4, (3, 3), activation='relu', padding='same')(x)
    print(x.shape)


    
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

    encoder = keras.Model(encoder_inputs, encoded, name="encoder")
    print(encoded.shape)
    # Decoder
    latent_inputs = keras.Input(shape=(encoded.shape[1:]))

    
    print(x.shape)

    x = layers.Conv2D(4, (3, 3), activation='relu', padding='same')(latent_inputs)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    print(x.shape)

    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    

    print(x.shape)

    # Crop final
    x = layers.Cropping2D(((x.shape[1] - H, 0), (x.shape[2] - W, 0)))(x)
   
    print(x.shape)

    decoder_outputs = x
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    # full ae
    autoencoder_inputs = encoder_inputs
    decoded = decoder(encoder(autoencoder_inputs))
    autoencoder = keras.Model(autoencoder_inputs, decoded, name="autoencoder")
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["mse"]
    )

    return autoencoder, encoder, decoder
    

#%%
sos = butter(6, 0.1, 'highpass', fs=1, output='sos')

#%% separate out test part

X_TTL_train, X_TTL_val = train_test_split(X_TTL, test_size = 0.2, random_state = 15)

#%%
df_meta_TTL_use, df_meta_TTL_test = train_test_split(df_meta_TTL, test_size = 0.2, random_state = 15)


#%% Train on TTL flights

ae_TTL, enc_TTL, dec_TTL = build_autoencoder(X_TTL_train.shape[1:])#, latent_dim=16)

hist_TTL = ae_TTL.fit(X_TTL_train, X_TTL_train, epochs=20, batch_size=512, validation_data = (X_TTL_val, X_TTL_val))


#%%
X_TTL_recon = ae_TTL.predict(X_TTL, verbose=0)

#%%

vae= ae_TTL
X_test = X_TTL
cmap="viridis"
exclude_flights=[]#["01","03","05","06","07","10","11","15","16"]
df_meta=df_meta_TTL
X_recon=X_TTL_recon
log_freq=True 
start_idx=0

if X_recon.shape[1:3] != X_test.shape[1:3]:
    from tensorflow.image import resize
    X_recon = resize(X_recon, (X_test.shape[1], X_test.shape[2])).numpy()

if exclude_flights is not None:
    mask_valid = ~df_meta['vol'].isin(exclude_flights)
else:
    mask_valid = np.ones(len(df_meta), dtype=bool)

valid_idx = np.where(mask_valid)[0]
#valid_idx = valid_idx[:2]
counter=0

full_idx = np.arange(X_recon.shape[0])
entrop = np.zeros(full_idx.shape)
var = np.zeros(full_idx.shape)

for idx in full_idx:
    orig = X_test[idx].squeeze()
    recon = X_recon[idx].squeeze()
    diff = (orig - recon)**2
    var[counter] = np.std(orig.reshape(-1,1))
    #entrop[counter] = sp.stats.entropy(orig.reshape(-1,1))
    entrop[counter] = np.nansum(sp.special.kl_div(orig.reshape(-1,),recon.reshape(-1,)))
    counter=counter+1
    #print(counter)
    
# MSE 
mse = np.mean((X_test - X_recon) ** 2, axis=(1, 2, 3))

mse_valid = mse[valid_idx]
entrop_valid = entrop[valid_idx]
var_valid = var[valid_idx]
#%%

thresh =  np.percentile(entrop_valid,99)

print(thresh)


fig = plt.figure(constrained_layout=True, figsize=(5,3))
axd = fig.subplot_mosaic(
   """
   B
   """
   )

numbers= np.arange(len(entrop_valid))
sortedvals= np.sort(entrop_valid)
numbers = numbers#/max(numbers)
axd["B"].plot(sortedvals,numbers,'.', ms=2,color="royalblue")

height = np.where(sortedvals>thresh)[0]

axd["B"].vlines(thresh, 0, numbers[height[0]], color='royalblue', ls=':')
axd["B"].annotate('99th percentile', xy=(8, 15000), xycoords='data',va="center", ha="center",size=12)


axd["B"].set_xlim((sortedvals[0],20))
axd["B"].set_ylim((0,len(sortedvals)+1000))

axd["B"].set_xlabel("KL Divergence",size=12)
axd["B"].set_ylabel("Ordered sample",size=12)

axd["B"].set_title("Removal threshold",size=12)

axd["B"].tick_params(axis='both', labelsize=12)



#%% Remove artifact threshold

thresh =  np.percentile(entrop_valid,99)

sorted_idx = np.argsort(entrop_valid)


worst_idx_valid = valid_idx[sorted_idx]
entrop_valid_sort = entrop_valid[worst_idx_valid]

Idx_thresh = np.where(entrop_valid_sort < thresh)[0]
Idx_under_thresh = worst_idx_valid[Idx_thresh]

plt.plot((entrop_valid[Idx_under_thresh]))

X_TTL_reduced = X_TTL[Idx_under_thresh]

#%% Recreate training and validation data on reduced set

X_TTL_train, X_TTL_val = train_test_split(X_TTL_reduced, test_size = 0.2, random_state=20)



#%% Train on TTL flights

ae2_TTL, enc2_TTL, dec2_TTL = build_autoencoder(X_TTL_train.shape[1:])


#%%
hist_TTL = ae2_TTL.fit(X_TTL_train, X_TTL_train, epochs=50, batch_size=512, validation_data = (X_TTL_val, X_TTL_val))

#%%
X_TTL_recon = ae2_TTL.predict(X_TTL, verbose=0)



#%% Figure 2

vae= ae2_TTL
X_test = X_TTL
cmap="viridis"
exclude_flights=[]#["01","03","05","06","07","10","11","15","16"]
df_meta=df_meta_TTL
X_recon=X_TTL_recon
log_freq=True 
start_idx=0

if X_recon.shape[1:3] != X_test.shape[1:3]:
    from tensorflow.image import resize
    X_recon = resize(X_recon, (X_test.shape[1], X_test.shape[2])).numpy()

if exclude_flights is not None:
    mask_valid = ~df_meta['vol'].isin(exclude_flights)
else:
    mask_valid = np.ones(len(df_meta), dtype=bool)

valid_idx = np.where(mask_valid)[0]

counter=0

full_idx = np.arange(X_recon.shape[0])
entrop = np.zeros(full_idx.shape)
for idx in full_idx:
    orig = X_test[idx].squeeze()
    recon = X_recon[idx].squeeze()
    diff = (orig - recon)**2
    
    entrop[counter] = np.nansum(sp.special.kl_div(orig.reshape(-1,),recon.reshape(-1,)))
    counter=counter+1

    
# MSE 
mse = np.mean((X_test - X_recon) ** 2, axis=(1, 2, 3))

mse_valid = mse[valid_idx]
entrop_valid = entrop[valid_idx]
#%% Figure 3

first, twenty, fifty, sevenfive, ninenine =  np.percentile(entrop_valid,[1, 25, 50, 75, 99])#

fig = plt.figure(constrained_layout=True, figsize=(5,3))
axd = fig.subplot_mosaic(
   """
   B
   """
   )

numbers= np.arange(len(entrop_valid))
sortedvals= np.sort(entrop_valid)
numbers = numbers#/max(numbers)
axd["B"].plot(sortedvals,numbers,'.', ms=2,color="royalblue")
#sns.rugplot(ax=axd["B"],data=entrop_valid)

height = np.where(sortedvals>twenty)[0]
axd["B"].vlines(twenty, 0, numbers[height[0]], color='royalblue', ls=':')

height = np.where(sortedvals>fifty)[0]
axd["B"].vlines(fifty, 0, numbers[height[0]], color='royalblue', ls=':')


height = np.where(sortedvals>sevenfive)[0]
axd["B"].vlines(sevenfive, 0, numbers[height[0]], color='royalblue', ls=':')

height = np.where(sortedvals>ninenine)[0]
axd["B"].vlines(ninenine, 0, numbers[height[0]], color='royalblue', ls=':')



axd["B"].set_xlim((sortedvals[0],5))
axd["B"].set_ylim((0,len(sortedvals)+1000))

axd["B"].set_xlabel("KL Divergence",size=12)
axd["B"].set_ylabel("Ordered sample",size=12)

axd["B"].set_title("Anomaly distribution",size=12)

axd["B"].tick_params(axis='both', labelsize=12)

