import pandas as pd

def identify_sedentary(signal, t_sedentary):
    # impute missing values by 0
    signal = signal.fillna(0)
    
    # identify phases where 7 consecutive values are = 0
    sedentary = signal.rolling(t_sedentary).sum() == 0
    
    return sedentary

def resting_heart_rate(heart, steps, t_sedentary):
    # set rhr to NaN 
    rhr = pd.Series(index=heart.index, dtype=float)
    
    # set resting heart rate rhh to the average of "heart" when "sedentary" is True
    rhr[identify_sedentary(steps, t_sedentary)] = heart[identify_sedentary(steps, t_sedentary)].rolling(t_sedentary).mean()
    
    # impute missing values by previous value
    rhr = rhr.fillna(method='ffill')
    
    # calculate daily average and resample to 1 minute
    rhr = rhr.resample('1D').mean().resample('1T').ffill()
    
    return rhr

def phrr(heart, steps, age, t_sedentary=7, lower=0.0):    
    # calculate the resting heart rate
    rhr = resting_heart_rate(heart, steps, t_sedentary)
    
    # calculate the predicted heart rate reserve
    pHRR = (heart - rhr) / (220 - age - rhr)
    
    # clip
    pHRR = pHRR.clip(lower)-lower
    return pHRR