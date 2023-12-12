import pandas as pd
import numpy as np
import os

from lxml import etree
from ..base import BaseRaw
from pyActigraphy.light import LightRecording
import matplotlib.pyplot as plt


class RawFTB_SPHYNCS(BaseRaw):

    """Raw object from pandas.DataFrame (recorded by Fitbit in SPHYNCS format)

    Parameters
    ----------
    df: pandas.DataFrame
        Preloaded data in pandas dataframe.
    start_time: datetime-like, optional
        Read data from this time.
        Default is None.
    period: str, optional
        Length of the read data.
        Cf. #timeseries-offset-aliases in
        <https://pandas.pydata.org/pandas-docs/stable/timeseries.html>.
        Default is None (i.e all the data).
    data_dtype: dtype, optional
        The dtype of the raw data.
        Default is np.int.
    light_dtype: dtype, optional
        The dtype of the raw light data. Important: Fitbit does not record light! The heart rate is taken as surrogate.
        Default is np.float.
    """

    def __init__(
        self,
        raw_data,
        activity,
        start_time=None,
        period=None
    ):

        if type(raw_data) != pd.DataFrame:
            raise TypeError('Input <raw_data> must be a pandas.DataFrame.')

        # read csv file
        raw_data = self.__reading_and_parsing_file(raw_data)
        raw_data = self.__preprocess_activity_data(raw_data)

        # extract informations from the header
        name = self.__extract_ftb_name(raw_data)
        start = self.__extract_ftb_start_time(raw_data)
        frequency = self.__extract_frequency(raw_data)
        frequency_light = frequency # same frequency as "activity"

        activity_data = self.__extract_activity_data(raw_data, activity)
        light_data = self.__extract_light_data(raw_data)
        
        # index the motion time serie
        index_data = pd.Series(
            data=activity_data.values,
            index=pd.date_range(
                start=start,
                periods=len(activity_data),
                freq=frequency
            ),
            dtype=float
        )
            
        # index the light time serie
        if light_data is not None:
            index_light = pd.Series(
                data=light_data.values,
                index=pd.date_range(
                    start=start,
                    periods=len(light_data),
                    freq=frequency_light
                    ),
                dtype=float
        )

        if start_time is not None:
            start_time = pd.to_datetime(start_time)
        else:
            start_time = start

        if period is not None:
            period = pd.Timedelta(period)
            stop_time = start_time+period
        else:
            stop_time = index_data.index[-1]
            period = stop_time - start_time
    
        # Create a new index covering the entire desired period
        full_index = pd.date_range(start=start_time, end=stop_time, freq=frequency)
        index_data = index_data.reindex(full_index, fill_value=pd.NA)
        index_light = index_light.reindex(full_index, fill_value=pd.NA)

        # call __init__ function of the base class
        super().__init__(
            fpath=raw_data,
            name=name,
            uuid=None, # not available for Fitbit
            format='FTB',
            axial_mode=None,
            start_time=start_time,
            period=period,
            frequency=frequency,
            data=index_data,
            light=LightRecording(
                name=name,
                uuid=None,
                data=index_light.to_frame(name='whitelight'),
                frequency=frequency_light
            ) if index_light is not None else None
        )

    @property
    def white_light(self):
        r"""Value of the light intensity in µw/cm²."""
        if self.light is None:
            return None
        else:
            return self.light.get_channel("whitelight")

    def __reading_and_parsing_file(self, input_fname):
        return input_fname #TODO: replace by function to read CSV file

    def __extract_ftb_name(self, df):
        """ Extract name from the raw dataframe"""
        return df.name #TODO: replace such that the name is taken from the path or something...

    def __extract_ftb_start_time(self, df):
        """ Extract start time from the raw dataframe"""
        return df.iloc[0].name #TODO: replace by general function. Make sure it returns a datetime!

    def __extract_frequency(self, df):
        """ Extract frequency from the raw dataframe"""
        return pd.Timedelta(1, unit=pd.infer_freq(df.index[:3])) #TODO: replace by general function. Make sure the input is datetime!

    def __extract_activity_data(self, df, activity):
        """ Extract activity measurement from the raw dataframe"""
        try: 
            activity_data = df[activity]
        except:
            raise ValueError('"activity" must be "calories", "steps", or "heart"')
        return activity_data

    def __extract_light_data(self, df):
        """ Extract heart rate measurement from the raw dataframe as surrogate 'light' """
        try: 
            light_data = df['heart']
        except:
            raise ValueError('"activity" must be "calories" or "steps"')
        return light_data
    
    def __preprocess_activity_data(self, df):
        # clean the steps
        df = self.__preprocess_steps_data(df)
        # normalize the calories
        df = self.__preprocess_calories_data(df)
        # normalize the heart
        df = self.__preprocess_heart_data(df)
        return df
    
    def __preprocess_steps_data(self, df):        
        # get indeces of NaN in 'steps' where 'heart' is not NaN
        to_zero = (df['steps'].isna() & df['heart'].notna()).values
        # set these entries for 'steps' to 0
        steps_idx = df.columns.get_loc("steps")
        df.iloc[to_zero, steps_idx] = 0
        return df
    
    def __preprocess_calories_data(self, df, var='calories'):
        """ Normalize the calories in segments."""
        # set zeros to nan
        df= self.__preprocess_zeros_to_nan(df, 'calories') 
        # get baseline
        baseline = self.__get_baseline(df, var)
        # subtract baseline from data    
        df[var] = df[var] - baseline.values
        # normalize by min-max scaling
        df[var] = self.__preprocess_minmax(df, var)
        return df   
    
    def __preprocess_heart_data(self, df):      
        # set zero values to NaN
        df= self.__preprocess_zeros_to_nan(df, 'heart') 
        # apply min-max scaling
        df['heart'] = self.__preprocess_minmax(df, 'heart')
        return df
    
    def __get_baseline(self, df, var):
        """ estimate the baseline. """
        # get daily baseline
        daily_offset = df.groupby(df.index.date)[var].min()
        baseline = pd.Series(df.index.date).map(daily_offset)
        return baseline
        
    def __preprocess_zeros_to_nan(self, df, var):
        """ set zero values to nan """
        # get index of zero values
        to_nan = (df[var] <= 0)  
        # get column index of variable
        col_idx = df.columns.get_loc(var)
        # set zero values to NaN
        df.iloc[to_nan, col_idx] = np.nan
        return df
        
    def __preprocess_minmax(self, df, var):      
        """ Normalize the data by subtracting the minimum and dividng by the maximum. """
        # apply min-max scaling
        df[var] -= df[var].min()
        df[var] /= df[var].max()
        return df[var]


def read_raw_ftb_sphyncs(
    raw_data,
    activity,
    start_time=None,
    period=None
):
    """Reader function for raw .csv file recorded by MotionWatch 8 (CamNtech).

    Parameters
    ----------
    df: pandas.DataFrame
        Preloaded data in pandas dataframe.
    start_time: datetime-like str
        If not None, the start_time will be used to slice the data.
        Default is None.
    period: str
        Default is None.
    data_dtype: dtype
        The dtype of the raw data. Default is np.int.
    light_dtype: dtype
        The dtype of the raw light data. Default is np.float.

    Returns
    -------
    raw : Instance of RawFTB
        An object containing raw FTB data
    """

    return RawFTB_SPHYNCS(
        raw_data=raw_data,
        activity=activity,
        start_time=start_time,
        period=period
    )
