import pandas as pd
import numpy as np
import os

from lxml import etree
from ..base import BaseRaw
from pyActigraphy.light import LightRecording
import matplotlib.pyplot as plt


class RawFTB(BaseRaw):

    """Raw object from .json file (recorded by Fitbit)

    Parameters
    ----------
    path_to_fitbit: str
        Path to the folder structure from Fitbit download (.../Fitbit/...).
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
        path_to_fitbit,
        start_time=None,
        period=None,
        name=None
    ):
        
        # read csv file
        self._url = path_to_fitbit
        raw_data = self.__reading_and_parsing_file()
        raw_data = self.__preprocess_raw_data(raw_data)

        # extract informations from the header
        start = self.__extract_ftb_start_time(raw_data)
        frequency = self.__extract_frequency(raw_data)
        frequency_light = frequency # same frequency as "activity"

        activity_data = self.__extract_activity_data(raw_data)
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
            fpath=path_to_fitbit,
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
        
    def __get_fnames(self, prefix):
        """ Get a list of all available JSON files for the given sensor type ('calories' | 'heart'). """
        # Get a list of filenames in the directory
        file_list = os.listdir(self._url)
        
        # Filter the filenames to select only JSON files with the given prefix
        fnames = [filename for filename in file_list if filename.startswith(prefix) and filename.endswith('.json')]
        return fnames
    
    def __load_fitbit_json(self, prefix):
        """ Open and read the JSON file. Save as pandas.DataFrame and resample to 1 minute resolution. """
        # get filenames
        fnames = self.__get_fnames(prefix)
        
        # Initialize an empty DataFrame to store the combined data
        df = pd.DataFrame()
        
        # loop over all files
        for fname in fnames:
            # make file path
            file_path = os.path.join(self._url, fname)
            
            # open the JSON file
            with open(file_path, 'r') as json_file:
                data = pd.read_json(json_file)
                # unpack heart data
                try:
                    bpm_array = np.array([item['bpm'] for item in data['value'].values])
                    data['value'] = bpm_array
                except:
                    pass
            # Concatenate the data into the combined DataFrame
            df = pd.concat([df, data], ignore_index=True)
        # set timeindex
        df = df.set_index('dateTime').rename(columns={'value': prefix})
        # resample to 1 minute (heart rate is sampled at 5 seconds)
        df = df.resample('1T').mean()
        return df       

    def __reading_and_parsing_file(self):
        """ Load the raw data from JSON file """
        # search for file names starting with 'calories'
        df_calories = self.__load_fitbit_json('calories')
        # search for file names starting with 'heart'
        df_heart = self.__load_fitbit_json('heart')
        return pd.concat([df_calories, df_heart], axis=1)

    def __extract_ftb_start_time(self, df):
        """ Extract start time from the raw dataframe"""
        return df.index[0]

    def __extract_frequency(self, df):
        """ Extract frequency from the raw dataframe"""
        return pd.Timedelta(1, unit=pd.infer_freq(df.index[:3]))

    def __extract_activity_data(self, df):
        """ Extract calories as surrogate activity measurement from the raw dataframe """
        return df['calories']

    def __extract_light_data(self, df):
        """ Extract heart rate measurement from the raw dataframe as surrogate 'light' """
        return df['heart']
    
    def __preprocess_raw_data(self, df):
        """ Normalize the calories and heart rate. """
        # replace zero values by nan 
        df['heart']    = self.__preprocess_zeros_to_nan(df, 'heart')
        df['calories'] = self.__preprocess_zeros_to_nan(df, 'calories') 
        # normalize the heart rate by min-max scaling
        df['heart']    = self.__preprocess_minmax(df, 'heart')
        # normalize the calories by removing the baseline followed by min-max scaling
        df['calories'] = self.__preprocess_calories(df, 'calories')
        return df
    
    def __preprocess_zeros_to_nan(self, df, var):
        """ set zero values to nan """
        # get index of zero values
        to_nan = (df[var] <= 0)  
        # get column index of variable
        col_idx = df.columns.get_loc(var)
        # set zero values to NaN
        df.iloc[to_nan, col_idx] = np.nan
        return df[var]
        
    def __preprocess_minmax(self, df, var):      
        """ Normalize the data by subtracting the minimum and dividng by the maximum. """
        # apply min-max scaling
        df[var] -= df[var].min()
        df[var] /= df[var].max()
        return df[var]
    
    def __preprocess_calories(self, df, var):
        """ Normalize the calories in segments."""
        # get baseline
        baseline = self.__get_baseline(df, var)
        # subtract baseline from data    
        df[var] = df[var] - baseline.values
        # get indeces of the segments
        edges = self.__get_baseline_edges(baseline)
        #extract segments according to the edges
        segments = [df.iloc[start:end] for start, end in zip(edges[:-1], edges[1:])]
        #normalize the segments
        normalized_segments = [self.__preprocess_minmax(segment.copy(), var) for segment in segments]
        # update the original df
        for i, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
            df.loc[df.index[start:end], var] =normalized_segments[i]
        return df[var]    
    
    def __get_baseline(self, df, var):
        """ estimate the baseline. """
        # get daily baseline
        daily_offset = df.groupby(df.index.date)[var].min()
        baseline = pd.Series(df.index.date).map(daily_offset)
        return baseline
    
    def __get_baseline_edges(self, baseline):
        """ get the indeces of edges of the baseline """
        # get indeces of the edges
        edges = np.where(baseline.diff() != 0)[0]
        # add length as last 'edge' to define the segments
        edges = np.append(edges, len(baseline))
        return edges


def read_raw_ftb(
    path_to_fitbit,
    start_time=None,
    period=None,
    name=None
):
    """Reader function for raw .json file recorded by Fitbit.

    Parameters
    ----------
    path_to_fitbit: str
        Path to the folder structure from Fitbit download (.../Fitbit/...).
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

    return RawFTB(
        path_to_fitbit=path_to_fitbit,
        start_time=start_time,
        period=period,
        name=name
    )
