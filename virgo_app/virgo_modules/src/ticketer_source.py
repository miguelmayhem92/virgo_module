import yfinance as yf
import pandas as pd
import numpy as np
import json

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns; sns.set()

from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import datetime
from dateutil.relativedelta import relativedelta

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm

import scipy.stats as stats

from ta.momentum import RSIIndicator, ROCIndicator, StochRSIIndicator,StochasticOscillator, WilliamsRIndicator
from ta.trend import VortexIndicator

import warnings
warnings.filterwarnings('ignore')

from hmmlearn.hmm import GaussianHMM

from plotly.colors import DEFAULT_PLOTLY_COLORS

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from feature_engine.imputation import MeanMedianImputer

from itertools import combinations, chain

from feature_engine.encoding import OneHotEncoder
from feature_engine.selection import DropFeatures, DropCorrelatedFeatures
from feature_engine.timeseries.forecasting import LagFeatures
from feature_engine.imputation import MeanMedianImputer
from feature_engine.discretisation import EqualWidthDiscretiser

from sklearn.linear_model import HuberRegressor

from .aws_utils import upload_file_to_aws

import logging

from virgo_modules.src.hmm_utils import trainer_hmm

class InverseHyperbolicSine(BaseEstimator, TransformerMixin):

    """
    Class that applies inverse hyperbolic sine for feature transformation.
    this class is compatible with scikitlearn pipeline

    Attributes
    ----------
    features : list
        list of features to apply the transformation
    prefix : str
        prefix for the new features. is '' the features are overwrite

    Methods
    -------
    fit(additional="", X=DataFrame, y=None):
        fit transformation.
    transform(X=DataFrame, y=None):
        apply feature transformation
    """

    def __init__(self, features, prefix = ''):
        self.features = features
        self.prefix = prefix

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for feature in self.features:
            X[f'{self.prefix}{feature}'] = np.arcsinh(X[feature])
        return X

class VirgoWinsorizerFeature(BaseEstimator, TransformerMixin):

    """
    Class that applies winsorirization of a feature for feature transformation.
    this class is compatible with scikitlearn pipeline

    Attributes
    ----------
    feature_configs : dict
        dictionary of features and configurations. the configuration has high and low limits per feature

    Methods
    -------
    fit(additional="", X=DataFrame, y=None):
        fit transformation.
    transform(X=DataFrame, y=None):
        apply feature transformation
    """

    def __init__(self, feature_configs):
        self.feature_configs = feature_configs
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for feature in self.feature_configs:
            lower = self.feature_configs[feature]['min']
            upper = self.feature_configs[feature]['max']
            X[feature] = np.where( lower > X[feature], lower, X[feature])
            X[feature] = np.where( upper < X[feature], upper, X[feature])
        return X

class FeatureSelector(BaseEstimator, TransformerMixin):

    """
    Class that applies selection of features.
    this class is compatible with scikitlearn pipeline

    Attributes
    ----------
    columns : list
        list of features to select

    Methods
    -------
    fit(additional="", X=DataFrame, y=None):
        fit transformation.
    transform(X=DataFrame, y=None):
        apply feature transformation
    """

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.columns]

class FeaturesEntropy(BaseEstimator, TransformerMixin):
    """
    Class that creates a feature that calculate entropy for a given feature classes, but it might get some leackeage in the training set.
    this class is compatible with scikitlearn pipeline

    Attributes
    ----------
    columns : list
        list of features to select
    entropy_map: pd.DataFrame
        dataframe of the map with the entropies per class
    perc: float
        percentage of the dates using for calculate the entropy map
    
    Methods
    -------
    fit(additional="", X=DataFrame, y=None):
        fit transformation.
    transform(X=DataFrame, y=None):
        apply feature transformation
    """
    
    def __init__(self, features, target, feature_name = None, feature_type = 'discrete', perc = 0.5, default_null = 0.99):
        
        self.features = features
        self.feature_type = feature_type
        self.target = target
        self.perc = perc
        self.default_null = default_null
        
        if not feature_name:
            self.feature_name = '_'.join(features)
            self.feature_name = self.feature_name + '_' + target + '_' + feature_type
        else:
            self.feature_name = feature_name
            
    def fit(self, X, y=None):

        unique_dates = list(X['Date'].unique())
        unique_dates.sort()
        
        total_length = len(unique_dates)
        cut = int(round(total_length*self.perc,0))
        train_dates = unique_dates[:cut]
        max_train_date = max(train_dates)
        
        X_ = X[X['Date'] <= max_train_date].copy()
        df = X_.join(y, how = 'left')

        column_list = [f'{self.feature_type}_signal_{colx}' for colx in self.features]
        
        df_aggr = (
            df
            .groupby(column_list, as_index = False)
            .apply(
                lambda x: pd.Series(
                    dict(
                        counts = x[self.target].count(),
                        trues=(x[self.target] == 1).sum(),
                        falses=(x[self.target] == 0).sum(),
                    )
                )
            )
            .assign(
                trues_rate=lambda x: x['trues'] / x['counts']
            )
            .assign(
                falses_rate=lambda x: x['falses'] / x['counts']
            )
            .assign(
                log2_trues = lambda x: np.log2(1/x['trues_rate'])
            )
            .assign(
                log2_falses = lambda x: np.log2(1/x['falses_rate'])
            )
            .assign(
                comp1 = lambda x: x['trues_rate']*x['log2_trues']
            )
            .assign(
                comp2 = lambda x: x['falses_rate']*x['log2_falses']
            )
            .assign(
                class_entropy = lambda x: np.round(x['comp1']+x['comp2'],3)
            )
        )
        
        self.column_list = column_list
        self.entropy_map = (
            df_aggr
            [column_list+['class_entropy']]
            .rename(columns = {'class_entropy': self.feature_name})
            .copy()
        )
        
        del df, df_aggr, X_
        return self

    def transform(self, X, y=None):

        X = X.join(self.entropy_map.set_index(self.column_list), on=self.column_list, how = 'left')
        X[self.feature_name] = X[self.feature_name].fillna(self.default_null)
        return X

class signal_combiner(BaseEstimator, TransformerMixin):

    """
    Class that applies feature combination of binary signals.
    this class is compatible with scikitlearn pipeline

    ...

    Attributes
    ----------
    columns : list
        list of features to select
    drop : boolean
        drop combining features
    prefix_up : str
        up prefix of the base feature
    prefix_low : str
        low prefix of the base feature

    Methods
    -------
    fit(additional="", X=DataFrame, y=None):
        fit transformation.
    transform(X=DataFrame, y=None):
        apply feature transformation
    """

    def __init__(self, columns, drop = True, prefix_up = 'signal_up_', prefix_low = 'signal_low_'):
        self.columns = columns
        self.drop = drop
        self.prefix_up = prefix_up
        self.prefix_low = prefix_low

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for column in self.columns:
            X['CombSignal_'+column] = np.where(
                X[self.prefix_up + column] == 1,
                1,
                np.where(
                    X[self.prefix_low + column] == 1,
                    1,
                    0
                )
            )
            if self.drop:
                X = X.drop(columns = [self.prefix_up + column, self.prefix_low + column])
        return X

def data_processing_pipeline(features_base,features_to_drop = False, lag_dict = False, combine_signals = False, discretize_columns = False, correlation = 0.77):

    '''
    create a scikit learn pipeline object using different configurations and feature engineering blocks with a given flow

            Parameters:
                    features_to_drop (list): list of features to drop
                    lag_dict (dict): feature dictionary with configurations to apply lags
                    combine_signals (list): list of columns/signals to combine
                    discretize_columns (list): list of features to discretize, bins is fixed
                    correlation (float): correaltion score threshold for feature selection

            Returns:
                    pipe (obj): pipeline object
    '''

    lag_pipe_sec = [(f'lags_{key}', LagFeatures(variables = key, periods = lag_dict[key])) for key in lag_dict] if lag_dict else []
    drop_pipe = [('drop_features' , DropFeatures(features_to_drop=features_to_drop))] if features_to_drop else []
    merge = [('signal_combiner', signal_combiner(combine_signals))] if combine_signals else []
    discretize = [('discretize',EqualWidthDiscretiser(discretize_columns, bins = 20 ))] if discretize_columns else []
    drop_corr = [('drop_corr', DropCorrelatedFeatures(threshold=correlation))] if correlation else []

    pipe = Pipeline(
        [('selector', FeatureSelector(features_base))] + \
        [('encoding',OneHotEncoder(top_categories=None, variables=['hmm_feature']))]  + \
        merge + \
        discretize + \
        lag_pipe_sec + \
        [('fill na', MeanMedianImputer())] + \
        drop_corr + \
        drop_pipe
    )
    return pipe

def states_relevance_score(data, default_benchmark_sd = 0.00003, t_threshold = 2):
    '''
    calculate relevance score and summary report for hmm model 

            Parameters:
                    default_benchmark_sd (float): default value to bias SD for t calculation
                    t_threshold (float): alpha or z threshold for the normalized score

            Returns:
                    mean_relevance (float): mean relevance score of the states
                    cluster_returns (pd.DataFrame): summary report of the analysis
                    number_relevant_states (int): number of relevant states
    '''
    ## legnths
    cluster_lengths = data.groupby(['hmm_feature','chain_id'],as_index = False).agg(chain_lenght = ('hmm_chain_order','max'))
    cluster_lengths = cluster_lengths.groupby('hmm_feature').agg(cluster_length_median = ('chain_lenght','median'))
    ## means
    def quantile2(x):
        return x.quantile(0.25)
    def quantile3(x):
        return x.quantile(0.75)

    cluster_returns = data.groupby('hmm_feature').agg(
        n_uniques = ('chain_id','nunique'),
        n_obs = ('Date','count'),
        cluster_ret_q25 = ('chain_return',quantile2),
        cluster_ret_median = ('chain_return','median'),
        cluster_ret_q75 = ('chain_return',quantile3),
    )
    cluster_returns =  cluster_returns.join(cluster_lengths, how = 'left')
    cluster_returns['perc_dispute'] = np.where(
        np.sign(cluster_returns['cluster_ret_q25']) != np.sign(cluster_returns['cluster_ret_q75']),
        1,0
    )
    cluster_returns['iqr'] = cluster_returns.cluster_ret_q75 - cluster_returns.cluster_ret_q25
    cluster_returns['perc_25'] = abs(cluster_returns.cluster_ret_q25)/cluster_returns['iqr']
    cluster_returns['perc_75'] = abs(cluster_returns.cluster_ret_q75)/cluster_returns['iqr']
    cluster_returns['min_perc'] = cluster_returns[['perc_25','perc_75']].min(axis = 1)
    cluster_returns['min_overlap'] = np.where(cluster_returns['perc_dispute'] == 1,cluster_returns['min_perc'],0)
    cluster_returns['abs_median'] = abs(cluster_returns['cluster_ret_median'])
    cluster_returns = cluster_returns.drop(columns = ['perc_25','perc_75','min_perc'])

    ## relevance or importance
    # naive aproach
    cluster_returns['relevance'] =  cluster_returns['abs_median'] + ( 0.5 - cluster_returns['min_overlap'])
    cluster_returns['t_calc'] = (cluster_returns['cluster_ret_median'] - 0)/(cluster_returns['iqr']/cluster_returns['n_obs'] + default_benchmark_sd/cluster_returns['n_obs'])**(1/2)
    cluster_returns['abs_t_accpted'] = abs(cluster_returns['t_calc'])
    cluster_returns['t_accpted'] = abs(cluster_returns['abs_t_accpted']) > t_threshold

    mean_relevance = cluster_returns['abs_t_accpted'].mean()
    number_relevant_states = len(cluster_returns[cluster_returns.t_accpted == True])

    return mean_relevance, cluster_returns, number_relevant_states


class stock_eda_panel(object):

    """
    Class that initialy gets stock data then apply feature enginering, enrichment, analysis, plotting, model training etc.

    Attributes
    ----------
    stock_code : str
        symbol of the asset
    n_days : str
        number of days to extract data
    data_window : str
        large window to extract data. Large window is required o extract more data. e.g. '5y', '10y', '15'
    df : pd.DataFrame
        Pandas dataframe of the asset data with features
    strategy_log: pd.DataFrame
        Pandas dataframe that has the results of different tested strategies (result from strategy simulator hmm)
    best_strategy: list
        features of the best performing strategy (result from strategy simulator hmm)
    top_10_strategy: dict
        top 10 best performing strategies (result from strategy simulator hmm)
    settings: dict
        configuration dictionary of the features and other parameters

    Methods
    -------
    augmented_dickey_fuller_statistics(time_series=pd.Series, label=str):
        Perform dickey fuller or stationary test for a given time series
        It will print p value of the features
    get_data():
        Get asset data performing some data normalization or formating (in case of dates)
    plot_series_returns(roll_mean_lags1=int, roll_mean_lags2=int)
        Display plot that time series with mean rolling windows and rolling standard deviations of daily closing prices
    seasonal_plot():
        Display time series split by year
    plot_price_signal(feature=str, feature_2=str, opacity=float):
        Display botton and roof signals over the closing prices
    volatility_analysis(lags=int, trad_days=int, window_log_return=int, plot=boolean, save_features=boolean):
        this method performs log return and volatilyty analysis of the closing prices
    find_lag(feature=str, lag_list=list, column_target=str,posterior_lag=int, test_size=int):
        displays correlation curves, using spearman and pearson correlation, of a given feature at different time lags with respecto to a given target
    outlier_plot(zlim=float, plot=boolean, save_features=boolean):
        perform outlier analysis of the log returns. It also permors normality test of returns
    analysis_roll_mean_log_returns(lags=int, plot=boolean):
        perform analysis of lags of the mean rolling log return
    compute_clip_bands(feature_name=str,threshold=float):
        compute outlier detection for a given signal, Note that this follows mean reversion procedure and feature has to be stationary. Also botton and roof resulting signals is attached to the dataframe
    signal_plotter(feature_name=str):
        display analysis plot of a feature with high and low signals
    log_features_standard(feature_name=str):
        save resulting feature names in an standard structure
    relative_spread_MA(ma1=int, ma2=int, threshold=float, plot=boolean, save_features=boolean):
        perform relative moving average features, one for short term and another for long/mid term
    pair_feature(pair_symbol=str, plot=boolean):
        initialize pair feature data extraction and analysis
    calculate_cointegration(series_1=pd.series, series_2=pd.series):
        calculate cointegration score for two time series
    bidirect_count_feature(rolling_window=int, threshold=float, plot=boolean, save_features=boolean):
        perform negative and positive return counting in a given rolling time window
    get_relative_range_feature(window=int, threshold=float, plot=boolean, save_features=boolean):
        perform relative spread of opening and closing price
    rsi_feature_improved(window=int, threshold=float, plot=boolean, save_features=boolean):
        perform relative strength index
    days_features_bands(window=int, threshold=float, plot=boolean, save_features=boolean):
        compute mean returns for a given day of the week in a window scope per day
    analysis_smooth_volume(window=int, threshold=float, plot=boolean, save_features=boolean):
        compute feature of thrading volumes
    roc_feature(window=int, threshold=float, plot=boolean, save_features=boolean):
        perform price rate of change
    stoch_feature(window=int, smooth1=int, smooth2=int, threshold=float, plot=boolean, save_features=boolean):
        perform stochastic oscilator RSI feature
    stochastic_feature(window=int, smooth=int, threshold=float, plot=boolean, save_features=boolean):
        perform stochastic oscilator feature
    william_feature(lbp=int, threshold=float, plot=boolean, save_features=boolean):
        perfom fast stochastic oscilator or william indicator
    vortex_feature(window=int, threshold=float, plot=boolean, save_features=boolean):
        perform vortex oscilator
    minmax_pricefeature(type_func=str, window=int, distance=bolean, save_features=boolean)
        get relative price/ distance feature with respect to the min/max price in a given window
    pair_index_feature(pair_symbol=str, feature_label=str, window=int, threshold=float, plot=boolean, save_features=boolean):
        perform additional asset ROC feature, then a new feature is created in the main dataframe
    produce_order_features(feature_name=str, save_features=boolean):
        perform a feature that captures high and low values in an index. this is usefull to know duration/persistence of a signal
    create_hmm_derived_features():
        create features derived from hmm states features. Features are the index of the state, the duration of the state, chain raturn
    cluster_hmm_analysis(n_clusters=int,features_hmm=list, test_data_size=int, seed=int, lag_returns_state=int, plot=boolean, save_features=boolean, model=obj):
        create or use a hmm model
    sharpe_ratio(return_series=pd.Series, n_trad_days=int, rf=float):
        perform sharpe ratio of a given time series return
    treat_signal_strategy(test_data=pd.DataFrame, strategy=list):
        helper method that treats signals and converts signals to 1 or 0
    stategy_simulator(features=list, hmm_feature=boolean):
        execute strategy and get some performance metrics like sharpe ratio, return
    viz_strategy(strategy):
        display analysis plot of a given strategy
    deep_dive_analysis_hmm(test_data_size=int, split=str):
        display analysis plot hmm model
    get_targets(steps=int):
        produce regression target return taking future prices
    get_categorical_targets(horizon=int, flor_loss=float, top_gain=float):
        produce binary target return taking future prices. it produce two targets, one for high returns and another for low returns
    get_configurations(test_data_size=int, val_data_size=int, model_type=str):
        produce configuration dictionary that were saved in the feature generation methods if save_features was activated
    """

    def __init__(self, stock_code, n_days, data_window = '5y'):

        """
        Initialize object

        Parameters
        ----------
        stock_code (str): symbol of the asset
        n_days (str): number of days to extract data
        data_window (str): large window to extract data. Large window is required o extract more data. e.g. '5y', '10y', '15'

        Returns
        -------
        None
        """

        self.stock_code = stock_code
        self.n_days = n_days
        self.today = datetime.date.today()
        self.features = list()
        self.signals = list()
        self.data_window = data_window

    def augmented_dickey_fuller_statistics(self,time_series, label):
        """
        Perform dickey fuller or stationary test for a given time series
        It will print p value of the features

        Parameters
        ----------
        time_series (pd.Series): pandas series of the time series
        label (pd.Series): feature name

        Returns
        -------
        None
        """
        result = adfuller(time_series.dropna().values)
        print('p-value: {} for the series {}'.format(round(result[1],6), label))

    def get_data(self):
        """
        Get asset data performing some data normalization or formating (in case of dates)

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        begin_date = self.today - relativedelta(days = self.n_days)
        begin_date_str = begin_date.strftime('%Y-%m-%d')

        stock = yf.Ticker(self.stock_code)
        df = stock.history(period=self.data_window)

        df = df.sort_values('Date')
        df.reset_index(inplace=True)
        df['Date'] = pd.to_datetime(df['Date'], format='mixed',utc=True).dt.date
        df['Date'] = pd.to_datetime(df['Date'])

        df = df[df.Date >= begin_date_str ]
        self.settings_general = {
            'n_days':self.n_days,
            'begin_date':begin_date_str,
            'data_window': self.data_window,
            'execution_date': self.today.strftime('%Y-%m-%d')
        }
        self.df = df

        ### cleaning volume
        ### volume clearning
        self.df['Volume'] = np.where(self.df['Volume'] <= 10, np.nan, self.df['Volume'])
        self.df['Volume'] = self.df['Volume'].fillna(method='bfill')

        ## filling

        base_columns_unit_test = ['Open','High','Low','Close','Volume']
        self.df[base_columns_unit_test] = self.df[base_columns_unit_test].fillna(method='ffill')

        ## cleaning nulls

        xs = self.df[base_columns_unit_test].isnull().sum()/self.df[base_columns_unit_test].count()
        reject_columns = list(xs[xs > 0.5].index.values)

        if len(reject_columns) > 0:
            logging.warning("the following columns have many nulls and are drop: {}".format(reject_columns))
            self.df = self.df.drop(columns = reject_columns)

    def plot_series_returns(self,roll_mean_lags1,roll_mean_lags2):

        """
        Display plot that time series with mean rolling windows and rolling standard deviations of daily closing prices

        Parameters
        ----------
        roll_mean_lags1 (int): short term window
        roll_mean_lags2 (int): mid/long term window

        Returns
        -------
        None
        """

        df = self.df
        begin_date = self.today - relativedelta(days = self.n_days)
        begin_date_str = begin_date.strftime('%Y-%m-%d')

         ### getting rolling mean
        df["Close_roll_mean"] = (
            df.sort_values("Date")["Close"]
            .transform(lambda x: x.rolling(roll_mean_lags1, min_periods=1).mean())
        )

        df["Close_roll_mean_2"] = (
            df.sort_values("Date")["Close"]
            .transform(lambda x: x.rolling(roll_mean_lags2, min_periods=1).mean())
        )

        ### getting rolling stdv
        df["Close_roll_std"] = (
            df.sort_values("Date")["Close"]
            .transform(lambda x: x.rolling(roll_mean_lags1, min_periods=1).std())
        )
        df["upper"] = df['Close_roll_mean'] + df["Close_roll_std"]*2
        df["lower"] = df['Close_roll_mean'] - df["Close_roll_std"]*2

        df = df[df.Date >= begin_date_str ]

        fig = make_subplots(rows=1, cols=1,vertical_spacing = 0.1,shared_xaxes=True,
                           subplot_titles=(
                               f'stock: {self.stock_code} roll window analysis: {roll_mean_lags1} days'
                           ))

        fig.add_trace(go.Scatter(x=df['Date'], y=df.Close, marker_color = 'blue', name='Price'),row=1, col=1)

        fig.add_trace(go.Scatter(x=df['Date'], y=df.Close_roll_mean, marker_color = 'black', name='roll mean' ),row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df.Close_roll_mean_2, marker_color = 'grey', name='roll mean 2' ),row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df.lower, marker_color = 'pink',legendgroup='bound', name='bound' ),row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df.upper, marker_color = 'pink',legendgroup='bound', name='bound', showlegend=False ),row=1, col=1)

        fig.update_layout(height=500, width=1200, title_text=f"stock {self.stock_code} vizualization")
        fig.show()

    def seasonal_plot(self):

        """
        Display time series split by year

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        df = self.df
        years = list(df['Date'].dt.year.unique())
        years.sort()
        years = years[::-1]
        years_last = max(years)

        fig = make_subplots(rows=1, cols=1,vertical_spacing = 0.1,shared_xaxes=True,)

        for i,year in enumerate(years):
            df_plot = df[df.Date.dt.year == year].sort_values('Date')
            df_plot['Date_trunc'] = df_plot.Date.dt.strftime('%m-%d')
            df_plot['Date_trunc'] = pd.to_datetime(df_plot['Date_trunc'], format='%m-%d')
            if year == years_last:
                fig.add_trace(go.Scatter(x= df_plot.Date_trunc, y=df_plot.Close, name=str(year)),row=1, col=1)
                continue
            fig.add_trace(go.Scatter(x= df_plot.Date_trunc, y=df_plot.Close, name=str(year), line = dict(dash='dash')),row=1, col=1)

        fig.update_layout(height=500, width=1400, title_text=f"stock {self.stock_code} seasonal vizualization")
        fig.show()

    def plot_price_signal(self, feature, feature_2 = '', opacity = 0.3):

        """
        Display botton and roof signals over the closing prices

        Parameters
        ----------
        feature (str): name of the main feature to plot
        feature_2 (str): name of the alternative feature to plot
        opacity (float): opacity degree of the signals points

        Returns
        -------
        None
        """

        signal_up_list = [f'signal_up_{feature}', f'signal_up_{feature_2}']
        signal_low_list = [f'signal_low_{feature}', f'signal_low_{feature_2}']
        norm_list = [f'norm_{feature}', f'z_{feature}', feature]

        fig = make_subplots(rows=2, cols=1,vertical_spacing = 0.1, shared_xaxes=True, subplot_titles = [f'norm signal - {feature}',f'signal over price'] )

        for norm_feat in norm_list:
            if norm_feat in self.df.columns:
                fig.add_trace(go.Scatter(x=self.df['Date'], y=self.df[norm_feat],legendgroup="up", mode='lines',name = norm_feat, marker_color = 'blue'),col = 1, row = 1)
                break


        fig.add_trace(go.Scatter(x=self.df['Date'], y=self.df['Close'], mode='lines',name = 'history', marker_color = 'grey'),col = 1, row = 2)

        if feature == 'MA_spread':
            fig.add_trace(go.Scatter(x=self.df['Date'], y=self.df[self.ma1_column],legendgroup="ma", mode='lines',name = self.ma1_column, marker_color = 'black'),col = 1, row = 2)
            fig.add_trace(go.Scatter(x=self.df['Date'], y=self.df[self.ma2_column],legendgroup="ma", mode='lines',name = self.ma2_column, marker_color = 'grey'),col = 1, row = 2)

        for norm_feat in norm_list:
            if norm_feat in self.df.columns:
                fig.add_trace(go.Scatter(x=self.df['Date'], y=np.where(self.df[norm_feat] > 0, self.df['Close'], np.nan),legendgroup="up", mode='markers',name = 'up', marker_color = 'green',opacity = opacity),col = 1, row = 2)
                fig.add_trace(go.Scatter(x=self.df['Date'], y=np.where(self.df[norm_feat] <= 0, self.df['Close'], np.nan),legendgroup="low", mode='markers',name = 'low', marker_color = 'red',opacity = opacity),col = 1, row = 2)

        for signal_up in signal_up_list:
            if signal_up in self.df.columns:
                fig.add_trace(go.Scatter(x=self.df['Date'], y=np.where(self.df[signal_up] == 1, self.df['Close'], np.nan),legendgroup="high up", mode='markers',name = 'high up', marker_color = 'green'),col = 1, row = 2)

        for signal_low in signal_low_list:
            if signal_low in self.df.columns:
                fig.add_trace(go.Scatter(x=self.df['Date'], y=np.where(self.df[signal_low] == 1, self.df['Close'], np.nan),legendgroup="high low", mode='markers',name = 'high low', marker_color = 'red'),col = 1, row = 2)

        fig.update_layout(height=900, width=1200)
        fig.show()

    def volatility_analysis(self, lags, trad_days, window_log_return, plot = False, save_features = False):

        """
        this method performs log return and volatilyty analysis of the closing prices

        Parameters
        ----------
        lags (int): number of lags to apply to the closing prices
        trad_days (int): number of trading days to anualize returns or volatility
        window_log_return (int): window for rolling returns
        plot (boolean): True to display plot
        save_features (boolean): True to save feature configuration and feature names

        Returns
        -------
        None
        """

        df = self.df
        df['log_return'] = np.log(df.Close/df.Close.shift(lags))
        df['sqr_log_return'] = np.square(df.log_return)
        df['volatility_log_return'] = df.log_return.rolling(window = trad_days).std()*np.sqrt(252)

        df["roll_mean_log_return"] = (
                df.sort_values("Date")["log_return"]
                .transform(lambda x: x.rolling(window_log_return, min_periods=1).mean())
            )

        if save_features:
            self.features.append('volatility_log_return')
            self.features.append('roll_mean_log_return')
            self.features.append('log_return')
            self.settings_volatility = {'lags':lags, 'trad_days':trad_days, 'window_log_return':window_log_return}

        if plot:
            fig = make_subplots(rows=3, cols=1,vertical_spacing = 0.02,shared_xaxes=True,
                            specs=[
                                [{}],
                                [{"secondary_y": True}],
                                [{}],
                                  ])

            fig.add_trace(go.Scatter(x= df.Date, y=df.Close, name='Price'),row=1, col=1)
            fig.add_trace(go.Scatter(x= df.Date, y=df.log_return, name='log_return'),secondary_y=False, row=2, col=1)
            fig.add_trace(go.Scatter(x= df.Date, y=df.roll_mean_log_return, name='roll_mean_log_return'),secondary_y=False, row=2, col=1)
            #fig.add_trace(go.Scatter(x= df.Date, y=df.sqr_log_return, name='sqr_log_return'),secondary_y=True, row=2, col=1)
            fig.add_trace(go.Scatter(x= df.Date, y=df.volatility_log_return, name='volatility_log_return'),row=3, col=1)

            fig.update_yaxes(title_text='Price',row=1, col=1)
            fig.update_yaxes(title_text='log_return', secondary_y=False, row=2, col=1)
            fig.update_yaxes(title_text='sqr_log_return', secondary_y=True, row=2, col=1)
            fig.update_yaxes(title_text='volatility_log_return',row=3, col=1)

            fig.update_layout(height=1000, width=1400, title_text=f"stock {self.stock_code} volatility vizualization, lags: {lags} and trading days: {trad_days}")
            fig.show()

            print('___________________________________________')

            fig, axs = plt.subplots(1, 4,figsize=(20,4))
            plot_acf(df['log_return'].dropna(),lags=25, ax=axs[0])
            axs[0].set_title('acf log return')
            plot_pacf(df['log_return'].dropna(),lags=25, ax=axs[1])
            axs[1].set_title('pacf log return')
            plot_acf(df['roll_mean_log_return'].dropna(),lags=25, ax=axs[2])
            axs[2].set_title('acf roll_mean_log_return')
            plot_pacf(df['roll_mean_log_return'].dropna(),lags=25, ax=axs[3])
            axs[3].set_title('pacf roll_mean_log_return')
            plt.show()

            print('___________________________________________')

            self.augmented_dickey_fuller_statistics(df['log_return'], 'log_return')
            self.augmented_dickey_fuller_statistics(df['roll_mean_log_return'], 'roll_mean_log_return')


    def find_lag(self, feature, lag_list, column_target = 'log_return',posterior_lag = 4, test_size = 350):

        """
        displays correlation curves, using spearman and pearson correlation, of a given feature at different time lags with respecto to a given target

        Parameters
        ----------
        feature (str): feature name to apply lags
        lag_list (list): list of lags, each lag as integer
        column_target (str): target to get correlation, e.g return or mean reaturn
        posterior_lag (int): for the target, posterior window shift to calculate a window return
        test_size (int): data size of the test data. The remaining is going to be used as training data. This parameters is ment to avoid overfiting and leackage

        Returns
        -------
        None
        """

        results = dict()
        df = self.df.iloc[:-test_size,:][['Date','Close','roll_mean_log_return','log_return',feature]].sort_values('Date').copy()
        for i,lag in enumerate(lag_list):
            lag_column = f'{feature}_lag_{lag}'
            df[lag_column] = df[feature].shift(lag)
            df['target_posterior_lag'] = df[column_target].shift(-posterior_lag)
            df = df.dropna()
            r_log = stats.mstats.pearsonr(df['target_posterior_lag'], df[lag_column])
            sp_log = stats.spearmanr(df['target_posterior_lag'], df[lag_column])

            results[i] = {
                'lag':lag,
                'pearsonr_log_return':r_log[0],
                'spearman_log_return': sp_log[0],
            }
        del df
        results_df = pd.DataFrame(results).T

        fig = plt.figure(figsize = (10,3))
        plt.plot(results_df.lag,results_df.pearsonr_log_return,label = f'pearsonr_{column_target}')
        plt.plot(results_df.lag,results_df.spearman_log_return,label = f'spearman_{column_target}')
        plt.scatter(results_df.lag,results_df.pearsonr_log_return)
        plt.scatter(results_df.lag,results_df.spearman_log_return)
        plt.title(f'{feature}: correlation curve with the target {column_target} lag -{posterior_lag} periods')
        plt.legend()
        plt.axhline(y=0, color='grey', linestyle='--')
        plt.show()


    def outlier_plot(self, zlim, plot = False, save_features = False):

        """
        perform outlier analysis of the log returns. It also permors normality test of returns

        Parameters
        ----------
        zlim (float): alpha or z thrsholds for normalized returns
        plot (boolean): True to display plot
        save_features (boolean): True to save feature configuration and feature names

        Returns
        -------
        None
        """

        mean = self.df.log_return.mean()
        std = self.df.log_return.std()
        self.df['z_log_return'] = (self.df.log_return - mean)/std
        l1,l2 = 1.96, 1.0

        mean_ = self.df['z_log_return'].mean()
        self.df['z_std_log_return'] = self.df.sort_values("Date")["z_log_return"].rolling(50).std()
        self.df['up_outlier'] = zlim*self.df['z_std_log_return'] + mean_
        self.df['low_outlier'] = -zlim*self.df['z_std_log_return'] + mean_

        self.df['signal_low_outlier'] = np.where( (self.df['z_log_return'] < self.df['low_outlier'] ), 1, 0)
        self.df['signal_up_outlier'] = np.where( (self.df['z_log_return'] > self.df['up_outlier'] ), 1, 0)
        if save_features:
            self.signals.append('signal_low_outlier')
            self.signals.append('signal_up_outlier')
            self.settings_outlier = {'zlim':zlim}
        if plot:
            mu = self.df['z_log_return'].mean()
            sigma = self.df['z_log_return'].std()
            x = np.linspace(self.df['z_log_return'].min(),self.df['z_log_return'].max(), 15000)
            y = stats.norm.pdf(x, loc = mu, scale = sigma)

            fig, axs = plt.subplots(2, 1,figsize=(15,8))

            axs[0].hist(self.df['z_log_return'],density = True,bins = 100 , label = 'Returns distribution')
            axs[0].axvline(l1, color='r', linestyle='--')
            axs[0].axvline(-l1, color='r', linestyle='--')
            axs[0].axvline(l2, color='green', linestyle='--')
            axs[0].axvline(-l2, color='green', linestyle='--')
            axs[0].plot(x,y, linewidth = 3, color = 'r', label = 'Normal Dist Curve')

            axs[1].plot(self.df['Date'],self.df['z_log_return'])
            axs[1].plot(self.df['Date'],self.df['low_outlier'], linestyle='--')
            axs[1].plot(self.df['Date'],self.df['up_outlier'], linestyle='--')

            fig.legend()
            plt.show()

            z_stat, p_stat = stats.normaltest(self.df['z_log_return'].dropna())
            p_stat = round(p_stat, 7)
            print('---------------------- returns normality tests ----------------------------')
            if p_stat < 0.05:
                print(f'pvalue: {p_stat} then, returns do not follow a normal distribution')
            else:
                print(f'pvalue: {p_stat} then, returns follow a normal distribution')

    def analysis_roll_mean_log_returns(self, lags, plot = False):

        """
        perform analysis of lags of the mean rolling log return

        Parameters
        ----------
        lags (int): lags to apply to the roll log return
        plot (boolean): True to display plot

        Returns
        -------
        None
        """

        self.df['lag'] = self.df.roll_mean_log_return.shift(lags)
        self.df['Diff'] = self.df['roll_mean_log_return'] - self.df['lag']

        if plot:

            fig, axs = plt.subplots(1, 3,figsize=(19,4))
            self.df['Diff'].plot(ax=axs[0])
            plot_acf(self.df['Diff'].dropna(),lags=25, ax=axs[1])
            plot_pacf(self.df['Diff'].dropna(),lags=25, ax=axs[2])
            axs[0].set_title('Integration of the roll mean log-returns')
            axs[1].set_title('acf Integration of the roll mean log-returns')
            axs[2].set_title('pacf Integration of the roll mean log-returns')
            plt.show()

    def compute_clip_bands(self,feature_name,threshold):

        """
        compute outlier detection for a given signal, Note that this follows mean reversion procedure and feature has to be stationary. Also botton and roof resulting signals is attached to the dataframe

        Parameters
        ----------
        feature_name (str): feature name
        threshold (float): alpha or z thrsholds for normalized returns

        Returns
        -------
        None
        """

        self.df[f'norm_{feature_name}'] =  (self.df[feature_name] - self.df[feature_name].mean())/self.df[feature_name].std()
        mean_ = self.df[f'norm_{feature_name}'].mean()

        self.df[f'up_rollstd_{feature_name}'] = self.df.sort_values("Date")[f'norm_{feature_name}'].clip(0,100).rolling(50).std()
        self.df[f'low_rollstd_{feature_name}'] = self.df.sort_values("Date")[f'norm_{feature_name}'].clip(-100,0).rolling(50).std()

        self.df[f'upper_{feature_name}'] = threshold*self.df[f'up_rollstd_{feature_name}'] + mean_
        self.df[f'lower_{feature_name}'] = -threshold*self.df[f'low_rollstd_{feature_name}'] + mean_

        self.df[f'signal_low_{feature_name}'] = np.where( (self.df[f'norm_{feature_name}'] < self.df[f'lower_{feature_name}'] ), 1, 0)
        self.df[f'signal_up_{feature_name}'] = np.where( (self.df[f'norm_{feature_name}'] > self.df[f'upper_{feature_name}'] ), 1, 0)

    def signal_plotter(self, feature_name):

        """
        display analysis plot of a feature with high and low signals

        Parameters
        ----------
        feature_name (str): feature name

        Returns
        -------
        None
        """

        fig, axs = plt.subplots(1, 3,figsize=(17,5))

        axs[0].plot(self.df[f'upper_{feature_name}'],color = 'grey', linestyle='--')
        axs[0].plot(self.df[f'lower_{feature_name}'],color = 'grey', linestyle='--')
        axs[0].plot(self.df[f'norm_{feature_name}'])

        plot_acf(self.df[feature_name].dropna(),lags=25,ax = axs[1])
        axs[1].set_title(f'acf {feature_name}')

        plot_pacf(self.df[feature_name].dropna(),lags=25,ax = axs[2])
        axs[2].set_title(f'pacf {feature_name}')

        fig.show()

    def log_features_standard(self, feature_name):
        """
        save resulting feature names in an standard structure

        Parameters
        ----------
        feature_name (str): feature name

        Returns
        -------
        None
        """
        self.features.append(feature_name)
        self.signals.append(f'signal_up_{feature_name}')
        self.signals.append(f'signal_low_{feature_name}')

    #######################
    #### to be deprecated ####
    # def spread_MA(self, ma1, ma2, limit = 1.95, plot = False, save_features = False):

    #     self.df[f'MA_{ma1}'] = (self.df.sort_values("Date")["Close"].transform(lambda x: x.rolling(ma1, min_periods=1).mean()))
    #     self.df[f'MA_{ma2}'] = (self.df.sort_values("Date")["Close"].transform(lambda x: x.rolling(ma2, min_periods=1).mean()))

    #     self.ma1_column = f'MA_{ma1}'
    #     self.ma2_column = f'MA_{ma2}'
    #     self.df['MA_spread'] = self.df[f'MA_{ma1}'] - self.df[f'MA_{ma2}']

    #     self.df['norm_MA_spread'] =  (self.df['MA_spread'] - self.df['MA_spread'].mean())/self.df['MA_spread'].std()
    #     mean_ = self.df['norm_MA_spread'].mean()
    #     self.df['rollstd_MA_spread'] = self.df.sort_values("Date")["norm_MA_spread"].rolling(50).std()

    #     self.df['upper_MA_spread'] = limit*self.df['rollstd_MA_spread'] + mean_
    #     self.df['lower_MA_spread'] = -limit*self.df['rollstd_MA_spread'] + mean_

    #     self.df['signal_low_MA_spread'] = np.where( (self.df['norm_MA_spread'] < self.df['lower_MA_spread'] ), 1, 0)
    #     self.df['signal_up_MA_spread'] = np.where( (self.df['norm_MA_spread'] > self.df['upper_MA_spread'] ), 1, 0)

    #     ### ploting purposes
    #     self.df[f"Roll_mean_{ma1}"] = (
    #         self.df.sort_values("Date")["Close"]
    #         .transform(lambda x: x.rolling(ma1, min_periods=1).mean())
    #     )
    #     self.df[f"Roll_mean_{ma2}"] = (
    #         self.df.sort_values("Date")["Close"]
    #         .transform(lambda x: x.rolling(ma2, min_periods=1).mean())
    #     )


    #     print('--------------------------------------------------------------------')
    #     if save_features:
    #         self.features.append('MA_spread')
    #         self.signals.append('signal_low_MA_spread')
    #         self.signals.append('signal_up_MA_spread')
    #         self.settings_spread_ma = {'ma1':ma1, 'ma2':ma2, 'limit':limit}

    #     if plot:

    #         fig, axs = plt.subplots(1, 3,figsize=(21,4))

    #         axs[0].plot(self.df['Date'],self.df['norm_MA_spread'])
    #         axs[0].plot(self.df['Date'],self.df['upper_MA_spread'], linestyle='--')
    #         axs[0].plot(self.df['Date'],self.df['lower_MA_spread'], linestyle='--')
    #         axs[0].set_title('MA_spread series')

    #         plot_acf(self.df['MA_spread'].dropna(),lags=25, ax=axs[1])
    #         axs[1].set_title('acf MA_spread series')

    #         plot_pacf(self.df['MA_spread'].dropna(),lags=25, ax=axs[2])
    #         axs[2].set_title('acf MA_spread series')
    #         plt.show()
    ##################################################

    def relative_spread_MA(self, ma1, ma2, threshold = 1.95, plot = False, save_features = False):
        """
        perform relative moving average features, one for short term and another for long/mid term

        Parameters
        ----------
        ma1 (int): short term moving average window
        ma2 (int): long/mid term moving average window
        threshold (float): alpha or z thrsholds for the normalized feature
        plot (boolean): True to display plot
        save_features (boolean): True to save feature configuration and feature names

        Returns
        -------
        None
        """
        feature_name = 'rel_MA_spread'

        self.df[f'MA_{ma1}'] = (self.df.sort_values("Date")["Close"].transform(lambda x: x.rolling(ma1, min_periods=1).mean()))
        self.df[f'MA_{ma2}'] = (self.df.sort_values("Date")["Close"].transform(lambda x: x.rolling(ma2, min_periods=1).mean()))

        self.ma1_column = f'MA_{ma1}'
        self.ma2_column = f'MA_{ma2}'
        self.df[feature_name] = self.df[f'MA_{ma1}'] / self.df[f'MA_{ma2}']

        self.compute_clip_bands(feature_name,threshold)

        ### ploting purposes
        self.df[f"Roll_mean_{ma1}"] = (
            self.df.sort_values("Date")["Close"]
            .transform(lambda x: x.rolling(ma1, min_periods=1).mean())
        )
        self.df[f"Roll_mean_{ma2}"] = (
            self.df.sort_values("Date")["Close"]
            .transform(lambda x: x.rolling(ma2, min_periods=1).mean())
        )

        print('--------------------------------------------------------------------')
        if save_features:
            self.log_features_standard(feature_name)
            self.settings_relative_spread_ma = {'ma1':ma1, 'ma2':ma2, 'threshold':threshold}

        if plot:

            self.signal_plotter(feature_name)

    def pair_feature(self, pair_symbol, plot = False):
        """
        initialize pair feature data extraction and analysis

        Parameters
        ----------
        pair_symbol (str): symbol of the pair asset to extract
        plot (boolean): True to display plot
        
        Returns
        -------
        None
        """

        self.pair_symbol = pair_symbol
        begin_date = self.today - relativedelta(days = self.n_days)
        begin_date_str = begin_date.strftime('%Y-%m-%d')

        stock = yf.Ticker(self.pair_symbol)
        df = stock.history(period=self.data_window)
        df = df.sort_values('Date')
        df.reset_index(inplace=True)
        df['Date'] = pd.to_datetime(df['Date'], format='mixed',utc=True).dt.date
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[df.Date >= begin_date_str ]
        self.pair_df = df

        #### converting the same index ####
        dates_vector = self.df.Date.to_frame()
        self.pair_df = dates_vector.merge(self.pair_df, on ='Date',how = 'left')
        self.pair_df = self.pair_df.fillna(method = 'bfill')
        self.pair_df = self.pair_df.fillna(method = 'ffill')
        ########

        series_1 = self.df.Close.values.astype(float)
        series_2 = self.pair_df.Close.values.astype(float)

        series_2 = series_2[-len(series_1):]

        coint_flag, hedge_ratio = self.calculate_cointegration(series_1,series_2)
        self.df['pair_spread'] = series_1 - (hedge_ratio * series_2)

        if plot:
            asset_1 = self.stock_code
            asset_2 = self.pair_symbol
            asset_1_values = self.df['Close'].values/self.df['Close'].iloc[0].item()
            asset_2_values = self.pair_df['Close'].values/self.pair_df['Close'].iloc[0].item()
            plt.figure(1, figsize=(10,5))
            plt.plot(self.df['Date'],asset_1_values,label = asset_1)
            plt.plot(self.df['Date'],asset_2_values,label = asset_2)
            plt.legend()
            plt.show()

    def calculate_cointegration(self,series_1, series_2):
        """
        calculate cointegration score for two time series

        Parameters
        ----------
        series_1 (pd.series): time series
        series_2 (pd.series): time series
        
        Returns
        -------
        coint_flag (boolean): 1 if the p_value cointegration_t are lower than 0.05 and critical value
        hedge_value (float): beta from the regression model
        """

        coint_flag = 0
        coint_res = coint(series_1, series_2)
        coint_t = coint_res[0]
        p_value = coint_res[1]
        critical_value = coint_res[2][1]

        model = sm.OLS(series_1, series_2).fit()
        hedge_value = model.params[0]
        coint_flag = 1 if p_value < 0.05 and coint_t < critical_value else 0

        return coint_flag, hedge_value

    def produce_pair_score_plot(self, window, z_threshold, plot = False, save_features = False):
        """
        display analysis of the pair feature and save results in case if needed

        Parameters
        ----------
        window (int): window to apply to the rolling spread between pair and main asset
        z_threshold (float): alpha or z thrsholds for the normalized feature
        plot (boolean): True to display plot
        save_features (boolean): True to save feature configuration and feature names
        
        Returns
        -------
        None
        """
        spread_series = pd.Series(self.df.pair_spread)
        mean = spread_series.rolling(center = False, window = window).mean()
        std = spread_series.rolling(center = False, window = window).std()
        x = spread_series.rolling(center=False, window =  1).mean()
        z_score = (x - mean)/std
        self.df['pair_z_score'] = z_score
        self.df['signal_low_pair_z_score'] = np.where(self.df['pair_z_score'] < -z_threshold, 1, 0)
        self.df['signal_up_pair_z_score'] = np.where(self.df['pair_z_score'] > z_threshold, 1, 0)

        if save_features:
            self.log_features_standard('pair_z_score')
            self.settings_pair_feature = {'pair_symbol':self.pair_symbol,'window':window, 'z_threshold':z_threshold}

        if plot:
            pvalue = round(adfuller(z_score.dropna().values)[1],4)
            print(f'p value of the rolling z-score is {pvalue}')

            fig, axs = plt.subplots(2, 2,figsize=(17,11))

            axs[0,0].axhline(y=2, color='r', linestyle='--')
            axs[0,0].axhline(y=-2, color='r', linestyle='--')
            axs[0,0].axhline(y=1.1, color='grey', linestyle='--')
            axs[0,0].axhline(y=-1.1, color='grey', linestyle='--')
            axs[0,0].axhline(y=0, color='blue', linestyle='-.')
            axs[0,0].plot(self.df.pair_z_score)
            axs[0,0].set_title('z score from the spread')

            axs[0,1].plot(self.df['Date'],self.df['pair_spread'])
            axs[0,1].plot(self.df['Date'],np.where(self.df['signal_low_pair_z_score'] == 1, self.df['pair_spread'], np.nan),'o-r',color = 'red')
            axs[0,1].plot(self.df['Date'],np.where(self.df['signal_up_pair_z_score'] == 1, self.df['pair_spread'], np.nan),'o-r',color = 'green')
            axs[0,1].axhline(y=0, color='blue', linestyle='-.')
            axs[0,1].set_title('pair_sprear_plot')

            plot_acf(self.df['pair_z_score'].dropna(),lags=25, ax=axs[1,0])
            axs[1,0].set_title('acf pair_z_score')

            plot_pacf(self.df['pair_z_score'].dropna(),lags=25, ax=axs[1,1])
            axs[1,1].set_title('pacf pair_z_score')

            plt.show()

    #######################
    #### to be deprecated ####
    # def get_count_feature(self, rolling_window, threshold, plot = False, save_features = False):

    #     # negative countiing and rolling countingng
    #     self.df['RetClose'] = self.df['Close'].pct_change()
    #     self.df['roll_pos_counting'] = np.where(self.df['RetClose'].shift(1) > 0,1,0 )
    #     self.df['roll_pos_counting'] = self.df['roll_pos_counting'].rolling(window = rolling_window).sum()

    #     mean = self.df['roll_pos_counting'].mean()
    #     std = self.df['roll_pos_counting'].std()
    #     self.df['norm_counting'] =  (self.df['roll_pos_counting'] - mean )/std

    #     self.df['signal_up_roll_pos_counting'] = np.where((self.df['norm_counting'] > threshold),1,0)
    #     self.df['signal_low_roll_pos_counting'] = np.where((self.df['norm_counting'] < -threshold),1,0)

    #     if save_features:
    #         self.features.append('roll_pos_counting')
    #         self.signals.append('signal_up_roll_pos_counting')
    #         self.signals.append('signal_low_roll_pos_counting')
    #         self.settings_count_features = {'rolling_window':rolling_window, 'threshold':threshold}

    #     if plot:
    #         fig = plt.figure(figsize = (10,4))
    #         plt.plot(self.df['Date'],self.df.norm_counting)
    #         plt.axhline(y=threshold, color='grey', linestyle='--')
    #         plt.axhline(y=-threshold, color='grey', linestyle='--')
    #         plt.show()
    #######################

    def bidirect_count_feature(self, rolling_window, threshold, plot = False, save_features = False):
        """
        perform negative and positive return counting in a given rolling time window

        Parameters
        ----------
        rolling_window (int): window to apply to positive and negative returns
        threshold (float): alpha or z thrsholds for the normalized feature
        plot (boolean): True to display plot
        save_features (boolean): True to save feature configuration and feature names
        
        Returns
        -------
        None
        """
        feature_name = 'bidirect_counting'
        # negative countiing and rolling countingng
        self.df['RetClose'] = self.df['Close'].pct_change()
        self.df['roll_pos_counting'] = np.where(self.df['RetClose'].shift(1) > 0,1,0 )
        self.df['roll_pos_counting'] = self.df['roll_pos_counting'].rolling(window = rolling_window).sum()

        self.df['roll_neg_counting'] = np.where(self.df['RetClose'].shift(1) <= 0,1,0 )
        self.df['roll_neg_counting'] = self.df['roll_neg_counting'].rolling(window = rolling_window).sum()

        self.df[feature_name] = np.where(self.df['roll_pos_counting'] > self.df['roll_neg_counting'], self.df['roll_pos_counting'], -self.df['roll_neg_counting'])

        self.compute_clip_bands(feature_name,threshold)

        if save_features:
            self.log_features_standard(feature_name)
            self.settings_bidirect_count_features = {'rolling_window':rolling_window, 'threshold':threshold}

        if plot:
            fig = plt.figure(figsize = (10,4))
            plt.plot(self.df['Date'],self.df[f'norm_{feature_name}'])
            plt.plot(self.df['Date'],self.df[f'upper_{feature_name}'], linestyle='--')
            plt.plot(self.df['Date'],self.df[f'lower_{feature_name}'], linestyle='--')
            plt.show()

    #######################
    #### to be deprecated ####
    # def get_range_feature(self, window, up_threshold, low_threshold, plot = False, save_features = False):

    #     self.df["Range"] = self.df["High"] / self.df["Low"] - 1
    #     self.df['Avg_range'] = self.df['Range'].rolling(window = 5).mean()
    #     self.df['dist_range'] = self.df['Range'] - self.df['Avg_range']
    #     self.df['norm_dist_range'] = (self.df['dist_range'] - self.df['dist_range'].mean())/ self.df['dist_range'].std()

    #     mean_ = self.df['norm_dist_range'].mean()
    #     self.df[f'std_norm_dist_range'] = (self.df.sort_values("Date")["norm_dist_range"].transform(lambda x: x.rolling(window, min_periods=1).std()))

    #     self.df['up_bound_norm_dist_range'] = up_threshold*self.df['std_norm_dist_range'] + mean_
    #     self.df['low_bound_norm_dist_range'] = -low_threshold*self.df['std_norm_dist_range'] + mean_

    #     self.df['signal_up_dist_range'] = np.where(self.df['norm_dist_range'] > self.df['up_bound_norm_dist_range'],1,0 )
    #     self.df['signal_low_dist_range'] = np.where(self.df['norm_dist_range'] < self.df['low_bound_norm_dist_range'],1,0 )

    #     if save_features:
    #         self.features.append('dist_range')
    #         self.signals.append('signal_up_dist_range')
    #         self.signals.append('signal_low_dist_range')
    #         self.settings_price_range = {'window':window, 'up_threshold':up_threshold, 'low_threshold':low_threshold}

    #     if plot:
    #         fig, axs = plt.subplots(2, 2,figsize=(17,11))

    #         axs[0,0].plot(self.df['Range'])
    #         axs[0,0].set_title('range')

    #         axs[0,1].plot(self.df['Avg_range'])
    #         axs[0,1].set_title('Avg_range')

    #         axs[1,0].plot(self.df['up_bound_norm_dist_range'],color = 'grey', linestyle='--')
    #         axs[1,0].plot(self.df['low_bound_norm_dist_range'],color = 'grey', linestyle='--')
    #         axs[1,0].plot(self.df['norm_dist_range'])
    #         axs[1,0].set_title('norm_dist_range')
    #######################

    def get_relative_range_feature(self, window, threshold, plot = False, save_features = False):
        """
        perform relative spread of opening and closing price

        Parameters
        ----------
        window (int): window to apply to the feature
        threshold (float): alpha or z thrsholds for the normalized feature
        plot (boolean): True to display plot
        save_features (boolean): True to save feature configuration and feature names
        
        Returns
        -------
        None
        """
        feature_name = 'CO_Range'
        self.df[feature_name] = self.df["Close"] / self.df["Open"]-1
        self.df[f'norm_{feature_name}'] = (self.df[feature_name] - self.df[feature_name].mean())/ self.df[feature_name].std()

        mean_ = self.df[f'norm_{feature_name}'].mean()
        self.df[f'std_norm_{feature_name}'] = (self.df.sort_values("Date")[f'norm_{feature_name}'].transform(lambda x: x.rolling(window, min_periods=1).std()))

        self.df[f'up_bound_norm_{feature_name}'] = threshold*self.df[f'std_norm_{feature_name}'] + mean_
        self.df[f'low_bound_norm_{feature_name}'] = -threshold*self.df[f'std_norm_{feature_name}'] + mean_

        self.df[f'signal_up_{feature_name}'] = np.where(self.df[f'norm_{feature_name}'] > self.df[f'up_bound_norm_{feature_name}'],1,0 )
        self.df[f'signal_low_{feature_name}'] = np.where(self.df[f'norm_{feature_name}'] < self.df[f'low_bound_norm_{feature_name}'],1,0 )

        if save_features:
            self.log_features_standard(feature_name)
            self.settings_relative_price_range = {'window':window, 'threshold':threshold}

        if plot:
            fig, axs = plt.subplots(1, 2,figsize=(14,5))

            axs[0].plot(self.df[feature_name])
            axs[0].set_title(feature_name)

            axs[1].plot(self.df[f'up_bound_norm_{feature_name}'],color = 'grey', linestyle='--')
            axs[1].plot(self.df[f'low_bound_norm_{feature_name}'],color = 'grey', linestyle='--')
            axs[1].plot(self.df[f'norm_{feature_name}'])
            axs[1].set_title(f'norm_{feature_name}')

    #######################
    #### to be deprecated ####
    # def rsi_feature(self, window, lag_rsi_ret, threshold, plot = False, save_features = False):

    #     rsi = RSIIndicator(close = self.df['Close'], window = window).rsi()
    #     self.df['RSI'] = rsi
    #     self.df['RSI_ret'] = self.df['RSI']/self.df['RSI'].shift(lag_rsi_ret)

    #     mean = self.df['RSI_ret'].mean()
    #     std = self.df['RSI_ret'].std()
    #     self.df['norm_RSI_ret'] = (self.df['RSI_ret']-mean)/std
    #     self.df['signal_up_RSI_ret'] = np.where(self.df['norm_RSI_ret'] > threshold,1,0)
    #     self.df['signal_low_RSI_ret'] = np.where(self.df['norm_RSI_ret'] < -threshold,1,0)

    #     if save_features:
    #         self.features.append('RSI_ret')
    #         self.signals.append('signal_up_RSI_ret')
    #         self.signals.append('signal_low_RSI_ret')
    #         self.settings_rsi_feature= {'window':window, 'lag_rsi_ret':lag_rsi_ret, 'threshold':threshold}

    #     if plot:
    #         fig, axs = plt.subplots(1, 3,figsize=(17,5))

    #         axs[0].plot(self.df.norm_RSI_ret)
    #         axs[0].axhline(y=threshold, color='grey', linestyle='--')
    #         axs[0].axhline(y=-threshold, color='grey', linestyle='--')

    #         plot_acf(self.df['RSI_ret'].dropna(),lags=25,ax = axs[1])
    #         axs[1].set_title('acf RSI_ret')

    #         plot_pacf(self.df['RSI_ret'].dropna(),lags=25,ax = axs[2])
    #         axs[2].set_title('pacf RSI_ret')

    #         fig.show()
    #######################

    def rsi_feature_improved(self, window, threshold, plot = False, save_features = False):
        """
        perform relative strength index

        Parameters
        ----------
        window (int): window to apply to the feature
        threshold (float): alpha or z thrsholds for the normalized feature
        plot (boolean): True to display plot
        save_features (boolean): True to save feature configuration and feature names
        
        Returns
        -------
        None
        """
        feature_name = 'RSI'
        rsi = RSIIndicator(close = self.df['Close'], window = window).rsi()
        self.df[feature_name] = rsi.replace([np.inf, -np.inf], 0).fillna(method = 'ffill')
        self.compute_clip_bands(feature_name,threshold)

        if save_features:
            self.log_features_standard(feature_name)
            self.settings_rsi_feature_v2 = {'window':window, 'threshold':threshold}

        if plot:
            self.signal_plotter(feature_name)

    #######################
    #### to be deprecated ####
    # def days_features(self, window_day, limit, plot = False, save_features = False):

    #     self.df['dow'] = self.df.Date.dt.dayofweek
    #     self.df['dow'] = self.df['dow'].astype('str')

    #     self.df['target_mean_input'] = (self.df.sort_values("Date").groupby('dow')['roll_mean_log_return'].transform(lambda x: x.rolling(window_day, min_periods=1).mean()))

    #     mean = self.df['target_mean_input'].mean()
    #     std = self.df['target_mean_input'].std()

    #     self.df['norm_dow_input'] = (self.df['target_mean_input']-mean)/std
    #     mean_ = self.df['norm_dow_input'].mean()
    #     self.df['std_dow_input'] = self.df.sort_values("Date")["norm_dow_input"].rolling(50).std()

    #     self.df['up_dow_input'] = limit*self.df['std_dow_input'] + mean_
    #     self.df['low_dow_input'] = -limit*self.df['std_dow_input'] - mean_

    #     self.df['signal_up_target_mean_input'] = np.where(self.df['norm_dow_input'] > self.df['up_dow_input'],1,0)
    #     self.df['signal_low_target_mean_input'] = np.where(self.df['norm_dow_input'] < self.df['low_dow_input'],1,0)

    #     if save_features:

    #         self.features.append('target_mean_input')
    #         self.signals.append('signal_up_target_mean_input')
    #         self.signals.append('signal_low_target_mean_input')
    #         self.settings_days_features = {'window_day':window_day, 'limit':limit}

    #     if plot:
    #         fig, axs = plt.subplots(1, 3,figsize=(17,5))

    #         axs[0].plot(self.df['norm_dow_input'])
    #         axs[0].plot(self.df['up_dow_input'], linestyle='--')
    #         axs[0].plot(self.df['low_dow_input'], linestyle='--')

    #         plot_acf(self.df['norm_dow_input'].dropna(),lags=25,ax = axs[1])
    #         axs[1].set_title('acf day feature')

    #         plot_pacf(self.df['norm_dow_input'].dropna(),lags=25,ax = axs[2])
    #         axs[2].set_title('pacf day feature')

    #         fig.show()
    #######################

    def days_features_bands(self, window, threshold, plot = False, save_features = False):
        """
        compute mean returns for a given day of the week in a window scope per day 

        Parameters
        ----------
        window (int): window to apply to the feature
        threshold (float): alpha or z thrsholds for the normalized feature
        plot (boolean): True to display plot
        save_features (boolean): True to save feature configuration and feature names
        
        Returns
        -------
        None
        """
        self.df['dow'] = self.df.Date.dt.dayofweek
        self.df['dow'] = self.df['dow'].astype('str')

        feature_name = 'target_mean_dow'

        self.df[feature_name] = (self.df.sort_values("Date").groupby('dow')['roll_mean_log_return'].transform(lambda x: x.rolling(window, min_periods=1).mean()))

        self.compute_clip_bands(feature_name,threshold)

        if save_features:

            self.log_features_standard(feature_name)
            self.settings_days_features_v2 = {'window':window, 'threshold':threshold}

        if plot:
            self.signal_plotter(feature_name)

    #######################
    #### to be deprecated ####
    # def analysis_volume(self,lag_volume, threshold, window, plot = False, save_features = False):

    #     self.df['log_Volume'] = np.log(self.df['Volume'])
    #     self.df['ret_log_Volume'] = self.df['log_Volume'].pct_change(lag_volume)

    #     self.df['norm_ret_log_Volume'] = (self.df['ret_log_Volume'] - self.df['ret_log_Volume'].mean())/ self.df['ret_log_Volume'].std()
    #     mean_ = self.df['norm_ret_log_Volume'].mean()
    #     self.df[f'std_norm_ret_log_Volume'] = (self.df.sort_values("Date")["norm_ret_log_Volume"].transform(lambda x: x.rolling(window, min_periods=1).std()))

    #     self.df['up_bound_ret_log_Volume'] = threshold*self.df['std_norm_ret_log_Volume'] + mean_
    #     self.df['low_bound_ret_log_Volume'] = -threshold*self.df['std_norm_ret_log_Volume'] + mean_

    #     self.df['signal_up_ret_log_Volume'] = np.where(self.df['norm_ret_log_Volume'] > self.df['up_bound_ret_log_Volume'],1,0 )
    #     self.df['signal_low_ret_log_Volume'] = np.where(self.df['norm_ret_log_Volume'] < self.df['low_bound_ret_log_Volume'],1,0 )

    #     if save_features:
    #         self.features.append('ret_log_Volume')
    #         self.signals.append('signal_up_ret_log_Volume')
    #         self.signals.append('signal_low_ret_log_Volume')
    #         self.settings_volume_feature= {'lag_volume':lag_volume, 'threshold':threshold, 'window':window}
    #     if plot:
    #         fig, axs = plt.subplots(3, 2,figsize=(11,13))
    #         axs[0,0].plot(self.df.Date, self.df.Volume)
    #         axs[0,0].set_title('Volume')
    #         axs[0,1].plot(self.df.Date, self.df.log_Volume)
    #         axs[0,1].set_title('log Volume')

    #         plot_acf(self.df['log_Volume'].dropna(),lags=25, ax = axs[1,0])
    #         axs[1,0].set_title('acf log_Volume')
    #         plot_pacf(self.df['log_Volume'].dropna(),lags=25, ax = axs[1,1])
    #         axs[1,1].set_title('pacf log_Volume')

    #         plot_acf(self.df['ret_log_Volume'].dropna(),lags=25, ax = axs[2,0])
    #         axs[2,0].set_title('acf ret_log_Volume')
    #         plot_pacf(self.df['ret_log_Volume'].dropna(),lags=25, ax = axs[2,1])
    #         axs[2,1].set_title('pacf ret_log_Volume')

    #         plt.show()

    #         print('--------------------------------------------------------------')

    #         fig, axs = plt.subplots(1, 2,figsize=(10,4))

    #         axs[0].plot(self.df.Date, self.df.norm_ret_log_Volume)
    #         axs[0].plot(self.df.Date, self.df.up_bound_ret_log_Volume)
    #         axs[0].plot(self.df.Date, self.df.low_bound_ret_log_Volume)
    #         axs[0].set_title('norm_ret_log_Volume')

    #         axs[1].plot(self.df.Date, self.df.std_norm_ret_log_Volume)
    #         axs[1].set_title('std_norm_ret_log_Volume')

    #         plt.show()
    #######################

    def analysis_smooth_volume(self, window, threshold, plot = False, save_features = False):
        """
        compute feature of thrading volumes

        Parameters
        ----------
        window (int): window to apply to the feature
        threshold (float): alpha or z thrsholds for the normalized feature
        plot (boolean): True to display plot
        save_features (boolean): True to save feature configuration and feature names
        
        Returns
        -------
        None
        """
        feature_name = 'smooth_Volume'
        self.df[feature_name] = np.log(self.df['Volume'])
        # self.df[feature_name] = self.df['log_Volume'].rolling(window).mean()

        self.df[f'roll_mean_{feature_name}'] = self.df[feature_name].rolling(window).mean()
        self.df[f'roll_std_{feature_name}'] = self.df[feature_name].rolling(window).std()

        self.df[f'z_{feature_name}'] = (self.df[f'roll_mean_{feature_name}']- self.df[feature_name])/self.df[f'roll_std_{feature_name}']

        self.df[f'signal_low_{feature_name}'] = np.where( (self.df[f'z_{feature_name}'] < -threshold ), 1, 0)
        self.df[f'signal_up_{feature_name}'] = np.where( (self.df[f'z_{feature_name}'] > threshold ), 1, 0)

        if save_features:
            self.log_features_standard(feature_name)
            self.settings_smooth_volume = {'window':window, 'threshold':threshold}
        if plot:
            fig, axs = plt.subplots(2, 2,figsize=(11,6))
            axs[0,0].plot(self.df.Date, self.df.Volume)
            axs[0,0].set_title('Volume')
            axs[0,1].plot(self.df.Date, self.df.smooth_Volume)
            axs[0,1].set_title('log Volume')

            plot_acf(self.df['smooth_Volume'].dropna(),lags=25, ax = axs[1,0])
            axs[1,0].set_title('acf log_Volume')
            plot_pacf(self.df['smooth_Volume'].dropna(),lags=25, ax = axs[1,1])
            axs[1,1].set_title('pacf log_Volume')

            plt.show()

            print('--------------------------------------------------------------')

            fig, axs = plt.subplots(1,2,figsize=(10,4))

            axs[0].plot(self.df[f'{feature_name}'])
            axs[0].set_title(f'{feature_name}')

            axs[1].plot(self.df[f'z_{feature_name}'], linestyle='--')
            axs[1].set_title(f'z_{feature_name}')

            plt.show()

    def roc_feature(self, window, threshold, plot = False, save_features = False):
        """
        perform price rate of change

        Parameters
        ----------
        window (int): window to apply to the feature
        threshold (float): alpha or z thrsholds for the normalized feature
        plot (boolean): True to display plot
        save_features (boolean): True to save feature configuration and feature names
        
        Returns
        -------
        None
        """
        feature_name = 'ROC'
        roc = ROCIndicator(close = self.df['Close'], window = window).roc()
        self.df[feature_name] = roc.replace([np.inf, -np.inf], 0).fillna(method = 'ffill')
        self.compute_clip_bands(feature_name,threshold)

        if save_features:
            self.log_features_standard(feature_name)
            self.settings_roc_feature = {'window':window, 'threshold':threshold}
        if plot:
            self.signal_plotter(feature_name)

    def stoch_feature(self, window, smooth1, smooth2, threshold, plot = False, save_features = False):
        """
        perform stochastic oscilator RSI feature

        Parameters
        ----------
        window (int): window to apply to the feature
        smooth1 (int): smoothing parameter 1
        smooth2 (int): smoothing parameter 2
        threshold (float): alpha or z thrsholds for the normalized feature
        plot (boolean): True to display plot
        save_features (boolean): True to save feature configuration and feature names
        
        Returns
        -------
        None
        """
        feature_name = 'STOCH'
        stoch = StochRSIIndicator(close = self.df['Close'], window = window, smooth1=smooth1, smooth2=smooth2).stochrsi()
        self.df[feature_name] = stoch.replace([np.inf, -np.inf], 0).fillna(method = 'ffill')
        self.compute_clip_bands(feature_name,threshold)

        if save_features:
            self.log_features_standard(feature_name)
            self.settings_stoch_feature = {'window':window, 'smooth1':smooth1, 'smooth2':smooth2, 'threshold':threshold}
        if plot:
            self.signal_plotter(feature_name)

    def stochastic_feature(self, window, smooth, threshold, plot = False, save_features = False):
        """
        perform stochastic oscilator feature

        Parameters
        ----------
        window (int): window to apply to the feature
        smooth (int): smoothing parameter 
        threshold (float): alpha or z thrsholds for the normalized feature
        plot (boolean): True to display plot
        save_features (boolean): True to save feature configuration and feature names
        
        Returns
        -------
        None
        """
        feature_name = 'STOCHOSC'
        stochast = StochasticOscillator(close = self.df['Close'], high = self.df['High'], low = self.df['Low'], window = window,smooth_window=smooth).stoch()
        self.df[feature_name] = stochast.replace([np.inf, -np.inf], 0).fillna(method = 'ffill')
        self.compute_clip_bands(feature_name,threshold)

        if save_features:
            self.log_features_standard(feature_name)
            self.settings_stochastic_feature = {'window':window, 'smooth':smooth,'threshold':threshold}
        if plot:
            self.signal_plotter(feature_name)

    def william_feature(self, lbp, threshold, plot = False, save_features = False):
        """
        perfom fast stochastic oscilator or william indicator

        Parameters
        ----------
        lbp (int): look back parameter
        threshold (float): alpha or z thrsholds for the normalized feature
        plot (boolean): True to display plot
        save_features (boolean): True to save feature configuration and feature names
        
        Returns
        -------
        None
        """
        feature_name = 'WILL'
        will = WilliamsRIndicator(close = self.df['Close'], high = self.df['High'], low = self.df['Low'], lbp = lbp).williams_r()
        self.df[feature_name] = will.replace([np.inf, -np.inf], 0).fillna(method = 'ffill')
        self.compute_clip_bands(feature_name,threshold)

        if save_features:
            self.log_features_standard(feature_name)
            self.settings_william_feature = {'lbp':lbp,'threshold':threshold}
        if plot:
            self.signal_plotter(feature_name)

    def vortex_feature(self, window, threshold, plot = False, save_features = False):
        """
        perform vortex oscilator

        Parameters
        ----------
        window (int): window to apply to the feature
        threshold (float): alpha or z thrsholds for the normalized feature
        plot (boolean): True to display plot
        save_features (boolean): True to save feature configuration and feature names
        
        Returns
        -------
        None
        """
        feature_name = 'VORTEX'
        vortex = VortexIndicator(close = self.df['Close'], high = self.df['High'], low = self.df['Low'], window = window).vortex_indicator_diff()
        self.df[feature_name] = vortex.replace([np.inf, -np.inf], 0).fillna(method = 'ffill')
        self.compute_clip_bands(feature_name,threshold)

        if save_features:
            self.log_features_standard(feature_name)
            self.settings_vortex_feature = {'window':window, 'threshold':threshold}
        if plot:
            self.signal_plotter(feature_name)

    def minmax_pricefeature(self, type_func, window, distance = False, plot = False, save_features = False):
        """
        perform relative price/distance with respect to the min/max price in a given time scope

        Parameters
        ----------
        type_func (str): either min or max
        window (int): window scope
        distance (boolean): if true, get distance feature else relative feature
        save_features (boolean): True to save feature configuration and feature names
        
        Returns
        -------
        None
        """
        if type_func == 'min':
            self.df['Price_ref'] = self.df[['Open','High', 'Low','Close']].min(axis = 1)
        elif type_func == 'max':
            self.df['Price_ref'] = self.df[['Open','High', 'Low','Close']].max(axis = 1)

        init_shape = self.df.shape[0]
        df_date = self.df[['Date','Price_ref']].rename(columns = {'Date':'Date_ref'}).copy()
        
        self.df = self.df.rename(columns = {'Price_ref':'Price_to_use'})
        
        if type_func == 'min':
            self.df[f'window_price'] = (self.df.sort_values("Date")["Price_to_use"].transform(lambda x: x.rolling(window, min_periods=1).min()))
        elif type_func == 'max':
            self.df[f'window_price'] = (self.df.sort_values("Date")["Price_to_use"].transform(lambda x: x.rolling(window, min_periods=1).max()))
        
        
        self.df = self.df.merge(df_date, left_on = 'window_price', right_on = 'Price_ref', how = 'left')
        self.df['date_span'] = self.df['Date'] - self.df['Date_ref']
        
        self.df['RN'] = self.df.sort_values(['date_span'], ascending=False).groupby(['Date']).cumcount() + 1
        self.df = self.df[self.df['RN'] == 1]

        if distance:
            self.df[f'{type_func}_distance_to_price'] = pd.to_numeric(self.df['date_span'].dt.days, downcast='integer')

        if not distance:
            if type_func == 'min':
                self.df[f'{type_func}_relprice'] = self.df['Price_to_use']/self.df['window_price']-1
            
            if type_func == 'max':
                self.df[f'{type_func}_relprice'] = self.df['window_price']/self.df['Price_to_use']-1
        
        self.df = self.df.drop(columns = ['RN', 'date_span', 'Price_to_use', 'window_price', 'Date_ref','Price_ref'])
        
        end_shape = self.df.shape[0]

        if init_shape != end_shape:
            raise Exception("shapes are not the same")

        if save_features:
            if distance:
                self.features.append(f'{type_func}_distance_to_price')
                name_attr = f'{type_func}_distance'
            if not distance:
                self.features.append(f'{type_func}_relprice')
                name_attr = f'{type_func}_relprice'
                
            setattr(self,f'settings_{name_attr}_pricefeature' , {'type_func': type_func, 'window': window, 'distance': distance})

    def pair_index_feature(self, pair_symbol, feature_label, window, threshold, plot = False, save_features = False):
        """
        perform additional asset ROC feature, then a new feature is created in the main dataframe

        Parameters
        ----------
        pair_symbol (str): symbol of the asset to extract the data
        feature_label (str): name of the resulting feature
        window (int): window to apply to the feature
        threshold (float): alpha or z thrsholds for the normalized feature
        plot (boolean): True to display plot
        save_features (boolean): True to save feature configuration and feature names
        
        Returns
        -------
        None
        """
        self.pair_index = pair_symbol
        begin_date = self.today - relativedelta(days = self.n_days)
        begin_date_str = begin_date.strftime('%Y-%m-%d')

        if feature_label in self.df.columns:
            self.df = self.df.drop(columns = [feature_label])

        stock = yf.Ticker(self.pair_index)
        df = stock.history(period=self.data_window)
        df = df.sort_values('Date')
        df.reset_index(inplace=True)
        df['Date'] = pd.to_datetime(df['Date'], format='mixed',utc=True).dt.date
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[df.Date >= begin_date_str ]
        self.pair_index_df = df

        #### converting the same index ####
        dates_vector = self.df.Date.to_frame()
        self.pair_index_df = dates_vector.merge(self.pair_index_df, on ='Date',how = 'left')
        self.pair_index_df = self.pair_index_df.fillna(method = 'bfill')
        self.pair_index_df = self.pair_index_df.fillna(method = 'ffill')

        self.pair_index_df[feature_label] = ROCIndicator(close = self.pair_index_df['Close'], window = window).roc()
        df_to_merge = self.pair_index_df[['Date',feature_label]]
        self.df = self.df.merge(df_to_merge, on ='Date',how = 'left')

        ########
        self.compute_clip_bands(feature_label,threshold)

        if save_features:
            self.log_features_standard(feature_label)
            parameters = {feature_label:{'pair_symbol':pair_symbol, 'feature_label':feature_label, 'window':window,'threshold':threshold}}
            try:
                len(self.settings_pair_index_feature)
                print('existing')
                self.settings_pair_index_feature.append(parameters)
            except:
                print('creation')
                self.settings_pair_index_feature = list()
                self.settings_pair_index_feature.append(parameters)

        if plot:
            self.signal_plotter(feature_label)

    def produce_order_features(self, feature_name, save_features = False):
        """
        perform a feature that captures high and low values in an index. this is usefull to know duration/persistence of a signal

        Parameters
        ----------
        feature_name (str): name of the feature
        save_features (boolean): True to save feature configuration and feature names
        
        Returns
        -------
        None
        """
        signal_feature_name = f'discrete_signal_{feature_name}'
        order_feature_name = f'order_signal_{feature_name}'

        self.df[signal_feature_name] = np.where(
            self.df[f'signal_up_{feature_name}'] == 1,1,
            np.where(
                self.df[f'signal_low_{feature_name}'] == 1,-1,0
            )
        )

        ## indexing chains
        self.df[f'lag_{signal_feature_name}'] = self.df[signal_feature_name].shift(1)
        self.df['breack'] = np.where(self.df[f'lag_{signal_feature_name}'] != self.df[signal_feature_name],1,0)
        self.df["chain_id"] = self.df.groupby("breack")["Date"].rank(method="first", ascending=True)
        self.df["chain_id"] = np.where(self.df['breack'] == 1,self.df["chain_id"],np.nan)
        self.df["chain_id"] = self.df["chain_id"].fillna(method='ffill')
        self.df[order_feature_name] = self.df.groupby('chain_id')["Date"].rank(method="first", ascending=True)
        self.df[order_feature_name] = self.df[order_feature_name]*self.df[signal_feature_name]
        self.df = self.df.drop(columns = [f'lag_{signal_feature_name}', 'breack', "chain_id"])

        ## saving features
        if save_features:
            self.signals.append(signal_feature_name)
            self.signals.append(order_feature_name)

    def create_hmm_derived_features(self, lag_returns):
        """
        create features derived from hmm states features. Features are the index of the state, the duration of the state, chain raturn

        Parameters
        ----------
        lag_returns (int): lag paramter (not used)
        
        Returns
        -------
        None
        """
        self.df = self.df.sort_values('Date')
        ## indexing chains
        self.df['lag_hmm_feature'] = self.df['hmm_feature'].shift(1)
        self.df['breack'] = np.where(self.df['lag_hmm_feature'] != self.df['hmm_feature'],1,0)
        self.df["chain_id"] = self.df.groupby("breack")["Date"].rank(method="first", ascending=True)
        self.df["chain_id"] = np.where(self.df['breack'] == 1,self.df["chain_id"],np.nan)
        self.df["chain_id"] = self.df["chain_id"].fillna(method='ffill')
        self.df["hmm_chain_order"] = self.df.groupby('chain_id')["Date"].rank(method="first", ascending=True)

        ### returns using the windowsseeds
        self.df['lag_chain_close'] = self.df.sort_values(by=["Date"]).groupby(['chain_id'])['Close'].shift(lag_returns)
        self.df['chain_return'] = (self.df['Close']/self.df['lag_chain_close'] -1) * 100
        self.df = self.df.drop(columns = ['breack'])

    def cluster_hmm_analysis(self, n_clusters,features_hmm, test_data_size, seed, lag_returns_state=7, corr_threshold = 0.75, plot = False, save_features = False, model = False):
        """
        create or use a hmm model

        Parameters
        ----------
        n_clusters (int): number of clusters or states to calculate
        features_hmm (list): features to be considered in hmm model when training
        test_data_size (int): size of the test data. Note that the remaining is going to be used as training data
        seed (int): seed for the model inizialization
        lag_returns_state (int) : lags for returns of the state
        corr_threshold (float): correlation threshold for initial feature selection
        plot (boolean): True to display hmm states analysis
        save_features (boolean): True to save features and configurations
        model (obj): if provided, no model will be trainend and the provided model will be used to get hmm features
        
        Returns
        -------
        None
        """
        if not model:

            df_new = self.df
            data_train = df_new.iloc[:-test_data_size,:]
            data_test = df_new.iloc[-test_data_size:,:]

            th = trainer_hmm(data_train, features_hmm, n_clusters=n_clusters,corr_thrshold=corr_threshold, seed = seed)
            th.train()
            pipeline_hmm = th.hmm_model
            self.model_hmm = pipeline_hmm
            self.test_data_hmm = data_test

            ### first feature: the hidden state
            self.df['hmm_feature'] = self.model_hmm.predict(self.df)
            self.create_hmm_derived_features(lag_returns = lag_returns_state)

            ## completion

            hidden_states = pipeline_hmm.predict(data_train)
            map_ = {i:f'state_{i}' for i in range(n_clusters)}
            color_map = { i:DEFAULT_PLOTLY_COLORS[i] for i in range(n_clusters)}

            data_train['HMM'] = hidden_states
            data_train['HMM_state'] =  data_train['HMM'].map(map_)

            hidden_states = pipeline_hmm.predict(data_test)
            data_test['HMM'] = hidden_states
            data_test['HMM_state'] =  data_test['HMM'].map(map_)

        if model:
            self.df['hmm_feature'] = model.predict(self.df)
            self.create_hmm_derived_features(lag_returns = lag_returns_state)

        if save_features:
            self.features.append('hmm_feature')
            self.features.append('hmm_chain_order')
            self.settings_hmm = {'n_clusters':n_clusters,'features_hmm':features_hmm, 'test_data_size':test_data_size, 'seed':seed,'lag_returns_state':lag_returns_state, 'corr_threshold':corr_threshold }

        if plot:

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data_train['Date'], y=data_train['Close'], mode='lines',name = 'history', marker_color = 'grey'))
            for state in data_train['HMM_state'].unique():
                dfi = data_train[data_train['HMM_state'] == state]
                hmm_id = dfi['HMM'].unique()[0]
                fig.add_trace(go.Scatter(x=dfi['Date'], y=dfi['Close'], mode='markers',name = state, marker_color = color_map[hmm_id]))
            fig.update_layout(height=500, width=1200)
            fig.show()

            print('---------------------------------------------------------')

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data_test['Date'], y=data_test['Close'], mode='lines',name = 'history', marker_color = 'grey'))
            for state in data_test['HMM_state'].unique():
                dfi = data_test[data_test['HMM_state'] == state]
                hmm_id = dfi['HMM'].unique()[0]
                fig.add_trace(go.Scatter(x=dfi['Date'], y=dfi['Close'], mode='markers',name = state, marker_color = color_map[hmm_id]))
            fig.update_layout(height=500, width=1200)
            fig.show()

    def sharpe_ratio(self, return_series, n_trad_days = 255, rf = 0.01):
        """
        perform sharpe ratio of a given time series return

        Parameters
        ----------
        return_series (pd.series): time series of the returns
        n_trad_days (int): trading days to anualize returns
        rf (float): anual free risk rate
        
        Returns
        -------
        sharpe_ratio (float): sharpe ratio 
        """
        nsqrt = np.sqrt(n_trad_days)
        mean = return_series.mean() * n_trad_days
        sigma = return_series.std() * nsqrt
        sharpe_ratio = round((mean-rf)/sigma,2)
        return sharpe_ratio

    def treat_signal_strategy(self,test_data, strategy):
        """
        helper method that treats signals and converts signals to 1 or 0

        Parameters
        ----------
        test_data (pd.DataFrame): test data
        strategy (list): features to get the strategy
        
        Returns
        -------
        test_data (pd.DataFrame): test data with extra columns that are the strategy (main_signal)
        """
        hmm_states_list = [x for x in strategy if 'hmm_state_' in x]
        other_features = [x for x in strategy if x not in hmm_states_list]

        test_data['hmm_signal'] = 0
        test_data['features_signal'] = 0
        test_data['main_signal'] = 0

        ## hmm_feature
        if len(hmm_states_list) > 0:
            test_data['hmm_signal'] = test_data.loc[:,hmm_states_list].sum(axis=1)
            test_data['hmm_signal'] = np.where(test_data['hmm_signal'] > 0,1,0)

        ### other features
        if len(other_features) > 0:
            test_data['features_signal'] = test_data.loc[:,other_features].sum(axis=1)
            test_data['features_signal'] = np.where(test_data['features_signal'] == len(other_features),1,0)

        ## combined signals

        if len(hmm_states_list) > 0 and len(other_features) > 0:
            test_data['main_signal'] = np.where((test_data['features_signal'] == 1) & (test_data['hmm_signal'] == 1),1,0)

        elif len(hmm_states_list) > 0 and len(other_features) == 0:
            test_data['main_signal'] = np.where((test_data['features_signal'] == 0) & (test_data['hmm_signal'] == 1),1,0)

        elif len(hmm_states_list) == 0 and len(other_features) > 0:
            test_data['main_signal'] = np.where((test_data['features_signal'] == 1) & (test_data['hmm_signal'] == 0),1,0)

        return test_data

    def stategy_simulator(self, features, hmm_feature = True):
        """
        execute strategy and get some performance metrics like sharpe ratio, return. This method creates some new attributes

        Parameters
        ----------
        features (list): list of features to be tested as strategies
        hmm_feature (boolean): include hmm feature
        
        Returns
        -------
        None
        """
        columns_ = ['Date', 'Close','Open'] + features + ['HMM']
        states = list(self.df.hmm_feature.unique())
        states.sort()
        test_data = self.test_data_hmm[columns_]

        ## benchmark return
        test_data['lrets_bench'] = np.log(test_data['Close']/test_data['Close'].shift(1))
        test_data['bench_prod'] = test_data['lrets_bench'].cumsum()
        test_data['bench_prod_exp'] = np.exp(test_data['bench_prod']) - 1

        signal_feature_list = list()
        ## continous signals
        for feature in features:
            signal_name = f'signal_{feature}'
            test_data[signal_name] = np.where(test_data[feature]>0,1,0)
            signal_feature_list.append(signal_name)

        ## one hot encoding of states
        if hmm_feature:
            for state in states:
                state_name = f'hmm_state_{state}'
                test_data[state_name] = np.where(test_data['HMM'] == state,1,0)
                signal_feature_list.append(state_name)

        self.test_data_strategy = test_data

        ### combination of features

        signal_feature_list_combination = chain.from_iterable(combinations(signal_feature_list, r) for r in range(len(signal_feature_list)+1))
        signal_feature_list_combination = [list(x) for x in signal_feature_list_combination][1:]

        ### testing strategy

        ##### benchmark
        bench_sharpe = self.sharpe_ratio(test_data['bench_prod_exp'].values)
        bench_rets = round(test_data['bench_prod_exp'].values[-1]*100,1)

        benchmark = {
            'bench_rets':bench_rets,
            'bench_sharpe':bench_sharpe
        }

        returns_log = dict()

        for i,strategy in enumerate(signal_feature_list_combination):

            test_data = self.treat_signal_strategy(test_data, strategy)

            ## strategy return
            # test_data['lrets_strat'] = np.log(test_data['Open'].shift(-1)/test_data['Open']) * test_data['main_signal']
            test_data['lrets_strat'] = np.log(test_data['Close'].shift(-1)/test_data['Close']) * test_data['main_signal']
            test_data['lrets_prod'] = test_data['lrets_strat'].cumsum()
            test_data['strat_prod_exp'] = np.exp(test_data['lrets_prod']) - 1
            test_data = test_data.dropna(inplace = False)

            strat_rets = round(test_data['strat_prod_exp'].values[-1]*100,1)
            strat_sharpe = self.sharpe_ratio(test_data['strat_prod_exp'].values)

            returns_log[i] = {
                'strategy': strategy,
                'strat_rets':strat_rets,
                'strat_sharpe':strat_sharpe

            }
            df_returns_log = pd.DataFrame(returns_log).T.sort_values('strat_rets', ascending = False)

        self.strategy_log = df_returns_log
        self.best_strategy =  df_returns_log.iloc[0,:].strategy
        self.top_10_strategy = list(df_returns_log.iloc[0:10,:].strategy.values)

    def viz_strategy(self, strategy):
        """
        display analysis plot of a given strategy

        Parameters
        ----------
        strategy (list): list of features of the strategy
        
        Returns
        -------
        None
        """
        test_data = self.test_data_strategy

        test_data = self.treat_signal_strategy(test_data, strategy)

        ## strategy return
        # test_data['lrets_strat'] = np.log(test_data['Open'].shift(-1)/test_data['Open']) * test_data['main_signal']
        test_data['lrets_strat'] = np.log(test_data['Close'].shift(-1)/test_data['Close']) * test_data['main_signal']
        test_data['lrets_prod'] = test_data['lrets_strat'].cumsum()
        test_data['strat_prod_exp'] = np.exp(test_data['lrets_prod']) - 1
        test_data = test_data.dropna(inplace = False)

        bench_rets = round(test_data['bench_prod_exp'].values[-1]*100,1)
        strat_rets = round(test_data['strat_prod_exp'].values[-1]*100,1)

        bench_sharpe = self.sharpe_ratio(test_data['bench_prod_exp'].values)
        strat_sharpe = self.sharpe_ratio(test_data['strat_prod_exp'].values)

        print('----------------------------')
        print('strategy: ', strategy)
        print('----------------------------')
        print(f'returns benchmark {bench_rets}%')
        print(f'returns strategy {strat_rets}%')
        print('-----------------------------')
        print(f'sharpe benchmark {bench_sharpe}')
        print(f'sharpe strategy {strat_sharpe}')

        fig = plt.figure(figsize = (10,4))
        plt.plot(test_data['bench_prod_exp'], label= 'benchmark')
        plt.plot(test_data['strat_prod_exp'], label= 'strategy')
        plt.legend()
        plt.show()

    ### deprecated ############################
    def create_strategy(self, favourable_states):
        
        test_data = self.test_data_hmm
        # add MA signal
        test_data.loc[test_data[self.ma1_column] > test_data[self.ma2_column], 'MA_signal'] = 1
        test_data.loc[test_data[self.ma1_column] <= test_data[self.ma2_column], 'MA_signal'] = 0

        # add hnn signal

        test_data['HMM_signal'] =  np.where(test_data['HMM'].isin(favourable_states),1,0)

        ## combined signals
        test_data['main_signal'] = 0
        test_data.loc[(test_data['MA_signal'] == 1) & (test_data['HMM_signal'] == 1), 'main_signal'] = 1
        test_data['main_signal'] = test_data['main_signal'].shift(1)

        ## benchmark return
        test_data['lrets_bench'] = np.log(test_data['Close']/test_data['Close'].shift(1))
        test_data['bench_prod'] = test_data['lrets_bench'].cumsum()
        test_data['bench_prod_exp'] = np.exp(test_data['bench_prod']) - 1

        ## strategy return
        # test_data['lrets_strat'] = np.log(test_data['Open'].shift(-1)/test_data['Open']) * test_data['main_signal']
        test_data['lrets_strat'] = np.log(test_data['Close'].shift(-1)/test_data['Close']) * test_data['main_signal']
        test_data['lrets_prod'] = test_data['lrets_strat'].cumsum()
        test_data['strat_prod_exp'] = np.exp(test_data['lrets_prod']) - 1
        test_data.dropna(inplace = True)

        bench_rets = round(test_data['bench_prod_exp'].values[-1]*100,1)
        strat_rets = round(test_data['strat_prod_exp'].values[-1]*100,1)

        bench_sharpe = self.sharpe_ratio(test_data['bench_prod_exp'].values)
        strat_sharpe = self.sharpe_ratio(test_data['strat_prod_exp'].values)

        print(f'returns benchmark {bench_rets}%')
        print(f'returns strategy {strat_rets}%')
        print('-----------------------------')
        print(f'sharpe benchmark {bench_sharpe}')
        print(f'sharpe strategy {strat_sharpe}')

        fig = plt.figure(figsize = (10,4))
        plt.plot(test_data['bench_prod_exp'])
        plt.plot(test_data['strat_prod_exp'])
        self.settings_hmm_states = {'favourable_states':favourable_states}
    ################################################

    def deep_dive_analysis_hmm(self, test_data_size, split = 'train'):
        """
        display analysis plot hmm model

        Parameters
        ----------
        test_data_size (int): test data size, the remaining is the train data
        split (str): options (train or test). Split type to assess
        
        Returns
        -------
        None
        """
        if split == 'train':
            df = self.df.iloc[:-test_data_size,:]
        elif split == 'test':
            df = self.df.iloc[-test_data_size:,:]

        ## returns plot
        fig = px.box(df.sort_values('hmm_feature'), y = 'chain_return',x = 'hmm_feature', color = 'hmm_feature',
                    height=400, width=1000, title = 'returns chain hmm feature')
        fig.add_shape(type='line',x0=-0.5,y0=0,x1=max(df.hmm_feature)+0.5,y1=0,line=dict(color='grey',width=1),xref='x',yref='y')
        fig.show()
        print('--------------------------------------------------------------')
        ## time series plot
        fig = px.line(
            df.sort_values(['hmm_feature','hmm_chain_order']),
            x="hmm_chain_order", y="chain_return", color='chain_id',facet_col = 'hmm_feature', title = 'time series by state')
        fig.update_layout(showlegend=False)
        fig.update_xaxes(matches=None)
        fig.show()
        print('--------------------------------------------------------------')
        ### length plot
        df_agg =  df.groupby(['hmm_feature','chain_id'],as_index = False).agg(chain_lenght = ('hmm_chain_order','max'))
        fig = px.box(df_agg, y = 'chain_lenght', color = 'hmm_feature', height=400, width=1000, title = 'length chain hmm feature')
        fig.show()
        print('--------------------------------------------------------------')
        ## transition plot
        fig, ax = plt.subplots()
        sns.heatmap((self.model_hmm['hmm'].transmat_)*100, annot=True, ax = ax)
        ax.set_title('Transition Matrix')
        ax.set_xlabel('State To')
        ax.set_ylabel('State From')
        fig.show()
        print('--------------------------------------------------------------')
        del df

    def get_targets(self, steps):
        """
        produce regression target return taking future prices

        Parameters
        ----------
        steps (int): number of lags and steps for future returns
        
        Returns
        -------
        None
        """
        self.targets = list()
        self.target = list()
        columns = list()
        for i in range(1,steps+1):
            self.df[f'target_{i}'] = self.df.log_return.shift(-i)
            self.targets.append(f'target_{i}')
            columns.append(f'target_{i}')

        self.df[f'mean_target'] = self.df[columns].mean(axis=1)
        self.target.append(f'mean_target')
        self.settings_target_lasts = {'steps':steps, 'type':'regression'}

    def get_categorical_targets(self, horizon, flor_loss, top_gain):
        """
        produce binary target return taking future prices. it produce two targets, one for high returns and another for low returns

        Parameters
        ----------
        horizon (int): number of lags and steps for future returns
        flor_loss (float): min loss return
        top_gain (float): max gain return

        Returns
        -------
        None
        """
        self.target = list()
        self.targets = list()
        columns = list()

        ## loops
        for i in range(1,horizon+1):
            self.df[f'target_{i}'] = self.df.High.shift(-i)
            self.df[f'target_{i}'] = (self.df[f'target_{i}']/self.df.Open-1)*100

            self.df[f'target_{i}'] = np.where(self.df[f'target_{i}'] >= top_gain,1,0)
            columns.append(f'target_{i}')
        self.df[f'target_up'] = self.df[columns].sum(axis=1)
        self.df[f'target_up'] = np.where(self.df[f'target_up'] >=1,1,0 )
        self.df = self.df.drop(columns = columns)

        for i in range(1,horizon+1):
            self.df[f'target_{i}'] = self.df.Low.shift(-i)
            self.df[f'target_{i}'] = (self.df[f'target_{i}']/self.df.Open-1)*100

            self.df[f'target_{i}'] = np.where(self.df[f'target_{i}'] <= flor_loss,1,0)
            columns.append(f'target_{i}')
        self.df[f'target_down'] = self.df[columns].sum(axis=1)
        self.df[f'target_down'] = np.where(self.df[f'target_down'] >= 1,1,0 )
        self.df = self.df.drop(columns = columns)

        self.targets.append('target_up')
        self.targets.append('target_down')

        self.settings_target_lasts = {'horizon':horizon, 'flor_loss':flor_loss, 'top_gain':top_gain, 'type': 'classification'}

    def get_configurations(self,test_data_size =250, val_data_size = 250, model_type = False):
        """
        produce configuration dictionary that were saved in the feature generation methods if save_features was activated

        Parameters
        ----------
        test_data_size (int): test data size
        val_data_size (int): validation data size 
        model_type (str): model type, options: 'Forecaster','Classifier'

        Returns
        -------
        None
        """
        self.settings = {
            'features':list(set(self.features)),
            'signals' :list(set(self.signals)),
            'test_data_size': test_data_size,
            'val_data_size': val_data_size,
            'settings' : {
                'general' : self.settings_general,
                'volatility' : self.settings_volatility,
                'outlier': self.settings_outlier,
            }
        }

        if model_type in ['Forecaster','Classifier']:

            target_list = list(set(self.targets))
            target_list.sort()
            self.settings['model_type'] = model_type
            self.settings['target'] = list(set(self.target))
            self.settings['targets'] = target_list

        ## for now this is hard coded
        feature_list = ['spread_ma','relative_spread_ma','pair_feature','count_features','bidirect_count_features','price_range','relative_price_range','rsi_feature',
                        'rsi_feature_v2', 'days_features','days_features_v2', 'volume_feature','smooth_volume', 'roc_feature', 'stoch_feature', 'stochastic_feature',
                        'william_feature', 'vortex_feature', 'pair_index_feature','hmm',
                        'min_distance_pricefeature', 'min_relprice_pricefeature', 'max_distance_pricefeature','max_relprice_pricefeature'
                        ]

        for feature in feature_list:
            try:
                self.settings['settings'][feature] = getattr(self, f'settings_{feature}')
            except:
                pass
        try:
            self.settings['settings']['target_lasts'] = self.settings_target_lasts
        except:
            pass

        try:
            self.settings['settings']['strategies'] = {
                'best_strategy':self.best_strategy,
                'top_10_strategies': self.top_10_strategy
            }
        except:
            pass

class produce_model:
    """
    Class that produces a machine learning model in a scikit-learn pipeline wrapper.

    Attributes
    ----------
    data  : pd.DataFrame
        symbol of the asset
    X_train : pd.DataFrame
    y_train : pd.Series
    X_test : pd.DataFrame
    y_test : pd.Series
    X_val : pd.DataFrame
    y_val : pd.Series
    pipeline : obj
        trained pipeline that includes a ml model
    features_to_model: list
        features in end step of the pipeline

    Methods
    -------
    preprocess(test_data_size=int, target=str, val_data_size=int):
        prepare data, split train, test, validation data and X and Y
    get_sample(x=pd.DataFrame, sample=int, max_=int):
        sample data
    """
    def __init__(self,data):
        """
        Initialize object

        Parameters
        ----------
        data (pd.DataFrame): data

        Returns
        -------
        None
        """
        self.data = data.copy()

    def preprocess(self, test_data_size, target, val_data_size = False):
        """
        prepare data, split train, test, validation data and X and Y

        Parameters
        ----------
        test_data_size (int): test data size
        target (str): target column
        val_data_size (int): validation data size

        Returns
        -------
        None
        """
        train_data, test_data = self.data.iloc[:-test_data_size,:].dropna() , self.data.iloc[-test_data_size:,:].dropna()

        if val_data_size:
            train_data, val_data = train_data.iloc[:-val_data_size,:], train_data.iloc[-val_data_size:,:]

        self.test_data = test_data

        X_train, y_train = train_data.iloc[0:,1:], train_data[target]
        X_test, y_test = test_data.iloc[0:,1:], test_data[target]
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        if val_data_size:
            X_val, y_val = val_data.iloc[0:,1:], val_data[target]
            self.X_val = X_val
            self.y_val = y_val

    def get_sample(self, x, sample, max_=900):
        """
        sample data

        Parameters
        ----------
        x (pd.DataFrame): input data
        sample (int): sample size
        max_ (int): max sample

        Returns
        -------
        sample (float): sample size
        """
        length = len(x)
        if length > max_:
            return 1.0
        else:
            return sample

    def train_model(self, pipe, model, cv_ = False):
        """
        train pipeline

        Parameters
        ----------
        pipe (obj): pipeline object
        model (obj): model object
        cv_ (obj): cross validation procedure

        Returns
        -------
        sample (float): sample size
        """
        self.model = model
        self.pipe_transform = pipe
        self.pipeline = Pipeline([('pipe_transform',self.pipe_transform), ('model',self.model)])
        self.pipeline.fit(self.X_train, self.y_train)
        self.features_to_model = self.pipeline[:-1].transform(self.X_train).columns

class hmm_feature_selector():
    """
    class that is going to train hmm models to perform feature selection

    Attributes
    ----------
    data  : pd.DataFrame
        symbol of the asset
    n_clusters : int
        number of clusters to search
    init_features_hmm : list
        list of features to consider in the search
    test_data_size :int
        test data size, meaning that the remaining is going to be used as training data
    select_n_features : int
        number of features to select
    n_trials : int
        total number of trials per combination
    limit_search : int
        limit number of combinations
    default_benchmark_sd : float
        default value to bias standard deviation
    t_threshold : float
        alpha or z threshold
    pipeline_hmm: obj
        pipeline object of the hmm model
    features_used_in_model:list
        features in model
    train_model(features_hmm=list):
        train hmm model
    feature_combinations: list
        list of combination of features
    mean_relevance: float
        relevance score of the model
    best_features: list
        list of best performing features

    Methods
    -------
    split_data():
        split data in train and test
    train_model(features_hmm=list):
        train hmm model
    feature_list_generator():
        perform combination of features
    get_error():
        get error or score of a given model using relevance score
    execute_selector():
        select the best combination of features
    """
    def __init__(self, data, n_clusters, init_features_hmm, test_data_size, select_n_features, n_trials = 1,limit_search = False, default_benchmark_sd = 0.00003, t_threshold = 2):
        """
        Initialize object

        Parameters
        ----------
        data (pd.DataFrame): data
        n_clusters (int): number of clusters to search
        init_features_hmm (list): list of features to consider in the search
        test_data_siz:(int:  test data size, meaning that the remaining is going to be used as training data
        select_n_features (int): number of features to select
        n_trials (int): total number of trials per combination
        limit_search (int): limit number of combinations
        default_benchmark_sd (float): default value to bias standard deviation
        t_threshold (float): alpha or z threshold

        Returns
        -------
        None
        """
        self.data = data.copy()
        self.n_clusters = n_clusters
        self.init_features_hmm = init_features_hmm
        self.test_data_size = test_data_size
        self.select_n_features = select_n_features
        self.n_trials = n_trials
        self.limit_search= limit_search
        self.default_benchmark_sd = default_benchmark_sd
        self.t_threshold = t_threshold

    def split_data(self):
        """
        split data in train and test

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.data_train = self.data.iloc[:-self.test_data_size,:]
        self.data_test = self.data.iloc[-self.test_data_size:,:]

    def train_model(self,features_hmm):
        """
        train hmm model

        Parameters
        ----------
        features_hmm (list): list of features to be selected in the model

        Returns
        -------
        None
        """
        pipeline_hmm = Pipeline([
                ('selector', FeatureSelector(columns=features_hmm)),
                ('fillna', MeanMedianImputer(imputation_method='median',variables=features_hmm)),
                ('hmm',GaussianHMM(n_components =  self.n_clusters, covariance_type = 'full'))
                ])

        self.pipeline_hmm = pipeline_hmm.fit(self.data_train)
        self.features_used_in_model = features_hmm

    def feature_list_generator(self):
        """
        perform combination of features

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        feature_combinations = set(list(combinations(self.init_features_hmm, self.select_n_features)))
        feature_combinations = list(map(list, feature_combinations))

        self.feature_combinations = feature_combinations

    def get_error(self):
        """
        get error or score of a given model using relevance score

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.data_train_ = self.data_train.copy()

        self.data_train_['hmm_feature'] = self.pipeline_hmm.predict(self.data_train_)
        self.data_train_ = self.data_train_[['Date','hmm_feature','Close']].sort_values('Date')

        ## indexing chains
        self.data_train_['lag_hmm_feature'] = self.data_train_['hmm_feature'].shift(1)
        self.data_train_['breack'] = np.where(self.data_train_['lag_hmm_feature'] != self.data_train_['hmm_feature'],1,0)
        self.data_train_["chain_id"] = self.data_train_.groupby("breack")["Date"].rank(method="first", ascending=True)
        self.data_train_["chain_id"] = np.where(self.data_train_['breack'] == 1,self.data_train_["chain_id"],np.nan)
        self.data_train_["chain_id"] = self.data_train_["chain_id"].fillna(method='ffill')
        self.data_train_["hmm_chain_order"] = self.data_train_.groupby('chain_id')["Date"].rank(method="first", ascending=True)

        ### returns using the first element in a chain
        self.data_train_['first'] = np.where(self.data_train_['hmm_chain_order'] == 1, self.data_train_['Close'], np.nan)
        self.data_train_['first'] = self.data_train_.sort_values('Date')['first'].fillna(method='ffill')
        self.data_train_['chain_return'] = (self.data_train_['Close']/self.data_train_['first'] -1) * 100

        self.data_train_ = self.data_train_.drop(columns = ['first'])

        mean_relevance, cluster_returns, number_relevant_states = states_relevance_score(self.data_train_)
        self.mean_relevance = mean_relevance

    def execute_selector(self):
        """
        select the best combination of features

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.split_data()
        self.feature_list_generator()
        maxi = -1
        print(f'it is expected {len(self.feature_combinations)} combinations')
        feature_results = dict()

        if self.limit_search:
            print(f' taking just {self.limit_search} combinations')
            maxi = self.limit_search

        for i,features_hmm in enumerate(self.feature_combinations[0:maxi]):

            feature_results[f'group_{i}'] = {
                'features':list(features_hmm),
                'relevances':list()
            }

            for _ in range(self.n_trials):
                try:
                    self.train_model(features_hmm)
                    self.get_error()
                    feature_results[f'group_{i}']['relevances'].append(self.mean_relevance)
                except:
                    print('error')
            feature_results[f'group_{i}']['mean relevance'] = np.mean(feature_results[f'group_{i}']['relevances'])
        self.feature_results = feature_results
        self.best_features = pd.DataFrame(self.feature_results).T.sort_values('mean relevance').iloc[-1,:].features

class analyse_index(stock_eda_panel):
    """
    class that is going to train hmm models to perform feature selection

    Attributes
    ----------
    data_index : pd.DataFrame
         name of the index
    indexes: list
        list of indexes
    asset : str
         name of the asset
    n_obs : int
         number of rows to extract
    lag : int
         lag to apply
    data_window : str
         5y 10y 15y
    show_plot : bool
         If True, show plots
    save_path : str
         local path for saving e.g r'C:/path/to/the/file/'
    save_aws : str
         remote key in s3 bucket path e.g. 'path/to/file/'
    aws_credentials : dict
         dict with the aws credentials
    merger_df : pd.DataFrame
        dataframe with the index and asset data
    states_result = dict
        betas and correlation score results

    Methods
    -------
    process_data():
        using stock_eda_panel, get data and merge data
    plot_betas(sample_size=int, offset=int, subsample_ts=int):
        display beta analysis plot
    get_betas(subsample_ts=int)
        get general beta and last sample beta, correlation score is included too
    """
    def __init__(self, index_data, asset, n_obs, lag, data_window = '5y', show_plot = False, save_path = False, save_aws = False, aws_credentials = False, return_fig = False):
        """
        Initialize object

        Parameters
        ----------
        index_data (pd.DataFrame or str): index data dataframe or index string
        asset (str): name of the asset
        n_obs (int): number of rows to extract
        lag (int): lag to apply
        data_window (str): 5y 10y 15y
        show_plot (bool): If True, show plots
        save_path (str): local path for saving e.g r'C:/path/to/the/file/'
        save_aws (str): remote key in s3 bucket path e.g. 'path/to/file/'
        aws_credentials (dict): dict with the aws credentials

        Returns
        -------
        None
        """

        
        if type(index_data) != str:
            index_data['Date'] = pd.to_datetime(index_data['Date'])
            self.index_data = index_data
            self.indexes = [ x for x in list(index_data.columns) if x != 'Date']
        else:
            self.indexes = [index_data]
            
        self.index_data = index_data
        self.asset = asset
        self.n_obs = n_obs
        self.data_window = data_window
        self.lag = lag

        self.show_plot = show_plot
        self.return_fig = return_fig
        self.save_path = save_path
        self.save_aws = save_aws

    def process_data(self):
        """
        using stock_eda_panel, get data and merge data
    
        Parameters
        ----------
        None
    
        Returns
        -------
        None
        """
        asset =  stock_eda_panel(self.asset, self.n_obs, data_window=self.data_window)
        asset.get_data()
        df = asset.df[['Date','Close']]
        
        if type(self.index_data) != str:
            df_merge = df.merge(self.index_data, on = ['Date'], how = 'left').sort_values('Date')
            
        else:
            indx =  stock_eda_panel(self.index_data, self.n_obs, data_window=self.data_window)
            indx.get_data()
            indx_df = indx.df[['Date','Close']].rename(columns = {'Close':self.index_data})
            df_merge = df.merge(indx_df, on = ['Date'], how = 'left').sort_values('Date')
            
        for colx in ['Close'] + self.indexes:
            df_merge[f'{colx}_pct'] = df_merge[colx]/df_merge[colx].shift(self.lag) - 1
            
        df_merge.dropna(inplace = True)
        self.merger_df = df_merge.rename(columns = {'Close_pct': 'asset_return'})

    def plot_betas(self,sample_size, offset, subsample_ts =False, index = False):
        """
        display beta analysis plot

        Parameters
        ----------
        sample_size (int): number of days or window size to calculate beta
        offset (int): overlap between windows
        subsample_ts (int): subsample size of data 

        Returns
        -------
        None
        """
        if (type(self.index_data) == str) & (index != False):
            raise Exception("No need of index argument")
        else:
            index = self.indexes[0]
            
        index_pct = f'{index}_pct'
        ### ploting analysis
        figure, ax = plt.subplot_mosaic(
            [["scatter_total", "scatter_sample",'ts','ts']],
            layout="constrained",
            figsize=(18, 5)
        )

        ax['scatter_total'].scatter(self.merger_df.asset_return, self.merger_df[index_pct])
    
        huber_regr = HuberRegressor(fit_intercept = True)
        huber_regr.fit(self.merger_df.asset_return.values.reshape(-1,1), self.merger_df[index_pct].values.reshape(-1,1))
        b, a = huber_regr.coef_[0], huber_regr.intercept_
    
        # b, a = np.polyfit(self.merger_df.asset_return, self.merger_df[index_pct], 1)
        ax['scatter_total'].plot(self.merger_df.asset_return, b*self.merger_df.asset_return+a, color='red')

        ax['ts'].plot(self.merger_df.Date, self.merger_df.Close, color = 'grey', alpha = 0.3)

        if subsample_ts:
            self.merger_df = self.merger_df.iloc[-subsample_ts:,:].dropna()

        for i in range(0,len(self.merger_df)-sample_size,offset):

            merger_ = self.merger_df.sort_values('Date', ascending = False).iloc[i:i+sample_size,:]
            x = merger_[index_pct]
            y = merger_.asset_return
            # b, a = np.polyfit(x,y, 1)
            huber_regr = HuberRegressor(fit_intercept = True)
            huber_regr.fit(x.values.reshape(-1,1), y.values.reshape(-1,1))
            b, a = huber_regr.coef_[0], huber_regr.intercept_
            
            normalize = mcolors.Normalize(vmin=-1, vmax=1)
            colormap = cm.jet

            ax['scatter_sample'].plot(x, y,'o', color = 'blue', alpha = 0.1)
            ax['scatter_sample'].plot(x, b*x+a, color=colormap(normalize(b)))
            ax['scatter_sample'].set_xlim(-0.06, 0.06)
            ax['scatter_sample'].set_ylim(-0.06, 0.06)

            plot = ax['ts'].scatter(merger_.Date, merger_.Close, color=colormap(normalize(b)), s = 10)

        scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
        scalarmappaple.set_array(x)

        plt.title(f'{self.asset} using index: {index}')
        plt.colorbar(scalarmappaple)

        if self.show_plot:
            plt.show()
            
        if self.save_path:
            result_plot_name = f'market_best_fit.png'
            figure.savefig(self.save_path+result_plot_name)

        if self.save_path and self.save_aws:
            # upload_file_to_aws(bucket = 'VIRGO_BUCKET', key = f'market_plots/{self.asset}/'+result_plot_name,input_path = self.save_path+result_plot_name)
            upload_file_to_aws(bucket = 'VIRGO_BUCKET', key = self.save_aws + result_plot_name, input_path = self.save_path + result_plot_name, aws_credentials = self.aws_credentials)
            
        if not self.show_plot:
            plt.close() 
            
        if self.return_fig:
            return figure
            
    def get_betas(self,subsample_ts=False):
        """
        get general beta and last sample beta, correlation score is included too
    
        Parameters
        ----------
        subsample_ts (int): subsample size of data 
    
        Returns
        -------
        None
        """
        result = list()
        for index in self.indexes:
            
            index_pct = f'{index}_pct'
            huber_regr = HuberRegressor(fit_intercept = True)
            huber_regr.fit(self.merger_df.asset_return.values.reshape(-1,1), self.merger_df[index_pct].values.reshape(-1,1))
            general_beta, a = huber_regr.coef_[0], huber_regr.intercept_
            general_r = stats.mstats.pearsonr(self.merger_df.asset_return, self.merger_df[index])[0]

            dict_res = {
                    'index':index,
                    'general_beta':general_beta,
                    'general_r':general_r,
                }
    
            if subsample_ts:
                tmp_df = self.merger_df.iloc[-subsample_ts:,:].dropna()
                huber_regr = HuberRegressor(fit_intercept = True)
                huber_regr.fit(tmp_df.asset_return.values.reshape(-1,1), tmp_df[index_pct].values.reshape(-1,1))
                sample_beta, a = huber_regr.coef_[0], huber_regr.intercept_
                sample_r = stats.mstats.pearsonr(tmp_df.asset_return, tmp_df[index])[0]
                dict_res['sample_beta'] = sample_beta
                dict_res['sample_r'] = sample_r
                
            result.append(dict_res)
    
        self.states_result = result


def get_relevant_beta(data_market, ticket_name,  show_plot = True, save_path = False, save_aws = False, aws_credentials = False):
    '''
    select relevant beta result data of a given asset

            Parameters:
                    data_market (pd.DataFrame): dataframe of the market results 
                    ticket_name (str): name of the asset
                    show_plot (bool): If tru, plot results
                    save_path (str): local path for saving e.g r'C:/path/to/the/file/'
                    save_aws (str):  remote key in s3 bucket path e.g. 'path/to/file/'
                    aws_credentials (dict): dict of the aws credentials

            Returns:
                    selection (pd.DataFrame): dataframe of the most relevant beta
    '''
    all_betas = data_market[data_market.asset == ticket_name].sort_values('general_r', ascending = False)
    all_betas['gen_r2'] = all_betas.general_r ** 2
    all_betas['sampl_r2'] = all_betas.sample_r ** 2
    selection = all_betas.sort_values('gen_r2',ascending =False).head(2).sort_values('sampl_r2',ascending =False).head(1).drop(columns = ['gen_r2','sampl_r2'])

    if show_plot:
        print(selection)
    if save_path:
        result_plot_name = f'market_best_fit.csv'
        selection.to_csv(save_path+result_plot_name)

    if save_path and save_aws:
        # upload_file_to_aws(bucket = 'VIRGO_BUCKET', key = f'market_plots/{ticket_name}/'+result_plot_name,input_path = save_path+result_plot_name)
        upload_file_to_aws(bucket = 'VIRGO_BUCKET', key = save_aws + result_plot_name, input_path = save_path + result_plot_name, aws_credentials = aws_credentials)
    return selection
