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

from .aws_utils import upload_file_to_aws

import logging

class InverseHyperbolicSine(BaseEstimator, TransformerMixin):
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
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.columns]
    
class signal_combiner(BaseEstimator, TransformerMixin):
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
    
    def __init__(self, stock_code, n_days, data_window = '5y'):
        self.stock_code = stock_code
        self.n_days = n_days
        self.today = datetime.date.today()
        self.features = list()
        self.signals = list()
        self.data_window = data_window
        
    def augmented_dickey_fuller_statistics(self,time_series, label):
        result = adfuller(time_series.dropna().values)
        print('p-value: {} for the series {}'.format(round(result[1],6), label))
        
    def get_data(self):
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
        
        print('---------------------------------------------------------------------------')
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
            fig, axs = plt.subplots(2, 1,figsize=(15,8))

            axs[0].hist(self.df['z_log_return'],bins = 100 )
            axs[0].axvline(l1, color='r', linestyle='--')
            axs[0].axvline(-l1, color='r', linestyle='--')
            axs[0].axvline(l2, color='green', linestyle='--')
            axs[0].axvline(-l2, color='green', linestyle='--')

            axs[1].plot(self.df['Date'],self.df['z_log_return'])
            axs[1].plot(self.df['Date'],self.df['low_outlier'], linestyle='--')
            axs[1].plot(self.df['Date'],self.df['up_outlier'], linestyle='--')

            plt.show()
    
    def analysis_roll_mean_log_returns(self, lags, plot = False):

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
    
        self.df[f'norm_{feature_name}'] =  (self.df[feature_name] - self.df[feature_name].mean())/self.df[feature_name].std()
        mean_ = self.df[f'norm_{feature_name}'].mean()

        self.df[f'up_rollstd_{feature_name}'] = self.df.sort_values("Date")[f'norm_{feature_name}'].clip(0,100).rolling(50).std()
        self.df[f'low_rollstd_{feature_name}'] = self.df.sort_values("Date")[f'norm_{feature_name}'].clip(-100,0).rolling(50).std()

        self.df[f'upper_{feature_name}'] = threshold*self.df[f'up_rollstd_{feature_name}'] + mean_
        self.df[f'lower_{feature_name}'] = -threshold*self.df[f'low_rollstd_{feature_name}'] + mean_

        self.df[f'signal_low_{feature_name}'] = np.where( (self.df[f'norm_{feature_name}'] < self.df[f'lower_{feature_name}'] ), 1, 0)
        self.df[f'signal_up_{feature_name}'] = np.where( (self.df[f'norm_{feature_name}'] > self.df[f'upper_{feature_name}'] ), 1, 0)

    def signal_plotter(self, feature_name):
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
        self.features.append(feature_name)
        self.signals.append(f'signal_up_{feature_name}')
        self.signals.append(f'signal_low_{feature_name}')
    
    #######################
    #### to be deprecated ####
    def spread_MA(self, ma1, ma2, limit = 1.95, plot = False, save_features = False):

        self.df[f'MA_{ma1}'] = (self.df.sort_values("Date")["Close"].transform(lambda x: x.rolling(ma1, min_periods=1).mean()))
        self.df[f'MA_{ma2}'] = (self.df.sort_values("Date")["Close"].transform(lambda x: x.rolling(ma2, min_periods=1).mean()))

        self.ma1_column = f'MA_{ma1}'
        self.ma2_column = f'MA_{ma2}'
        self.df['MA_spread'] = self.df[f'MA_{ma1}'] - self.df[f'MA_{ma2}']

        self.df['norm_MA_spread'] =  (self.df['MA_spread'] - self.df['MA_spread'].mean())/self.df['MA_spread'].std()
        mean_ = self.df['norm_MA_spread'].mean()
        self.df['rollstd_MA_spread'] = self.df.sort_values("Date")["norm_MA_spread"].rolling(50).std()

        self.df['upper_MA_spread'] = limit*self.df['rollstd_MA_spread'] + mean_
        self.df['lower_MA_spread'] = -limit*self.df['rollstd_MA_spread'] + mean_

        self.df['signal_low_MA_spread'] = np.where( (self.df['norm_MA_spread'] < self.df['lower_MA_spread'] ), 1, 0)
        self.df['signal_up_MA_spread'] = np.where( (self.df['norm_MA_spread'] > self.df['upper_MA_spread'] ), 1, 0)
        
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
            self.features.append('MA_spread')
            self.signals.append('signal_low_MA_spread')
            self.signals.append('signal_up_MA_spread')
            self.settings_spread_ma = {'ma1':ma1, 'ma2':ma2, 'limit':limit}  
            
        if plot:

            fig, axs = plt.subplots(1, 3,figsize=(21,4))

            axs[0].plot(self.df['Date'],self.df['norm_MA_spread'])
            axs[0].plot(self.df['Date'],self.df['upper_MA_spread'], linestyle='--')
            axs[0].plot(self.df['Date'],self.df['lower_MA_spread'], linestyle='--')
            axs[0].set_title('MA_spread series')

            plot_acf(self.df['MA_spread'].dropna(),lags=25, ax=axs[1])
            axs[1].set_title('acf MA_spread series')

            plot_pacf(self.df['MA_spread'].dropna(),lags=25, ax=axs[2])
            axs[2].set_title('acf MA_spread series')
            plt.show()
    ##################################################
    
    def relative_spread_MA(self, ma1, ma2, threshold = 1.95, plot = False, save_features = False):
    
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
    def get_count_feature(self, rolling_window, threshold, plot = False, save_features = False):

        # negative countiing and rolling countingng
        self.df['RetClose'] = self.df['Close'].pct_change()
        self.df['roll_pos_counting'] = np.where(self.df['RetClose'].shift(1) > 0,1,0 )
        self.df['roll_pos_counting'] = self.df['roll_pos_counting'].rolling(window = rolling_window).sum()

        mean = self.df['roll_pos_counting'].mean()
        std = self.df['roll_pos_counting'].std()
        self.df['norm_counting'] =  (self.df['roll_pos_counting'] - mean )/std

        self.df['signal_up_roll_pos_counting'] = np.where((self.df['norm_counting'] > threshold),1,0)
        self.df['signal_low_roll_pos_counting'] = np.where((self.df['norm_counting'] < -threshold),1,0)
        
        if save_features:
            self.features.append('roll_pos_counting')
            self.signals.append('signal_up_roll_pos_counting')
            self.signals.append('signal_low_roll_pos_counting')
            self.settings_count_features = {'rolling_window':rolling_window, 'threshold':threshold}  
            
        if plot:
            fig = plt.figure(figsize = (10,4))
            plt.plot(self.df['Date'],self.df.norm_counting)
            plt.axhline(y=threshold, color='grey', linestyle='--')
            plt.axhline(y=-threshold, color='grey', linestyle='--')
            plt.show()
    #######################
    
    def bidirect_count_feature(self, rolling_window, threshold, plot = False, save_features = False):
    
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
    def get_range_feature(self, window, up_threshold, low_threshold, plot = False, save_features = False):

        self.df["Range"] = self.df["High"] / self.df["Low"] - 1
        self.df['Avg_range'] = self.df['Range'].rolling(window = 5).mean()
        self.df['dist_range'] = self.df['Range'] - self.df['Avg_range']
        self.df['norm_dist_range'] = (self.df['dist_range'] - self.df['dist_range'].mean())/ self.df['dist_range'].std()

        mean_ = self.df['norm_dist_range'].mean()
        self.df[f'std_norm_dist_range'] = (self.df.sort_values("Date")["norm_dist_range"].transform(lambda x: x.rolling(window, min_periods=1).std()))

        self.df['up_bound_norm_dist_range'] = up_threshold*self.df['std_norm_dist_range'] + mean_
        self.df['low_bound_norm_dist_range'] = -low_threshold*self.df['std_norm_dist_range'] + mean_

        self.df['signal_up_dist_range'] = np.where(self.df['norm_dist_range'] > self.df['up_bound_norm_dist_range'],1,0 )
        self.df['signal_low_dist_range'] = np.where(self.df['norm_dist_range'] < self.df['low_bound_norm_dist_range'],1,0 )
        
        if save_features:
            self.features.append('dist_range')
            self.signals.append('signal_up_dist_range')
            self.signals.append('signal_low_dist_range')
            self.settings_price_range = {'window':window, 'up_threshold':up_threshold, 'low_threshold':low_threshold} 

        if plot:
            fig, axs = plt.subplots(2, 2,figsize=(17,11))

            axs[0,0].plot(self.df['Range'])
            axs[0,0].set_title('range')

            axs[0,1].plot(self.df['Avg_range'])
            axs[0,1].set_title('Avg_range')

            axs[1,0].plot(self.df['up_bound_norm_dist_range'],color = 'grey', linestyle='--')
            axs[1,0].plot(self.df['low_bound_norm_dist_range'],color = 'grey', linestyle='--')
            axs[1,0].plot(self.df['norm_dist_range'])
            axs[1,0].set_title('norm_dist_range')
    #######################
    
    def get_relative_range_feature(self, window, threshold, plot = False, save_features = False):
    
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
    def rsi_feature(self, window, lag_rsi_ret, threshold, plot = False, save_features = False):

        rsi = RSIIndicator(close = self.df['Close'], window = window).rsi()
        self.df['RSI'] = rsi 
        self.df['RSI_ret'] = self.df['RSI']/self.df['RSI'].shift(lag_rsi_ret)

        mean = self.df['RSI_ret'].mean()
        std = self.df['RSI_ret'].std()
        self.df['norm_RSI_ret'] = (self.df['RSI_ret']-mean)/std
        self.df['signal_up_RSI_ret'] = np.where(self.df['norm_RSI_ret'] > threshold,1,0)
        self.df['signal_low_RSI_ret'] = np.where(self.df['norm_RSI_ret'] < -threshold,1,0)

        if save_features:
            self.features.append('RSI_ret')
            self.signals.append('signal_up_RSI_ret')
            self.signals.append('signal_low_RSI_ret')
            self.settings_rsi_feature= {'window':window, 'lag_rsi_ret':lag_rsi_ret, 'threshold':threshold}

        if plot:
            fig, axs = plt.subplots(1, 3,figsize=(17,5))

            axs[0].plot(self.df.norm_RSI_ret)
            axs[0].axhline(y=threshold, color='grey', linestyle='--')
            axs[0].axhline(y=-threshold, color='grey', linestyle='--')

            plot_acf(self.df['RSI_ret'].dropna(),lags=25,ax = axs[1])
            axs[1].set_title('acf RSI_ret')

            plot_pacf(self.df['RSI_ret'].dropna(),lags=25,ax = axs[2])
            axs[2].set_title('pacf RSI_ret')

            fig.show()
    #######################
    
    def rsi_feature_improved(self, window, threshold, plot = False, save_features = False):
        feature_name = 'RSI'
        rsi = RSIIndicator(close = self.df['Close'], window = window).rsi()
        self.df[feature_name] = rsi 
        self.compute_clip_bands(feature_name,threshold)

        if save_features:
            self.log_features_standard(feature_name)
            self.settings_rsi_feature_v2 = {'window':window, 'threshold':threshold}

        if plot:
            self.signal_plotter(feature_name)
            
    #######################
    #### to be deprecated ####
    def days_features(self, window_day, limit, plot = False, save_features = False):

        self.df['dow'] = self.df.Date.dt.dayofweek
        self.df['dow'] = self.df['dow'].astype('str')

        self.df['target_mean_input'] = (self.df.sort_values("Date").groupby('dow')['roll_mean_log_return'].transform(lambda x: x.rolling(window_day, min_periods=1).mean()))

        mean = self.df['target_mean_input'].mean()
        std = self.df['target_mean_input'].std()

        self.df['norm_dow_input'] = (self.df['target_mean_input']-mean)/std
        mean_ = self.df['norm_dow_input'].mean()
        self.df['std_dow_input'] = self.df.sort_values("Date")["norm_dow_input"].rolling(50).std()

        self.df['up_dow_input'] = limit*self.df['std_dow_input'] + mean_
        self.df['low_dow_input'] = -limit*self.df['std_dow_input'] - mean_

        self.df['signal_up_target_mean_input'] = np.where(self.df['norm_dow_input'] > self.df['up_dow_input'],1,0)
        self.df['signal_low_target_mean_input'] = np.where(self.df['norm_dow_input'] < self.df['low_dow_input'],1,0)

        if save_features:

            self.features.append('target_mean_input')
            self.signals.append('signal_up_target_mean_input')
            self.signals.append('signal_low_target_mean_input')
            self.settings_days_features = {'window_day':window_day, 'limit':limit}

        if plot:
            fig, axs = plt.subplots(1, 3,figsize=(17,5))

            axs[0].plot(self.df['norm_dow_input']) 
            axs[0].plot(self.df['up_dow_input'], linestyle='--')
            axs[0].plot(self.df['low_dow_input'], linestyle='--')

            plot_acf(self.df['norm_dow_input'].dropna(),lags=25,ax = axs[1])
            axs[1].set_title('acf day feature')

            plot_pacf(self.df['norm_dow_input'].dropna(),lags=25,ax = axs[2])
            axs[2].set_title('pacf day feature')

            fig.show()
    #######################
    
    def days_features_bands(self, window, threshold, plot = False, save_features = False):

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
    def analysis_volume(self,lag_volume, threshold, window, plot = False, save_features = False):
    
        self.df['log_Volume'] = np.log(self.df['Volume'])
        self.df['ret_log_Volume'] = self.df['log_Volume'].pct_change(lag_volume)

        self.df['norm_ret_log_Volume'] = (self.df['ret_log_Volume'] - self.df['ret_log_Volume'].mean())/ self.df['ret_log_Volume'].std()
        mean_ = self.df['norm_ret_log_Volume'].mean()
        self.df[f'std_norm_ret_log_Volume'] = (self.df.sort_values("Date")["norm_ret_log_Volume"].transform(lambda x: x.rolling(window, min_periods=1).std()))

        self.df['up_bound_ret_log_Volume'] = threshold*self.df['std_norm_ret_log_Volume'] + mean_
        self.df['low_bound_ret_log_Volume'] = -threshold*self.df['std_norm_ret_log_Volume'] + mean_

        self.df['signal_up_ret_log_Volume'] = np.where(self.df['norm_ret_log_Volume'] > self.df['up_bound_ret_log_Volume'],1,0 )
        self.df['signal_low_ret_log_Volume'] = np.where(self.df['norm_ret_log_Volume'] < self.df['low_bound_ret_log_Volume'],1,0 )

        if save_features:
            self.features.append('ret_log_Volume')
            self.signals.append('signal_up_ret_log_Volume')
            self.signals.append('signal_low_ret_log_Volume')
            self.settings_volume_feature= {'lag_volume':lag_volume, 'threshold':threshold, 'window':window}
        if plot:
            fig, axs = plt.subplots(3, 2,figsize=(11,13))
            axs[0,0].plot(self.df.Date, self.df.Volume)
            axs[0,0].set_title('Volume')
            axs[0,1].plot(self.df.Date, self.df.log_Volume)
            axs[0,1].set_title('log Volume')

            plot_acf(self.df['log_Volume'].dropna(),lags=25, ax = axs[1,0])
            axs[1,0].set_title('acf log_Volume')
            plot_pacf(self.df['log_Volume'].dropna(),lags=25, ax = axs[1,1])
            axs[1,1].set_title('pacf log_Volume')

            plot_acf(self.df['ret_log_Volume'].dropna(),lags=25, ax = axs[2,0])
            axs[2,0].set_title('acf ret_log_Volume')
            plot_pacf(self.df['ret_log_Volume'].dropna(),lags=25, ax = axs[2,1])
            axs[2,1].set_title('pacf ret_log_Volume')

            plt.show()

            print('--------------------------------------------------------------')

            fig, axs = plt.subplots(1, 2,figsize=(10,4))

            axs[0].plot(self.df.Date, self.df.norm_ret_log_Volume)
            axs[0].plot(self.df.Date, self.df.up_bound_ret_log_Volume)
            axs[0].plot(self.df.Date, self.df.low_bound_ret_log_Volume)
            axs[0].set_title('norm_ret_log_Volume')

            axs[1].plot(self.df.Date, self.df.std_norm_ret_log_Volume)
            axs[1].set_title('std_norm_ret_log_Volume')

            plt.show()
    #######################
    
    def analysis_smooth_volume(self, window, threshold, plot = False, save_features = False):
    
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
        feature_name = 'ROC'
        roc = ROCIndicator(close = self.df['Close'], window = window).roc()
        self.df[feature_name] = roc 
        self.compute_clip_bands(feature_name,threshold)

        if save_features:
            self.log_features_standard(feature_name)
            self.settings_roc_feature = {'window':window, 'threshold':threshold}
        if plot:
            self.signal_plotter(feature_name)
    
    def stoch_feature(self, window, smooth1, smooth2, threshold, plot = False, save_features = False):
        feature_name = 'STOCH'
        stoch = StochRSIIndicator(close = self.df['Close'], window = window, smooth1=smooth1, smooth2=smooth2).stochrsi()
        self.df[feature_name] = stoch
        self.compute_clip_bands(feature_name,threshold)

        if save_features:
            self.log_features_standard(feature_name)
            self.settings_stoch_feature = {'window':window, 'smooth1':smooth1, 'smooth2':smooth2, 'threshold':threshold}
        if plot:
            self.signal_plotter(feature_name)

    def stochastic_feature(self, window, smooth, threshold, plot = False, save_features = False):
        feature_name = 'STOCHOSC'
        stochast = StochasticOscillator(close = self.df['Close'], high = self.df['High'], low = self.df['Low'], window = window,smooth_window=smooth).stoch()
        self.df[feature_name] = stochast
        self.compute_clip_bands(feature_name,threshold)

        if save_features:
            self.log_features_standard(feature_name)
            self.settings_stochastic_feature = {'window':window, 'smooth':smooth,'threshold':threshold}
        if plot:
            self.signal_plotter(feature_name)

    def william_feature(self, lbp, threshold, plot = False, save_features = False):
        feature_name = 'WILL'
        will = WilliamsRIndicator(close = self.df['Close'], high = self.df['High'], low = self.df['Low'], lbp = lbp).williams_r() 
        self.df[feature_name] = will
        self.compute_clip_bands(feature_name,threshold)

        if save_features:
            self.log_features_standard(feature_name)
            self.settings_william_feature = {'lbp':lbp,'threshold':threshold}
        if plot:
            self.signal_plotter(feature_name)

    def vortex_feature(self, window, threshold, plot = False, save_features = False):
        feature_name = 'VORTEX'
        vortex = VortexIndicator(close = self.df['Close'], high = self.df['High'], low = self.df['Low'], window = window).vortex_indicator_diff()
        self.df[feature_name] = vortex
        self.compute_clip_bands(feature_name,threshold)

        if save_features:
            self.log_features_standard(feature_name)
            self.settings_vortex_feature = {'window':window, 'threshold':threshold}
        if plot:
            self.signal_plotter(feature_name)

    def pair_index_feature(self, pair_symbol, feature_label, window, threshold, plot = False, save_features = False):
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
        
        self.df = self.df.sort_values('Date')
        ## indexing chains
        self.df['lag_hmm_feature'] = self.df['hmm_feature'].shift(1)
        self.df['breack'] = np.where(self.df['lag_hmm_feature'] != self.df['hmm_feature'],1,0)
        self.df["chain_id"] = self.df.groupby("breack")["Date"].rank(method="first", ascending=True)
        self.df["chain_id"] = np.where(self.df['breack'] == 1,self.df["chain_id"],np.nan)
        self.df["chain_id"] = self.df["chain_id"].fillna(method='ffill')
        self.df["hmm_chain_order"] = self.df.groupby('chain_id')["Date"].rank(method="first", ascending=True)
        
        ### returns using the first element in a chain
        self.df['first'] = np.where(self.df['hmm_chain_order'] == 1, self.df['Close'], np.nan)
        self.df['first'] = self.df.sort_values('Date')['first'].fillna(method='ffill')
        self.df['chain_return'] = (self.df['Close']/self.df['first'] -1) * 100

        self.df = self.df.drop(columns = ['breack','first'])

    def cluster_hmm_analysis(self, n_clusters,features_hmm, test_data_size, seed, lag_returns_state=7, plot = False, save_features = False, model = False):
        if not model:
            
            df_new = self.df
            pipeline_hmm = Pipeline([
                ('selector', FeatureSelector(columns=features_hmm)),
                ('fillna', MeanMedianImputer(imputation_method='median',variables=features_hmm)),
                ('hmm',GaussianHMM(n_components =  n_clusters, covariance_type = 'full', random_state = seed))
                ])
            data_train = df_new.iloc[:-test_data_size,:]
            data_test = df_new.iloc[-test_data_size:,:]

            pipeline_hmm.fit(data_train)

            self.model_hmm = pipeline_hmm
            self.test_data_hmm = data_test
            
            ### first feature: the hidden state
            self.df['hmm_feature'] = self.model_hmm.predict(self.df)
            self.create_hmm_derived_features(lag_returns = lag_returns_state)
        
        if model:
            self.df['hmm_feature'] = model.predict(self.df)
            self.create_hmm_derived_features(lag_returns = lag_returns_state)
            
        if save_features:
            self.features.append('hmm_feature')
            self.features.append('hmm_chain_order')
            self.settings_hmm = {'n_clusters':n_clusters,'features_hmm':features_hmm, 'test_data_size':test_data_size, 'seed':seed,'lag_returns_state':lag_returns_state }

        if plot:

            hidden_states = pipeline_hmm.predict(data_train)
            map_ = {i:f'state_{i}' for i in range(n_clusters)}
            color_map = { i:DEFAULT_PLOTLY_COLORS[i] for i in range(n_clusters)}

            data_train['HMM'] = hidden_states
            data_train['HMM_state'] =  data_train['HMM'].map(map_)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data_train['Date'], y=data_train['Close'], mode='lines',name = 'history', marker_color = 'grey'))
            for state in data_train['HMM_state'].unique():
                dfi = data_train[data_train['HMM_state'] == state]
                hmm_id = dfi['HMM'].unique()[0]
                fig.add_trace(go.Scatter(x=dfi['Date'], y=dfi['Close'], mode='markers',name = state, marker_color = color_map[hmm_id]))
            fig.update_layout(height=500, width=1200)
            fig.show()

            print('---------------------------------------------------------')
        
            hidden_states = pipeline_hmm.predict(data_test)
            data_test['HMM'] = hidden_states
            data_test['HMM_state'] =  data_test['HMM'].map(map_)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data_test['Date'], y=data_test['Close'], mode='lines',name = 'history', marker_color = 'grey'))
            for state in data_test['HMM_state'].unique():
                dfi = data_test[data_test['HMM_state'] == state]
                hmm_id = dfi['HMM'].unique()[0]
                fig.add_trace(go.Scatter(x=dfi['Date'], y=dfi['Close'], mode='markers',name = state, marker_color = color_map[hmm_id]))
            fig.update_layout(height=500, width=1200)
            fig.show()

    def sharpe_ratio(self, return_series, n_trad_days = 255, rf = 0.01):
        nsqrt = np.sqrt(n_trad_days)
        mean = return_series.mean() * n_trad_days
        sigma = return_series.std() * nsqrt
        sharpe_ratio = round((mean-rf)/sigma,2)
        return sharpe_ratio
    
    def treat_signal_strategy(self,test_data, strategy):
    
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
                        'william_feature', 'vortex_feature', 'pair_index_feature','hmm']

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
    def __init__(self,data):
        self.data = data.copy()
    
    def preprocess(self, test_data_size, target, val_data_size = False):
        
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
        length = len(x)
        if length > max_:
            return 1.0
        else:
            return sample
    
    def train_model(self, pipe, model, cv_ = False):
        self.model = model
        self.pipe_transform = pipe
        self.pipeline = Pipeline([('pipe_transform',self.pipe_transform), ('model',self.model)])
        self.features_to_model = self.pipe_transform.fit_transform(self.X_train).columns
        self.pipeline.fit(self.X_train, self.y_train)
        
        
class hmm_feature_selector():
    
    def __init__(self, data, n_clusters, init_features_hmm, test_data_size, select_n_features, n_trials = 1,limit_search = False, default_benchmark_sd = 0.00003, t_threshold = 2):
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
        
        self.data_train = self.data.iloc[:-self.test_data_size,:]
        self.data_test = self.data.iloc[-self.test_data_size:,:]
        
    def train_model(self,features_hmm):
        pipeline_hmm = Pipeline([
                ('selector', FeatureSelector(columns=features_hmm)),
                ('fillna', MeanMedianImputer(imputation_method='median',variables=features_hmm)),
                ('hmm',GaussianHMM(n_components =  self.n_clusters, covariance_type = 'full'))
                ])
        
        self.pipeline_hmm = pipeline_hmm.fit(self.data_train)
        self.features_used_in_model = features_hmm
        
    def feature_list_generator(self):
        
        feature_combinations = set(list(combinations(self.init_features_hmm, self.select_n_features)))
        feature_combinations = list(map(list, feature_combinations))
        
        self.feature_combinations = feature_combinations
        
    def get_error(self):
        
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
        
class signal_analyser_object:
    
    def __init__(self, data,symbol_name, show_plot = True, save_path = False, save_aws = False, aws_credentials = False, return_fig = False):
        """
        data: pandas df
        symbol_name: str name of the asset
        show_plot: bool
        save_path: str local path for saving e.g r'C:/path/to/the/file/'
        save_aws: str remote key in s3 bucket path e.g. 'path/to/file/'
        aws_credentials: dict
        return_fig: boolean return the image function as result
        """
        self.data = data.copy()
        self.ticket_name = symbol_name
        self.show_plot = show_plot
        self.save_path = save_path
        self.save_aws = save_aws
        self.aws_credentials = aws_credentials
        self.return_fig = return_fig

    def signal_analyser(self, test_size, feature_name, days_list, threshold = 0.05,verbose = False, signal_position = False):
        data = self.data
        self.feature_name = feature_name
        up_signal, low_signal= f'signal_up_{feature_name}', f'signal_low_{feature_name}'
        features_base = ['Date', up_signal, low_signal, 'Close']

        df = data[features_base].sort_values('Date').iloc[0:-test_size,:]
        returns_list = list()

        for days in days_list:

            feature_ = f'return_{days}d'
            df[feature_] = (df['Close'].shift(-days)/df['Close']-1)*100
            returns_list.append(feature_)

        df['signal_type'] = np.where(
            df[up_signal] == 1, 
            'up', 
            np.where(
                df[low_signal] == 1, 
                'down',
                None
            )
        )
        df = df[~df.signal_type.isna()]
        # df['Date'] = df.index
        df['lag_Date'] = df['Date'].shift(1)
        df['span'] = (pd.to_datetime(df['Date']) - pd.to_datetime(df['lag_Date'])).dt.days - 1
        df['break'] = np.where(df['span'] > 3, 1, 0)
        df['break'] = np.where(df['span'].isna(), 1, df['break'])

        df['chain_id'] = df.sort_values(['Date']).groupby(['break']).cumcount() + 1
        df['chain_id'] = np.where(df['break'] == 1, df['chain_id'], np.nan )
        df['chain_id'] = df['chain_id'].fillna(method = 'ffill')

        df['internal_rn'] = df.sort_values(['Date']).groupby(['chain_id']).cumcount() + 1
        df['inv_internal_rn'] = df.sort_values(['Date'],ascending = False).groupby(['chain_id']).cumcount() + 1

        df['first_in_chain'] = np.where(df['internal_rn'] == 1, True, False)
        df['last_in_chain'] = np.where(df['inv_internal_rn'] == 1, True, False)

        df = df.drop(columns = ['break','span','lag_Date','inv_internal_rn']).sort_values('Date')
        self.df_signal = df
        
        n_signals_up = len(list(df[df.signal_type == 'up'].chain_id.unique()))
        n_signals_down = len(list(df[df.signal_type == 'down'].chain_id.unique()))
        p_scores = list()
        medians_down = list()
        validations = list()
        if not signal_position: ### for now it is based on the last signal on a chain
            df_melt = df[df.last_in_chain == True].melt(id_vars=['signal_type'], value_vars=returns_list, var_name='time', value_name='value')
            df_melt = df_melt.dropna()

        for evalx in returns_list:

            sample1 = df_melt[(df_melt.time == evalx) & (df_melt.signal_type == 'up')].value.values
            sample2 = df_melt[(df_melt.time == evalx) & (df_melt.signal_type == 'down')].value.values
            pvalue = stats.ttest_ind(sample1, sample2).pvalue
            median_down = np.median(sample2)
            median_up = np.median(sample1) 
            validations.append(median_up < 0)
            validations.append(median_down > 0)
            p_scores.append(pvalue)
            medians_down.append(median_down)
        self.df_melt = df_melt
        null_ho_eval = threshold > np.mean(p_scores)
        mean_median_return = np.median(medians_down)  ## end metric
        median_signal_type_eval = validations.count(validations[0]) == len(validations)

        if verbose:
            print('number of signal up:',n_signals_up)
            print('number of signal down:',n_signals_down)
            print('reject ho: ', null_ho_eval)
            print('mean median:', mean_median_return)
            print('all validations: ', median_signal_type_eval)

        # if median_signal_type_eval == True and null_ho_eval == True:
        if null_ho_eval == True:
            if verbose:
                print('success evals')
            self.mean_median_return = mean_median_return
        else:
            self.mean_median_return = np.nan

        df2 = df.copy()
        df2 = df2[df2.last_in_chain == True]


        df2['lagdate'] = df2.Date.shift(1)
        df2['span'] = (pd.to_datetime(df2['Date']) - pd.to_datetime(df2['lagdate'])).dt.days

        fig, axs = plt.subplots(1, 3, figsize = (15,5))

        sns.boxplot(data=df2, y="span",ax = axs[0])
        axs[0].set_title('span between last signals')
        del df2
        sns.boxplot(data=df[df.last_in_chain == True], y="internal_rn",ax = axs[1])
        axs[1].set_title('signal duration distribution')
        sns.boxplot(data=df_melt, x="time", y="value", hue="signal_type",ax = axs[2])
        axs[2].axhline(y=0, color='grey', linestyle='--')
        axs[2].set_title('signal type expected returns distribution at different time lapses')
            
        if self.show_plot:
            plt.show()
            
        if self.save_path:
            result_plot_name = f'signals_strategy_distribution_{feature_name}.png'
            fig.savefig(self.save_path+result_plot_name)
            # pickle.dump(axs, open(self.save_path+result_plot_name, 'wb'))

        if self.save_path and self.save_aws:
            # upload_file_to_aws(bucket = 'VIRGO_BUCKET', key = f'market_plots/{self.ticket_name}/'+result_plot_name, input_path = self.save_path+result_plot_name)
            upload_file_to_aws(bucket = 'VIRGO_BUCKET', key = self.save_aws + result_plot_name, input_path = self.save_path + result_plot_name, aws_credentials = self.aws_credentials)
        if not self.show_plot:
            plt.close()

        del df

        if self.return_fig:
            return fig
        
    def create_backtest_signal(self,days_strategy, test_size, feature_name):
        asset_1 = 'Close'
        up_signal, low_signal= f'signal_up_{feature_name}', f'signal_low_{feature_name}'
        df1 = self.data.iloc[-test_size:,:].copy()
        df2 = df1.copy()
        df2['signal_type'] = np.where(
                    df2[up_signal] == 1, 
                    'up', 
                    np.where(
                        df2[low_signal] == 1, 
                        'down',
                        None
                    )
                )
        df2 = df2[~df2.signal_type.isna()]
        # df2['Date_'] = df2.index
        df2['lag_Date'] = df2['Date'].shift(1)
        df2['span'] = (pd.to_datetime(df2['Date']) - pd.to_datetime(df2['lag_Date'])).dt.days - 1
        df2['break'] = np.where(df2['span'] > 3, 1, 0)
        df2['break'] = np.where(df2['span'].isna(), 1, df2['break'])

        df2['chain_id'] = df2.sort_values(['Date']).groupby(['break']).cumcount() + 1
        df2['chain_id'] = np.where(df2['break'] == 1, df2['chain_id'], np.nan )
        df2['chain_id'] = df2['chain_id'].fillna(method = 'ffill')

        df2['internal_rn'] = df2.sort_values(['Date']).groupby(['chain_id']).cumcount() + 1
        df2['inv_internal_rn'] = df2.sort_values(['Date'],ascending = False).groupby(['chain_id']).cumcount() + 1

        df2['first_in_chain'] = np.where(df2['internal_rn'] == 1, True, False)
        df2['last_in_chain'] = np.where(df2['inv_internal_rn'] == 1, True, False)

        df2 = df2.drop(columns = ['break','span','lag_Date','inv_internal_rn']).sort_values('Date')

        df2 = df2[(df2.last_in_chain == True) & (df2.signal_type == 'down')][['last_in_chain']]
        dft = df1.merge(df2,how = 'left',left_index=True, right_index=True )

        dft['chain_id'] = dft.sort_values(['Date']).groupby(['last_in_chain']).cumcount() + 1
        dft['chain_id'] = np.where(dft['last_in_chain'] == True, dft['chain_id'], np.nan )
        dft['chain_id'] = dft['chain_id'].fillna(method = 'ffill')

        dft['internal_rn'] = dft.sort_values(['Date']).groupby(['chain_id']).cumcount() + 1
        dft['flag'] = np.where(dft['internal_rn'] < days_strategy, 1,0)

        dft['lrets_bench'] = np.log(dft[asset_1]/dft[asset_1].shift(1))
        dft['bench_prod'] = dft['lrets_bench'].cumsum()
        dft['bench_prod_exp'] = np.exp(dft['bench_prod']) - 1

        dft['lrets_strat'] = np.log(dft[asset_1].shift(-1)/dft[asset_1]) * dft['flag']
        dft['lrets_strat'] = np.where(dft['lrets_strat'].isna(),-0.0,dft['lrets_strat'])
        dft['lrets_prod'] = dft['lrets_strat'].cumsum()
        dft['strat_prod_exp'] = np.exp(dft['lrets_prod']) - 1

        bench_rets = round(dft['bench_prod_exp'].values[-1]*100,1)
        strat_rets = round(dft['strat_prod_exp'].values[-1]*100,1)

        
        message1 = f'returns benchmark {bench_rets}%'
        message2 = f'returns strategy {strat_rets}%'
        
        messages = {
            'benchmark return':message1,
            'strategy return':message2
        }
        if self.show_plot:
            print('----------------------------')
            print(message1)
            print(message2)
            print('----------------------------')
        fig = plt.figure(1)
        plt.plot(dft.bench_prod_exp.values, label = 'benchmark')
        plt.scatter(range(len(dft)),np.where(dft[low_signal] == 1,dft.bench_prod_exp.values,np.nan),color = 'red', label = 'signal')
        plt.plot(dft.strat_prod_exp.values, label = 'strategy')
        plt.legend()
        plt.title('strategy and cumulative returns based on signal strategy')
        if self.show_plot:
            plt.plot()
            
        if self.save_path:
            result_json_name = f'signals_strategy_return_{feature_name}.json'
            result_plot_name = f'signals_strategy_return_{feature_name}.png'
            
            plt.savefig(self.save_path+result_plot_name)
            # pickle.dump(fig, open(self.save_path+result_plot_name, 'wb'))
            
            with open(self.save_path+result_json_name, "w") as outfile: 
                json.dump(messages, outfile)
                
        if self.save_path and self.save_aws:
            # upload_file_to_aws(bucket = 'VIRGO_BUCKET', key = f'market_plots/{self.ticket_name}/'+result_json_name ,input_path = self.save_path+result_json_name)
            # upload_file_to_aws(bucket = 'VIRGO_BUCKET', key = f'market_plots/{self.ticket_name}/'+result_plot_name,input_path = self.save_path+result_plot_name)
            
            upload_file_to_aws(bucket = 'VIRGO_BUCKET', key = self.save_aws + result_json_name, input_path = self.save_path + result_json_name, aws_credentials = self.aws_credentials)
            upload_file_to_aws(bucket = 'VIRGO_BUCKET', key = self.save_aws + result_plot_name, input_path = self.save_path + result_plot_name, aws_credentials = self.aws_credentials)

        if not self.show_plot:
            plt.close()
            
        del df1,df2,dft

        if self.return_fig:
            return fig, messages
        
def execute_signal_analyser(test_data_size, feature_name, days_list, configuration, method, object_stock, signal_analyser_object, plot = False, backtest= False):
    method(**configuration)
    signal_assess = signal_analyser_object(object_stock.df,object_stock.stock_code,show_plot = plot)
    signal_assess.signal_analyser(test_size = test_data_size, feature_name = feature_name, days_list = days_list, threshold = 1)
    
    if backtest:
        print('-----------------------back test ---------------------------')
        signal_assess.create_backtest_signal(backtest, test_data_size, feature_name)
    
    return signal_assess.mean_median_return

def iterate_signal_analyser(test_data_size,feature_name, days_list, arguments_to_test, method, object_stock, signal_analyser_object, plot = True):

    results = list()
    for key in arguments_to_test.keys():
        configuration = arguments_to_test.get(key)
        mean_median_return = execute_signal_analyser(test_data_size, feature_name, days_list, configuration, method, object_stock, signal_analyser_object)
        results.append(mean_median_return)
    
    df_result = pd.DataFrame({'keys':arguments_to_test.keys(),'results':results})
    if plot:
        plt.plot(df_result['keys'], df_result['results'])
        plt.scatter(df_result['keys'], df_result['results'])
        plt.title('simulation between configurations')
        plt.ylabel('median expected return') 
        plt.show()
        
    best_result = df_result.sort_values('results',ascending = False)['keys'].values[0]
    return best_result
    
class analyse_index(stock_eda_panel):
    def __init__(self, index, asset, n_obs, lag, data_window = '5y', show_plot = True, save_path = False, save_aws = False, aws_credentials = False):

        """
        data: pandas df
        index: str name of the index
        asset: str name of the asset
        n_obs: int
        lag: int
        data_window: str eg 5y 10y 15y
        show_plot: bool
        save_path: str local path for saving e.g r'C:/path/to/the/file/'
        save_aws: str remote key in s3 bucket path e.g. 'path/to/file/'
        aws_credentials: dict
        """
         
        self.index = index
        self.asset = asset
        self.n_obs = n_obs
        self.data_window = data_window
        self.lag = lag
        
        self.show_plot = show_plot
        self.save_path = save_path
        self.save_aws = save_aws
        
    def process_data(self):
        
        index = stock_eda_panel(self.index, self.n_obs, self.data_window)
        index.get_data()
        index.df['shift'] = index.df.Close.shift(self.lag)
        index.df['index_return'] = index.df.Close/index.df['shift'] - 1

        asset =  stock_eda_panel(self.asset, self.n_obs, self.data_window)
        asset.get_data()
        asset.df['shift'] = asset.df.Close.shift(self.lag)
        asset.df['asset_return'] = asset.df.Close/asset.df['shift'] - 1
        
        df1 = index.df[['Date','index_return']]
        df2 = asset.df[['Date','asset_return','Close']]
        merger = df1.merge(df2, on = 'Date', how = 'inner')
        merger.dropna(inplace = True)
        self.merger_df = merger
        
    def plot_betas(self,sample_size, offset, subsample_ts =False):
    
        ### extracting data

        self.process_data()
        
         ### ploting analysis
        figure, ax = plt.subplot_mosaic(
            [["scatter_total", "scatter_sample",'ts','ts']],
            layout="constrained",
            figsize=(18, 5)
        )
        
        ax['scatter_total'].scatter(self.merger_df.asset_return, self.merger_df.index_return)
        b, a = np.polyfit(self.merger_df.asset_return, self.merger_df.index_return, 1)
        ax['scatter_total'].plot(self.merger_df.asset_return, b*self.merger_df.asset_return+a, color='red')

        ax['ts'].plot(self.merger_df.Date, self.merger_df.Close, color = 'grey', alpha = 0.3)
        
        if subsample_ts:
            self.merger_df = self.merger_df.iloc[-subsample_ts:,:].dropna()
        
        for i in range(0,len(self.merger_df)-sample_size,offset):

            merger_ = self.merger_df.sort_values('Date', ascending = False).iloc[i:i+sample_size,:]
            x = merger_.index_return 
            y = merger_.asset_return
            b, a = np.polyfit(x,y, 1)

            normalize = mcolors.Normalize(vmin=-1, vmax=1)
            colormap = cm.jet

            ax['scatter_sample'].plot(x, y,'o', color = 'blue', alpha = 0.1)
            ax['scatter_sample'].plot(x, b*x+a, color=colormap(normalize(b)))
            ax['scatter_sample'].set_xlim(-0.06, 0.06)
            ax['scatter_sample'].set_ylim(-0.06, 0.06)

            plot = ax['ts'].scatter(merger_.Date, merger_.Close, color=colormap(normalize(b)), s = 10)

        scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
        scalarmappaple.set_array(x)
        
        plt.title(f'{self.asset} using index: {self.index}')
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
        
    def get_betas(self,subsample_ts=False):
        
        self.process_data()
        general_beta, a = np.polyfit(self.merger_df.asset_return, self.merger_df.index_return, 1)
        general_r = stats.mstats.pearsonr(self.merger_df.asset_return, self.merger_df.index_return)[0]
        
        self.process_data()
        if subsample_ts:
            self.merger_df = self.merger_df.iloc[-subsample_ts:,:].dropna()
        sample_beta, a = np.polyfit(self.merger_df.asset_return, self.merger_df.index_return, 1)
        sample_r = stats.mstats.pearsonr(self.merger_df.asset_return, self.merger_df.index_return)[0]
        
        result = {
            'general_beta':general_beta,
            'general_r':general_r,
            'sample_beta':sample_beta,
            'sample_r':sample_r
        }
        
        self.states_result = result
        
class evaluate_markets(analyse_index):
    def __init__(self, stock_code, indexes):
        self.stock_code = stock_code
        self.indexes = indexes
    def evaluate_best_market_fit(self,sample_size, offset,lag= 3, n_obs = 3500, verbose = False, plot_best = False):

        results_dicts = dict()
        for index in self.indexes:
            betex = analyse_index(index = index,asset = self.stock_code,n_obs = n_obs, lag = lag)
            betex.get_betas(sample_size)
            results_dicts[index] = betex.states_result
        pd_result = pd.DataFrame(results_dicts).T
        pd_result['gen_r2'] = pd_result.general_r ** 2
        pd_result['sampl_r2'] = pd_result.sample_r ** 2
        self.stat_results = pd_result
        
        best_result = pd_result.sort_values('gen_r2',ascending = False).head(2).sort_values('sampl_r2',ascending = False).head(1)
        best_fit_index = best_result.index.values[0]
        
        self.stat_results = self.stat_results.drop(columns = ['gen_r2','sampl_r2'])
        
        if verbose:
            print(best_result)
        if plot_best:
            betex = analyse_index(index = best_fit_index,asset = self.stock_code, n_obs = n_obs, lag = lag)
            betex.plot_betas(sample_size = sample_size, offset = offset, subsample_ts = False)
            
        self.best_result = best_result
        
def get_relevant_beta(data_market, ticket_name,  show_plot = True, save_path = False, save_aws = False, aws_credentials = False):
    """
        data_market: pandas df
        ticket_name: str name of the asset
        show_plot: bool
        save_path: str local path for saving e.g r'C:/path/to/the/file/'
        save_aws: str remote key in s3 bucket path e.g. 'path/to/file/'
        aws_credentials: dict
        """
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