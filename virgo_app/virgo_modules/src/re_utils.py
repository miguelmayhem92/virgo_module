import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns; sns.set()
import matplotlib.patheffects as path_effects
from  matplotlib.dates import DateFormatter

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.colors import DEFAULT_PLOTLY_COLORS

import pandas as pd
import numpy as np
import math
import json

import datetime
from dateutil.relativedelta import relativedelta

from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from scipy import stats

from .ticketer_source import stock_eda_panel
from pathlib import Path

import mlflow

from pykalman import KalmanFilter
from .aws_utils import upload_file_to_aws

def calculate_cointegration(series_1, series_2):
    coint_flag = 0
    coint_res = coint(series_1, series_2)
    coint_t = coint_res[0]
    p_value = coint_res[1]
    critical_value = coint_res[2][1]
    
    model = sm.OLS(series_1, series_2).fit()
    hedge_value = model.params[0]
    coint_flag = 1 if p_value < 0.05 and coint_t < critical_value else 0
    
    return coint_flag, hedge_value

class pair_finder():
    def __init__(self, raw_data , asset_1 ,asset_2):
        
        df = raw_data[[asset_1, asset_2]]
        coint_flag, hedge_ratio = calculate_cointegration(df[asset_1], df[asset_2])
        spread = df[asset_1] - (hedge_ratio * df[asset_2])
        df['spread'] = spread
        self.df = df
        self.asset_1 = asset_1
        self.asset_2 = asset_2
        
    def produce_zscore(self, window, z_threshold, verbose = False):
        self.z_threshold = z_threshold
        spread_series = pd.Series(self.df.spread)
        mean = spread_series.rolling(center = False, window = window).mean()
        std = spread_series.rolling(center = False, window = window).std()
        x = spread_series.rolling(center=False, window =  1).mean()
        z_score = (x - mean)/std

        self.df['z_score'] = z_score

        pvalue = round(adfuller(z_score.dropna().values)[1],4)
        if verbose:
            print(f'p value of the rolling z-score is {pvalue}')
        up_signal = np.where(z_score >= z_threshold,1,0)
        low_signal = np.where(z_score <= -z_threshold,1,0)

        self.df['up_pair_signal'] = up_signal
        self.df['low_pair_signal'] = low_signal
        
    def plot_scores(self):
        
        plt.axhline(y=0.0, color='grey', linestyle='--')
        plt.figure(1, figsize = (10, 4))
        plt.plot(self.df.spread.values)
        plt.show()
        print('-----------------------------------------------------------------')
        plt.figure(1,figsize = (10,4))
        plt.axhline(y=self.z_threshold, color='r', linestyle='--')
        plt.axhline(y=-self.z_threshold, color='r', linestyle='--')
        plt.axhline(y=0, color='blue', linestyle='-.')
        plt.plot(self.df.z_score.values)
        plt.show() 
        print('-----------------------------------------------------------------')

        asset_1_values = self.df[self.asset_1].values/self.df[self.asset_1].iloc[0].item()
        asset_2_values = self.df[self.asset_2].values/self.df[self.asset_2].iloc[0].item()

        fig = go.Figure()
        fig.add_trace(go.Scatter(name=self.asset_1,x=self.df.index, y=asset_1_values, mode='lines',marker=dict(color='blue')))
        fig.add_trace(go.Scatter(name = 'up_signal',x=self.df.index, y=np.where(self.df['up_pair_signal'] == 1, asset_1_values, np.nan), mode='markers',marker=dict(color='green')))
        fig.add_trace(go.Scatter(name = 'low_signal',x=self.df.index, y=np.where(self.df['low_pair_signal'] == 1, asset_1_values, np.nan), mode='markers',marker=dict(color='red')))

        fig.add_trace(go.Scatter(name=self.asset_2,x=self.df.index, y=asset_2_values, mode='lines',marker=dict(color='orange')))
        fig.add_trace(go.Scatter(name = 'up_signal',x=self.df.index, y=np.where(self.df['up_pair_signal'] == 1, asset_2_values, np.nan), mode='markers',marker=dict(color='green')))
        fig.add_trace(go.Scatter(name = 'low_signal',x=self.df.index, y=np.where(self.df['low_pair_signal'] == 1, asset_2_values, np.nan), mode='markers',marker=dict(color='red')))
        fig.update_layout(height=500, width=1200)

        fig.show()
        
    def evaluate_signal(self, days_list,test_size, signal_position = False,threshold = 0.05,verbose = False, plot = False):
    
        df = self.df.sort_values('Date').iloc[0:-test_size,:].copy()
        returns_list = list()
        
        for days in days_list:

            feature_ = f'return_{days}d'
            df[feature_] = (df[self.asset_1].shift(-days)/df[self.asset_1]-1)*100
            returns_list.append(feature_)

        df['signal_type'] = np.where(
            df['up_pair_signal'] == 1, 
            'up', 
            np.where(
                df['low_pair_signal'] == 1, 
                'down',
                None
            )
        )
        df = df[~df.signal_type.isna()]
        df['Date_'] = df.index
        df['lag_Date'] = df['Date_'].shift(1)
        df['span'] = (pd.to_datetime(df['Date_']) - pd.to_datetime(df['lag_Date'])).dt.days - 1
        df['break'] = np.where(df['span'] > 3, 1, 0)
        df['break'] = np.where(df['span'].isna(), 1, df['break'])
        
        df['chain_id'] = df.sort_values(['Date_']).groupby(['break']).cumcount() + 1
        df['chain_id'] = np.where(df['break'] == 1, df['chain_id'], np.nan )
        df['chain_id'] = df['chain_id'].fillna(method = 'ffill')

        df['internal_rn'] = df.sort_values(['Date_']).groupby(['chain_id']).cumcount() + 1
        df['inv_internal_rn'] = df.sort_values(['Date_'],ascending = False).groupby(['chain_id']).cumcount() + 1

        df['first_in_chain'] = np.where(df['internal_rn'] == 1, True, False)
        df['last_in_chain'] = np.where(df['inv_internal_rn'] == 1, True, False)

        df = df.drop(columns = ['break','span','Date_','lag_Date','inv_internal_rn']).sort_index()
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

        if plot:
            
            fig, axs = plt.subplots(1, 3, figsize = (15,5))
            df2 = df.copy()
            df2 = df2[df2.last_in_chain == True]
            df2['date'] = df2.index
            df2['lagdate'] = df2.date.shift(1)
            df2['span'] = (pd.to_datetime(df2['date']) - pd.to_datetime(df2['lagdate'])).dt.days
            sns.boxplot(data=df2, y="span",ax = axs[0])
            axs[0].set_title('span between last signals')
            del df2
            
            sns.boxplot(data=df[df.last_in_chain == True], y="internal_rn",ax = axs[1])
            axs[1].set_title('signal duration distribution')
            
            sns.boxplot(data=df_melt, x="time", y="value", hue="signal_type",ax = axs[2])
            axs[2].axhline(y=0, color='grey', linestyle='--')
            axs[2].set_title('signal type expected returns distribution at different time lapses')
            plt.show()
            
        del df
        
    def create_backtest_signal(self,days_strategy, test_size):
        asset_1 = self.asset_1
        df1 = self.df.iloc[-test_size:,:].copy()
        df2 = df1.copy()
        df2['signal_type'] = np.where(
                    df2['up_pair_signal'] == 1, 
                    'up', 
                    np.where(
                        df2['low_pair_signal'] == 1, 
                        'down',
                        None
                    )
                )
        df2 = df2[~df2.signal_type.isna()]
        df2['Date_'] = df2.index
        df2['lag_Date'] = df2['Date_'].shift(1)
        df2['span'] = (pd.to_datetime(df2['Date_']) - pd.to_datetime(df2['lag_Date'])).dt.days - 1
        df2['break'] = np.where(df2['span'] > 3, 1, 0)
        df2['break'] = np.where(df2['span'].isna(), 1, df2['break'])

        df2['chain_id'] = df2.sort_values(['Date_']).groupby(['break']).cumcount() + 1
        df2['chain_id'] = np.where(df2['break'] == 1, df2['chain_id'], np.nan )
        df2['chain_id'] = df2['chain_id'].fillna(method = 'ffill')

        df2['internal_rn'] = df2.sort_values(['Date_']).groupby(['chain_id']).cumcount() + 1
        df2['inv_internal_rn'] = df2.sort_values(['Date_'],ascending = False).groupby(['chain_id']).cumcount() + 1

        df2['first_in_chain'] = np.where(df2['internal_rn'] == 1, True, False)
        df2['last_in_chain'] = np.where(df2['inv_internal_rn'] == 1, True, False)

        df2 = df2.drop(columns = ['break','span','Date_','lag_Date','inv_internal_rn']).sort_index()

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

        print('----------------------------')
        print(f'returns benchmark {bench_rets}%')
        print(f'returns strategy {strat_rets}%')
        print('----------------------------')
        plt.plot(dft.bench_prod_exp.values, label = 'benchmark')
        plt.scatter(range(len(dft)),np.where(dft.low_pair_signal == 1,dft.bench_prod_exp.values,np.nan),color = 'red', label = 'signal')
        plt.plot(dft.strat_prod_exp.values, label = 'strategy')
        plt.legend()
        plt.title('strategy and cumulative returns based on signal strategy')
        plt.plot()

        del df1,df2,dft
        
def produce_big_dataset(data_frames, stocks_codes_, feature_list, limit = 500):
    
    feature_list_ = list()
    columns_vector = list(data_frames[stocks_codes_[-1]].columns )
    for feat in feature_list:
        feature_list_.append(feat)
        items = [featx for featx in [f'norm_{feat}', f'z_{feat}'] if featx in columns_vector]
        if len(items) > 0:
            feature_list_.append(items[0])
        feature_list_.append('signal_up_' + feat)
        feature_list_.append('signal_low_' + feat)
        
    features_list = ['Date','Close'] + feature_list_
    
    list_df = list()
    for ticket in stocks_codes_:
        
        df = data_frames[ticket]
        for j in features_list:
            if j not in df.columns:
                df[j] = df.get(j, np.nan)   
        
        df = df[features_list].sort_values('Date').iloc[-limit:,:]
        df['Ticket'] = ticket
        list_df.append(df)
    dataframe = pd.concat(list_df)
    return dataframe

def ranking(data, weighted_features, top = 5, window = 5):
    
    features = weighted_features.keys()
    up_columns = ['signal_up_' + x for x in features]
    low_columns = ['signal_low_' + x for x in features]
    
    ticket_list= list(data.Ticket.unique())
    result = dict()
    for ticket in ticket_list:
        result[ticket] = dict()
        df = data[data.Ticket == ticket].sort_values('Date').iloc[-window:]
        max_date = max(df['Date'])
        
        for col in low_columns:
            weight = weighted_features.get(col.replace('signal_low_',''))
            new_col = f'weighted_{col}'
            df[new_col] =  df[col]/((max_date - df['Date']) / np.timedelta64(1, 'D')+1) * weight
            sum_signal = np.sum(df[new_col].values)
            result[ticket][col] = sum_signal
        for col in up_columns:
            weight = weighted_features.get(col.replace('signal_up_',''))
            new_col = f'weighted_{col}'
            df[new_col] =  df[col]/((max_date - df['Date']) / np.timedelta64(1, 'D')+1) * weight
            sum_signal = np.sum(df[new_col].values)
            result[ticket][col] = sum_signal
            
    df = pd.DataFrame(result).T
    df['up_signas'] = df[up_columns].sum(axis=1)
    df['low_signas'] = df[low_columns].sum(axis=1)
    
    top_up = list(df.sort_values('up_signas', ascending = False).index)[:top]
    top_low = list(df.sort_values('low_signas', ascending = False).index)[:top]
    
    return top_up, top_low

def produce_dashboard(data, columns , ticket_list, show_plot = True, nrows = 150,save_name = False, save_path = False, save_aws = False):
    
    top = len(ticket_list)
    columns = ['history'] + columns
    subtitles = list()
    
    for ticket in ticket_list:
        for col in columns:
            subtitles.append(ticket + ': ' + col)
    
    fig = make_subplots(rows=top, cols=len(columns),vertical_spacing = 0.02, horizontal_spacing = 0.01, shared_xaxes=True, subplot_titles = subtitles)
    
    for i,ticket in enumerate(ticket_list):
        
        ## history
        i = i+1
        show_legend = True if i == 1 else False
        df = data[data.Ticket == ticket].sort_values('Date').iloc[-nrows:,:]
        
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'],legendgroup="Close",showlegend = show_legend , mode='lines',name = 'Close', marker_color = 'blue'),col = 1, row = i)

        ### signals
        
        for j,feature in enumerate(columns[1:]):
            j = j+2
            norm_list = [f'norm_{feature}', f'z_{feature}', feature]
            for norm_feat in norm_list:
                if norm_feat in df.columns:
                    fig.add_trace(go.Scatter(x=df['Date'], y=df[norm_feat],legendgroup="Close",showlegend = False , mode='lines',name = 'Close', marker_color = 'blue'),col = j, row = i)
                    break
            signal_up = f'signal_up_{feature}'
            signal_low = f'signal_low_{feature}'
            try:
                fig.add_trace(go.Scatter(x=df['Date'], y=np.where(df[signal_up] == 1, df[norm_feat], np.nan),showlegend = False, mode='markers',name = 'high up', marker_color = 'green'),col = j, row = i)
                fig.add_trace(go.Scatter(x=df['Date'], y=np.where(df[signal_low] == 1, df[norm_feat], np.nan),showlegend = False, mode='markers',name = 'high low', marker_color = 'red'),col = j, row = i)
            except:
                pass
            
    fig.update_layout(height=top*300, width=2000, title_text = f'dashboard top {top} tickets')
    if show_plot:
        fig.show()
    if save_name and save_path:
        fig.write_html(save_path+save_name+'.html')
        fig.write_json(save_path+save_name+'.json')
        
    if save_name and save_path and save_aws:
        upload_file_to_aws(bucket = 'VIRGO_BUCKET', key = f'multi_dashboards/'+save_name+'.json',input_path = save_path+save_name+'.json')
    
def rank_by_return(data, lag_days, top_n = 5):
    
    data = data.sort_values(['Ticket','Date'], ascending=[False,False]).reset_index(drop = True)
    data['first'] = data.sort_values(['Date'], ascending=[False]).groupby(['Ticket']).cumcount() + 1
    data =  data[data['first'] <= lag_days]
    
    data = data.sort_values(['Ticket','Date'], ascending=[False,False]).reset_index(drop = True)
    data['last'] = data.sort_values(['Date'], ascending=[True]).groupby(['Ticket']).cumcount() + 1
    
    data = data[(data['first'] == 1) | (data['last'] == 1)]
    
    data['last_Close'] = data.groupby('Ticket')['Close'].transform(lambda x: x.shift(-1))
    data = data[(data['first'] == 1)]
    data['return'] = (data['Close']/data['last_Close'] - 1)*100
    data = data.sort_values('return', ascending = True)
    data = data.iloc[:top_n,:]
    
    result = list(data.Ticket.values)
    
    return result

def get_data(ticker_name:str, ticket_settings:dict, n_days:int = False, hmm_available: object = False) -> object:
    """
    this functions runs the stock_eda_panel
    it is shared between train model and predictions
    arguments:
    hmm_available: if the hmm is available, in prediction is required
    ticker_name: name of the asset
    ticket_settings: dictionary with all the parameters to compute features
    n_days: to set an arbitrary data size

    returns: stock eda panel
    """

    object_stock = stock_eda_panel(ticker_name , n_days )
    object_stock.get_data()

    # computing features if they exists in the ticketr settings

    if 'volatility' in ticket_settings['settings']:
        object_stock.volatility_analysis(
            lags = ticket_settings['settings']['volatility']['lags'],
            trad_days =  ticket_settings['settings']['volatility']['trad_days'], 
            window_log_return =  ticket_settings['settings']['volatility']['window_log_return']
        )
        
    if 'outlier' in ticket_settings['settings']:
        object_stock.outlier_plot(ticket_settings['settings']['outlier']['zlim']) 

    if 'spread_ma' in ticket_settings['settings']:
        object_stock.spread_MA(
            ma1 = ticket_settings['settings']['spread_ma']['ma1'],
            ma2 = ticket_settings['settings']['spread_ma']['ma2'],
            limit = ticket_settings['settings']['spread_ma']['limit']
        )
    if 'relative_spread_ma' in ticket_settings['settings']:
        object_stock.relative_spread_MA(
            ma1=ticket_settings['settings']['relative_spread_ma']['ma1'],
            ma2=ticket_settings['settings']['relative_spread_ma']['ma2'],
            threshold = ticket_settings['settings']['relative_spread_ma']['threshold'],
        )

    if 'pair_feature' in ticket_settings['settings']:
        object_stock.pair_feature(pair_symbol = ticket_settings['settings']['pair_feature']['pair_symbol'])
        object_stock.produce_pair_score_plot(
            window = ticket_settings['settings']['pair_feature']['window'],
            z_threshold = ticket_settings['settings']['pair_feature']['z_threshold']
        )  
        
    if 'count_features' in ticket_settings['settings']:
        object_stock.get_count_feature(
            rolling_window = ticket_settings['settings']['count_features']['rolling_window'],
            threshold = ticket_settings['settings']['count_features']['threshold']
        )
    
    if 'bidirect_count_features' in ticket_settings['settings']:
        object_stock.bidirect_count_feature(
            rolling_window = ticket_settings['settings']['bidirect_count_features']['rolling_window'],
            threshold = ticket_settings['settings']['bidirect_count_features']['threshold']
        )

    if 'price_range' in ticket_settings['settings']:
        object_stock.get_range_feature(
            window = ticket_settings['settings']['price_range']['window'], 
            up_threshold = ticket_settings['settings']['price_range']['up_threshold'], 
            low_threshold = ticket_settings['settings']['price_range']['low_threshold']
            )
    
    if 'relative_price_range' in ticket_settings['settings']:
        object_stock.get_relative_range_feature(
            window = ticket_settings['settings']['relative_price_range']['window'],
            threshold = ticket_settings['settings']['relative_price_range']['threshold']
        )
    
    if 'rsi_feature' in ticket_settings['settings']:
        object_stock.rsi_feature(
            window = ticket_settings['settings']['rsi_feature']['window'], 
            lag_rsi_ret = ticket_settings['settings']['rsi_feature']['lag_rsi_ret'], 
            threshold = ticket_settings['settings']['rsi_feature']['threshold']
        )
    
    if 'rsi_feature_v2' in ticket_settings['settings']:
        object_stock.rsi_feature_improved(
            window = ticket_settings['settings']['rsi_feature_v2']['window'],
            threshold = ticket_settings['settings']['rsi_feature_v2']['threshold']
        )

    if 'days_features' in ticket_settings['settings']:
        object_stock.days_features(
            window_day = ticket_settings['settings']['days_features']['window_day'], 
            limit = ticket_settings['settings']['days_features']['limit']
        )
    
    if 'days_features_v2' in ticket_settings['settings']:
        object_stock.days_features_bands(
            window = ticket_settings['settings']['days_features_v2']['window'],
            threshold = ticket_settings['settings']['days_features_v2']['threshold']
        )
    
    if 'volume_features' in ticket_settings['settings']:
        object_stock.analysis_volume(
            lag_volume = ticket_settings['settings']['volume_features']['lag_volume'], 
            threshold  = ticket_settings['settings']['volume_features']['threshold'],
            window = ticket_settings['settings']['volume_features']['window']
        )
    
    if 'smooth_volume' in ticket_settings['settings']:
        object_stock.analysis_smooth_volume(
            window = ticket_settings['settings']['smooth_volume']['window'],
            threshold = ticket_settings['settings']['smooth_volume']['threshold']
        )

    if hmm_available:
        object_stock.cluster_hmm_analysis( n_clusters = None, 
                                    features_hmm = None,
                                    test_data_size = None,
                                    seed = None, model = hmm_available)
    else:
        object_stock.cluster_hmm_analysis( n_clusters = ticket_settings['settings']['hmm']['n_clusters'],
                                        features_hmm = ticket_settings['settings']['hmm']['features_hmm'],
                                        test_data_size = ticket_settings['settings']['hmm']['test_data_size'],
                                        seed = ticket_settings['settings']['hmm']['seed'])
    
    return object_stock

trends = {'adjusted' : 0.001, 'smooth' : 0.0001}

def apply_KF(self, trends):
    for ttrend in trends:
        tcov = trends.get(ttrend)
        kf = KalmanFilter(transition_matrices = [1],
                             observation_matrices = [1],
                             initial_state_mean = 0,
                             initial_state_covariance = 1,
                             observation_covariance=1,
                             transition_covariance=tcov)
        vector = kf.filter(self.df[['Close']])[0]
        self.df[f'KalmanFilter_{ttrend}'] = vector.reshape((vector.shape[0]))
        
stock_eda_panel.apply_KF = apply_KF

def call_ml_objects(stock_code, client, call_models = False):
    
    objects = dict()
    
    registered_model_name = f'{stock_code}_models'
    latest_version_info = client.get_latest_versions(registered_model_name, stages=["Production"])
    latest_production_version = latest_version_info[0].version
    run_id_prod_model = latest_version_info[0].run_id
    
    
    ticket_settings = mlflow.artifacts.load_dict(
        f'runs:/{run_id_prod_model}/ticket_settings.json'
    )

     ## calling models
    
    hmm_model = mlflow.pyfunc.load_model(
            f"runs:/{run_id_prod_model}/{stock_code}-hmm-model",
             suppress_warnings = True
            )
    objects['called_hmm_models'] = hmm_model
    
    if call_models:
        
        forecasting_model = mlflow.pyfunc.load_model(
            f"runs:/{run_id_prod_model}/{stock_code}-forecasting-model",
             suppress_warnings = True
            )
        objects['called_forecasting_model'] = forecasting_model
        
    object_stock = get_data(
                ticker_name= stock_code, 
                ticket_settings = ticket_settings,
                n_days = ticket_settings['settings']['general']['n_days'],
                hmm_available = hmm_model
            )
    ### applying kalman
    object_stock.apply_KF(trends)
    
    
    objects['called_ticket_settings'] = ticket_settings
    objects['called_data_frame'] = object_stock.df
    
    return objects

class produce_plotly_plots:
    def __init__(self,ticket_name, data_frame,settings, save_path = False, save_aws = False, show_plot= True):
        self.ticket_name = ticket_name
        self.data_frame = data_frame
        self.settings = settings
        self.save_path = save_path
        self.save_aws = save_aws
        self.show_plot = show_plot
        
    def plot_asset_signals(self, feature_list,spread_column, date_intervals = False,):
        
        result_json_name = 'panel_signals.json'
        df = self.data_frame
        ma1 = self.settings['settings'][spread_column]['ma1']
        ma2 = self.settings['settings'][spread_column]['ma2']
        hmm_n_clust = self.settings['settings']['hmm']['n_clusters']
        feature_rows = len(feature_list)

        rows_subplot = feature_rows + 1
        height_plot = rows_subplot * 400
        color_map = { i:DEFAULT_PLOTLY_COLORS[i] for i in range(hmm_n_clust)}

        ### expand hmm analysis

        fig = make_subplots(
            rows= rows_subplot, cols=1,
            vertical_spacing = 0.02, horizontal_spacing = 0.02, shared_xaxes=True,
            subplot_titles = feature_list + ['Hidden states over closing prices'] )

        ### signal plots
        for row_i, feature in enumerate(feature_list,start=1):
            feature_2 = 'nan'
            signal_up_list = [f'signal_up_{feature}', f'signal_up_{feature_2}']  
            signal_low_list = [f'signal_low_{feature}', f'signal_low_{feature_2}']
            norm_list = [f'norm_{feature}', f'z_{feature}', feature]
            # signal
            for norm_feat in norm_list:
                if norm_feat in df.columns:
                    fig.add_trace(go.Scatter(x=df['Date'], y=df[norm_feat],showlegend= False, mode='lines', marker_color = 'grey'),col = 1, row = row_i)
                    break
            for norm_feat in norm_list:
                if norm_feat in df.columns:
                    fig.add_trace(go.Scatter(x=df['Date'], y=np.where(df[norm_feat] > 0, df[norm_feat], np.nan),showlegend= False, mode='markers', marker_color = 'green',opacity = 0.3),col = 1, row = row_i)
                    fig.add_trace(go.Scatter(x=df['Date'], y=np.where(df[norm_feat] <= 0, df[norm_feat], np.nan),showlegend= False, mode='markers', marker_color = 'red',opacity = 0.3),col = 1, row = row_i)
                    break
            for signal_up in signal_up_list:
                if signal_up in df.columns:
                    fig.add_trace(go.Scatter(x=df['Date'], y=np.where(df[signal_up] == 1, df[norm_feat], np.nan),showlegend= False, mode='markers', marker_color = 'green'),col = 1, row = row_i)

            for signal_low in signal_low_list:
                if signal_low in df.columns:
                    fig.add_trace(go.Scatter(x=df['Date'], y=np.where(df[signal_low] == 1, df[norm_feat], np.nan),showlegend= False, mode='markers', marker_color = 'red'),col = 1, row = row_i)

        fig.update_layout(height=height_plot, width=1600, title_text = f'asset plot and signals: {self.ticket_name}')

        ## state plot with close prices
        row_i = row_i + 1
        map_ = {i:f'state_{i}' for i in range(hmm_n_clust)}
        df['HMM_state'] =  df['hmm_feature'].map(map_)

        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines',marker_color ='blue'),row=row_i, col=1)
        for state in df['HMM_state'].unique():
            dfi = df[df['HMM_state'] == state]
            hmm_id = dfi['hmm_feature'].unique()[0]
            fig.add_trace(go.Scatter(x=dfi['Date'], y=dfi['Close'], mode='markers',name = state, marker_color = color_map[hmm_id]),row=row_i, col=1)

        fig.add_trace(go.Scatter(x=df['Date'], y=df['KalmanFilter_adjusted'], mode='lines',name = 'KF_adjusted', marker_color = 'grey'),row=row_i, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['KalmanFilter_smooth'], mode='lines',name = 'KF_smooth', marker_color = 'darkviolet'),row=row_i, col=1)

        if date_intervals:
            for interval in date_intervals:
                fig.add_vrect(x0=interval[0], x1=interval[1], line_width=0, fillcolor="red", opacity=0.2)

        if self.save_path:
            fig.write_json(self.save_path+result_json_name)
        if self.show_plot:
            fig.show()
        if self.save_path and self.save_aws:
            upload_file_to_aws(bucket = 'VIRGO_BUCKET', key = f'market_plots/{self.ticket_name}/'+result_json_name ,input_path = self.save_path+result_json_name)

    def explore_states_ts(self):
        result_json_name = 'ts_hmm.json'
        df = self.data_frame
        hmm_n_clust = self.settings['settings']['hmm']['n_clusters']
        state_rows = math.ceil(hmm_n_clust/2)

        rows_subplot = state_rows 
        height_plot = rows_subplot * 400

        states = list(df.hmm_feature.unique())
        states.sort()
        states_subtitles = [f'state {x}' for x in states]
        if len(states_subtitles)%2 == 1:
            states_subtitles = states_subtitles + [None]


        fig = make_subplots(
            rows= rows_subplot, cols=2,
            specs = [[{"type": "scatter"},{"type": "scatter"}]]*state_rows,
            vertical_spacing = 0.02, horizontal_spacing = 0.02,
            subplot_titles =  states_subtitles )

        ### only states scaled series
        states = list(df.hmm_feature.unique())
        color_map = { i:DEFAULT_PLOTLY_COLORS[i] for i in range(hmm_n_clust)}
        states.sort()
        row_i = 1
        for state in states:
            colx = int(state)%2 + 1
            dfi = df[df.hmm_feature == state]
            chains = list(dfi.chain_id.unique())
            for chain in chains:
                dfj = dfi[dfi.chain_id == chain]
                fig.add_trace(go.Scatter(x=dfj.hmm_chain_order, y=dfj.chain_return, mode='lines', marker_color = color_map[state],showlegend=False),row=row_i, col=colx)
            if colx == 2:
                row_i +=1
        fig.update_layout(height=height_plot, width=1600, title_text = f'time series by state: {self.ticket_name}') 
        
        if self.save_path:
            fig.write_json(self.save_path+result_json_name)
        if self.show_plot:
            fig.show()
        if self.save_path and self.save_aws:
            upload_file_to_aws(bucket = 'VIRGO_BUCKET', key = f'market_plots/{self.ticket_name}/'+result_json_name ,input_path = self.save_path+result_json_name)

    def plot_hmm_analysis(self,settings, hmm_model, date_intervals = False, model = False):
        result_json_name = 'hmm_analysis.json'
        df = self.data_frame
        hmm_n_clust = self.settings['settings']['hmm']['n_clusters']

        rows_subplot = 2
        height_plot = rows_subplot * 400
        color_map = { i:DEFAULT_PLOTLY_COLORS[i] for i in range(hmm_n_clust)}
        states = list(df.hmm_feature.unique())
        states.sort()    
        ### expand hmm analysis
        hmm_titles = ['Transition matrix heatmap' , 'state return (base first observation)','length chains dist']

        fig = make_subplots(
            rows= rows_subplot, cols=2,
            specs = 
                [[{"type": "box"}, {"type": "heatmap"}]]+\
                [[{"type": "box"}, {"type": "box"}]]
            ,vertical_spacing = 0.15, horizontal_spacing = 0.02,
            subplot_titles =  hmm_titles)

        ### transition probabilities
        row_i = 1
        t_matrix = (hmm_model._model_impl['hmm'].transmat_)*100
        fig.add_trace(go.Heatmap(z = t_matrix, text = np.round(t_matrix,2),texttemplate="%{text}",coloraxis='coloraxis'),row=row_i, col=2)
        fig.update_xaxes(title_text='State To', row=row_i, col=2)
        fig.update_yaxes(title_text='State From', row=row_i, col=2)

        ## returns of state
        df_ = df[['Date','hmm_feature','Close',"chain_return"]].sort_values('Date')
        df_['Daily_Returns'] = df['Close'].pct_change(7)

        for state in states:
            dfi = df_[df_.hmm_feature == state]
            fig.add_trace(go.Box(y = dfi.chain_return, name=str(state),showlegend=False, marker_color = color_map[state] ),row=1, col=1)

        ## lengths chains by state dist
        if 'hmm_chain_order' in df.columns:
            df_agg = df.groupby(['hmm_feature','chain_id'],as_index = False).agg(length_by_chain = ('hmm_chain_order','max'))

        else:
            df['lag_hmm_feature'] = df['hmm_feature'].shift(1)
            df['breack'] = np.where(df['lag_hmm_feature'] != df['hmm_feature'],1,0)
            df["chain_id"] = df.groupby("breack")["Date"].rank(method="first", ascending=True)
            df["chain_id"] = np.where(df['breack'] == 1,df["chain_id"],np.nan)
            df["chain_id"] = df["chain_id"].fillna(method='ffill')
            df["hmm_chain_order"] = df.groupby('chain_id')["Date"].rank(method="first", ascending=True)

            df_agg = df.groupby(['hmm_feature','chain_id'],as_index = False).agg(length_by_chain = ('hmm_chain_order','max'))

        for state in states:
            dfi = df_agg[df_agg.hmm_feature == state]
            fig.add_trace(go.Box(y = dfi.length_by_chain, name=str(state),showlegend=False, marker_color = color_map[state] ),row=2, col=1)
        
        ## feature importance of regressor
        if model and settings['model_type'] == 'Forecaster':
            
            n_regresors = self.settings['settings']['target_lasts']['steps']
            importances = list()
            for regressor_x in range(n_regresors): 
                importances.append(model._model_impl['model'].estimators_[regressor_x].feature_importances_)
            importances = np.vstack( importances )

            default_features_in_model = [f'label_{i}' for i in range(1,importances.shape[1]+1)]
            features_in_model = settings.get('selected_feature_list_prod', default_features_in_model)

            importances_df = pd.DataFrame(importances, columns = features_in_model)
            importances_df = importances_df.melt(value_vars=features_in_model,var_name='feature', value_name='importance')
            importances_df['median'] = importances_df['feature'].map(importances_df.groupby('feature')['importance'].median())
            importances_df = importances_df.sort_values('median', ascending = False)

            for feature in importances_df.feature.unique():
                dfi = importances_df[importances_df.feature == feature]
                fig.add_trace(go.Box(x = dfi.importance, name=str(feature),showlegend=False ),row=2, col=2)
            fig.update_yaxes(visible=False, title="feature",row=2, col=2)

        
        fig.update_layout(height=height_plot, width=1600, title_text = f'State model analysis: {self.ticket_name}', coloraxis=dict(colorbar_len=0.50))

        date_execution = datetime.datetime.today().strftime('%Y-%m-%d')
        current_step = df.iloc[-1,:].hmm_chain_order
        current_state = df.iloc[-1,:].hmm_feature
        message1 = 'current state: ' +  str(current_state)
        message2 = 'current step in state: ' + str(current_step)
        message3 = 'report date: ' +str(date_execution)

        messages = {
            'current_state':message1,
            'current_step_state': message2,
            'report_date':message3,
        }
        
        if self.show_plot:
            fig.show()
            print('---------------------------------------------------')
            print('------------------ strategy -----------------------')
            print('---------------------------------------------------')
            print(f'ticket: {self.ticket_name}')
            strategy_object = settings['settings'].get('strategies', False)
            if strategy_object:
                print('best strategy: ', strategy_object['best_strategy'])
            else:
                print('no strategy recorded')
            print(message1)
            print(message2)
            print(message3)
            
        if self.save_path:
            fig.write_json(self.save_path+result_json_name)
            with open(self.save_path+"market_message.json", "w") as outfile: 
                json.dump(messages, outfile)
                
        if self.save_path and self.save_aws:
            upload_file_to_aws(bucket = 'VIRGO_BUCKET', key = f'market_plots/{self.ticket_name}/'+result_json_name ,input_path = self.save_path+result_json_name)
            upload_file_to_aws(bucket = 'VIRGO_BUCKET', key = f'market_plots/{self.ticket_name}/'+'market_message.json',input_path = self.save_path+"market_message.json")
            
    def produce_forecasting_plot(self,predictions):
        result_json_name = 'forecast_plot.json'
        hmm_n_clust = self.settings['settings']['hmm']['n_clusters']
        model_type = self.settings.get('model_type',False)
        lags = self.settings['settings']['volatility']['lags']

        df = self.data_frame

        height_plot = 1 * 400

        fig = make_subplots(
            rows= 1, cols=2,vertical_spacing = 0.05, horizontal_spacing = 0.05,
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}]],
            subplot_titles = [f'asset returns {lags} lags', 'closing prices', 'hidden states']
        )


        predictions = predictions[predictions.StockCode == self.ticket_name]
        if len(predictions) > 1: 

            try:
                predictions['ExecutionDate'] = pd.to_datetime(predictions['ExecutionDate'], format='mixed',utc=True).dt.date
            except:
                predictions['ExecutionDate'] = pd.to_datetime(predictions['ExecutionDate'],utc=True).dt.date
            try:
                predictions['Date'] = pd.to_datetime(predictions['Date'], format='mixed',utc=True).dt.date
            except:
                predictions['Date'] = pd.to_datetime(predictions['Date'],utc=True).dt.date

            last_exe_prediction_date = predictions.ExecutionDate.unique()
            last_date = max(last_exe_prediction_date)

            history = predictions[(predictions.Type == 'History') & (predictions.ExecutionDate == last_date)]
            cut_date = history.loc[history.iloc[-1:,:].index[0]:,'Date'].item()

            prediction = predictions[predictions.Type == 'Prediction']

            ## log returns

            fig.add_trace(go.Scatter(x=history.Date, y=history.log_return, mode='lines',marker_color ='blue',showlegend=False),row=1, col=1)

            for i,datex in enumerate([x for x in last_exe_prediction_date if x != last_date]):
                df = prediction[prediction.ExecutionDate == datex]
                fig.add_trace(go.Scatter(x=df.Date, y=df.log_return, mode='markers',marker_color ='grey',showlegend=False),row=1, col=1)

            df = prediction[prediction.ExecutionDate == last_date]
            fig.add_trace(go.Scatter(x=df.Date, y=df.log_return, mode='lines',marker_color ='#ff7f0e',showlegend=False),row=1, col=1)
            fig.add_trace(go.Scatter(x=df.Date, y=df.log_return, mode='markers',marker_color ='#ff7f0e',showlegend=False),row=1, col=1)

            ## closing prices

            fig.add_trace(go.Scatter(x=history.Date, y=history.Close, mode='lines',marker_color ='blue',showlegend=False),row=1, col=2)
            for i,datex in enumerate([x for x in last_exe_prediction_date if x != last_date]):
                df = prediction[prediction.ExecutionDate == datex]
                fig.add_trace(go.Scatter(x=df.Date, y=df.Close, mode='markers',marker_color ='grey',showlegend=False),row=1, col=2)

            df = prediction[prediction.ExecutionDate == last_date]  
            fig.add_trace(go.Scatter(x=df.Date, y=df.Close, mode='lines',marker_color ='#ff7f0e',showlegend=False),row=1, col=2)
            fig.add_trace(go.Scatter(x=df.Date, y=df.Close, mode='markers',marker_color ='#ff7f0e',showlegend=False),row=1, col=2)
            fig.update_layout(height=height_plot, width=1600, title_text = f'forecasts: {self.ticket_name}')
        else:
            print('no forecasting history')
            
        if self.save_path:
            fig.write_json(self.save_path+result_json_name)
        if self.show_plot:
            fig.show()
        if self.save_path and self.save_aws:
            upload_file_to_aws(bucket = 'VIRGO_BUCKET', key = f'market_plots/{self.ticket_name}/'+result_json_name ,input_path = self.save_path+result_json_name)

def plot_hmm_analysis_logger(data_frame,test_data_size, save_path = False, show_plot = True):
    
    df = data_frame
    df_ = df[['Date','hmm_feature','Close',"chain_return"]].sort_values('Date')
    fig, axs = plt.subplots(1,2,figsize=(10,4))
    df__ = df_.iloc[:-test_data_size,]
    sns.boxplot(data=df__, x="hmm_feature", y="chain_return",ax = axs[0]).set_title('train dist')
    df__ = df_.iloc[-test_data_size:,]
    sns.boxplot(data=df__ , x="hmm_feature", y="chain_return",ax = axs[1]).set_title('test dist')
    if save_path:
        plt.savefig(save_path) 
    if not show_plot:
        plt.close()

def plot_hmm_tsanalysis_logger(data_frame, test_data_size,save_path = False, show_plot = True):
    
    df = data_frame
    df_ = df[['Date','hmm_feature','Close',"chain_return"]].sort_values('Date')
    states = list(df_['hmm_feature'].unique())
    states.sort()
    
    if test_data_size:
        df__ = df_.iloc[-test_data_size:,]
        date_limit = pd.Timestamp(str(df__.Date.min().strftime('%Y-%m-%d')))
    
    fig, ax1 = plt.subplots(figsize=(10,4))
    ax1.plot(df_['Date'],df_["Close"])
    
    for state in states:
        df__ = df_[df_.hmm_feature == state]
        ax1.scatter(df__['Date'],df__["Close"], label = state)
    formatter = DateFormatter('%Y-%m-%d')
    if test_data_size:
        plt.axvline(x=date_limit, color = 'r')
    fig.legend()
    fig.autofmt_xdate()
    if save_path:
        plt.savefig(save_path) 
    if not show_plot:
        plt.close()