import pandas as pd
import numpy as np
import json

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns; sns.set()

import warnings
warnings.filterwarnings('ignore')

from .aws_utils import upload_file_to_aws

def sharpe_ratio(return_series):

    '''
    calculate sharpe ratio for given array.

            Parameters:
                    return_series (pd.series): pandas series of the asset returns

            Returns:
                    sharpe (float): sharpe ratio
    '''

    N = 255 # Trading days in the year (change to 365 for crypto)
    rf = 0.005 # Half a percent risk free rare
    mean = return_series.mean() * N -rf
    sigma = return_series.std() * np.sqrt(N)
    sharpe = round(mean / sigma, 3)
    return sharpe


class SignalAnalyserObject:
    """
    Class that produces back-tests analysis for a given feature

    Attributes
    ----------
    symbol_name : str
        stock or asset to assess
    feature_name : str
    test_size: int
        testing data size
    show_plot: boolean
    save_path: str
        if available, save result locally
    save_aws: str
        if available, save result locally
    aws_credentials: dict
    signal_position: int
        if available, signal position to open a position
    df: pd.DataFrame
        transformed data of the selected feature to perform back-test
    median_return: float
        median return after end low signals
    
    Methods
    -------
    signal_analyser(days_list=list):
        given a signal position for either botton or roof signal, calculate the espected return and distributions for a time scope in the days list (time horizons)
    create_backtest_signal(days_strategy=int, high_exit=float, low_exit=float, open_in_list=list):
        create a back-test analysis using the test data using some opening anc closing postion criterias
    """

    def __init__(self, data,symbol_name, feature_name, test_size, signal_position = False, correct_signals = False, show_plot = True, save_path = False, save_aws = False, aws_credentials = False, return_fig = False):
        """
        Initialize object

        Parameters
        ----------
        data (pd.DataFrame): data
        ticket_name (str): name of the asset
        feature_name (str): name of the features
        test_size (int): size of the test data
        signal_position (int): signal position to open the position, False by default
        correct_signals (int): clean abnormal signals using interpolation
        show_plot (boolean): if true show plot for every method
        save_path (str): if true, save results in file e.g r'C:/path/to/the/file/'
        save_aws (str): if true, export results to remote repo e.g. 'path/to/file/'
        aws_credentials (dict): credentials for aws
        return_fig (boolean): if true, methods will return objects

        Returns
        -------
        None
        """
        self.ticket_name = symbol_name
        self.feature_name=feature_name
        self.test_size=test_size
        self.show_plot = show_plot
        self.save_path = save_path
        self.save_aws = save_aws
        self.aws_credentials = aws_credentials
        self.return_fig = return_fig
        self.signal_position = signal_position
        ## preprocessing 
        up_signal, low_signal= f'signal_up_{feature_name}', f'signal_low_{feature_name}'
        features_base = ['Date', up_signal, low_signal, 'Close','Open','High','Low']

        df = data[features_base].sort_values('Date')

        df['signal_type'] = np.where(
            df[up_signal] == 1,
            'up',
            np.where(
                df[low_signal] == 1,
                'down',
                'no signal'
            )
        )
        def correct_sygnals(df,correct_i = 1):
            ### signal cleaning
            for i in range(1+correct_i, len(df)-1):
                start_i, end_i = i-(correct_i+1), i+1
                dfw = df.iloc[start_i: end_i,]
                before_type = dfw.iloc[0].signal_type
                after_type = dfw.iloc[-1].signal_type
                window_types = dfw.iloc[1:-1].signal_type.unique()
                n_window_type = len(window_types)
                if n_window_type == 1:
                    if (before_type == after_type) and (window_types[0] != after_type):
                        df.iloc[start_i+1: end_i-1, df.columns.get_loc('signal_type')] = before_type
            return df.copy()

        if correct_signals:
            for correct_i in range(1,correct_signals+1):
                df = correct_sygnals(df,correct_i = correct_i)
            df[up_signal] = np.where(df['signal_type'] == 'up', 1,0)
            df[low_signal] = np.where(df['signal_type'] == 'down', 1,0)

        ## indexing chains
        df['lag_signal_type'] = df['signal_type'].shift(1)
        df['lag_Date'] = df['Date'].shift(1)
        df['span'] = (pd.to_datetime(df['Date']) - pd.to_datetime(df['lag_Date'])).dt.days - 1
        df['break'] = np.where((df['span'] > 3) & (df['lag_signal_type'] == df['signal_type']), 1, 0)
        df['break'] = np.where((df['lag_signal_type'] != df['signal_type']), 1, df['break'])
        df['chain_id'] = df.sort_values(['Date']).groupby(['break']).cumcount() + 1
        df['chain_id'] = np.where(df['break'] == 1, df['chain_id'], np.nan )
        df['chain_id'] = df['chain_id'].fillna(method = 'ffill')

        df['internal_rn'] = df.sort_values(['Date']).groupby(['chain_id']).cumcount() + 1
        df['inv_internal_rn'] = df.sort_values(['Date'],ascending = False).groupby(['chain_id']).cumcount() + 1

        df['first_in_chain'] = np.where(df['internal_rn'] == 1, True, False)
        df['last_in_chain'] = np.where(df['inv_internal_rn'] == 1, True, False)

        df['span'] = (pd.to_datetime(df['Date']) - pd.to_datetime(df['lag_Date'])).dt.days - 1
        self.df = df.drop(columns = ['span','break','lag_signal_type','lag_Date']).copy()
        
    def signal_analyser(self, days_list):
        """
        Initialize object

        Parameters
        ----------
        days_list (list): list of integers to calculate expected returns

        Returns
        -------
        if returns_fig is true, returns a matplotlib fig
        """
        signal_position = self.signal_position
        df = self.df.iloc[0:-self.test_size,:].copy()
        returns_list = list()

        for days in days_list:
            feature_ = f'return_{days}d'
            df[feature_] = (df['Close'].shift(-days)/df['Close']-1)*100
            returns_list.append(feature_)
            
        df['open_long'] = np.where(df.last_in_chain == True, True, np.nan)
        df['open_short'] = np.where(df.first_in_chain == True, True, np.nan)
        df.signal_type = df.signal_type.map({'up':'go down', 'down': 'go up'})
        
        # median return
        returns_list = [f'return_{days}d' for days in days_list]
        df_melt = df[df.open_long == True].pivot_table(index=['signal_type'], values=returns_list, aggfunc='median')
        df_melt['median'] = df_melt[returns_list].median(axis = 1)
        self.median_return = df_melt.loc['go up', 'median']
        
        # plotting
        fig, axs = plt.subplots(1, 4, figsize = (20,5))
        palette ={"go down": "tomato", "go up": "lightblue"}
        
        df2 = df[df.signal_type.isin(['go down','go up'])]
        df2['lag_Date'] = df2['Date'].shift(1)
        df2['lag_signal_type'] = df2['signal_type'].shift(1)
        df2 = df2[df2.lag_signal_type != df2.signal_type]
        df2['span'] = (pd.to_datetime(df2['Date']) - pd.to_datetime(df2['lag_Date'])).dt.days - 1
        sns.violinplot(data=df2, y="span",ax = axs[0], color = 'lightblue', linewidth=0.7,inner="quart")
        sns.stripplot(data=df2, y="span",ax = axs[0], jitter=True, zorder=1)
        axs[0].set_title('span between last signals')

        df_ = df[df.last_in_chain == True]
        df_['part'] = '-'
        sns.violinplot(data=df_, y="internal_rn", x='part', ax = axs[1], hue="signal_type", inner="quart",palette = palette,gap=0.1, split=True, linewidth=0.7)
        axs[1].set_title('signal duration distribution')

        if signal_position:
            for feature in returns_list:
                df[feature] = df[feature].shift(-signal_position)
        
        df_melt = df[df.open_long == 1].melt(id_vars=['signal_type'], value_vars=returns_list, var_name='time', value_name='value')
        df_melt = df_melt.dropna()
        sns.violinplot(data=df_melt, x="time", y="value", hue="signal_type",ax = axs[2], split=True, gap=0.1, inner="quart",palette = palette, linewidth=0.8)
        axs[2].axhline(y=0, color='grey', linestyle='--')
        axs[2].set_title('E. returns - end of the signal')

        df_melt = df[df.open_short == 1].melt(id_vars=['signal_type'], value_vars=returns_list, var_name='time', value_name='value')
        df_melt = df_melt.dropna()
        sns.violinplot(data=df_melt, x="time", y="value", hue="signal_type",ax = axs[3], split=True, gap=0.1, inner="quart",palette = palette, linewidth=0.8)
        axs[3].axhline(y=0, color='grey', linestyle='--')
        axs[3].set_title('E. returns - start of the signal')
        
        if self.show_plot:
            plt.show()

        if self.save_path:
            result_plot_name = f'signals_strategy_distribution_{self.feature_name}.png'
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
        
    def create_backtest_signal(self,days_strategy, high_exit = False, low_exit = False, open_in_list = ['down']):
        """
        Initialize object

        Parameters
        ----------
        days_strategy (int): position horizon
        high_exit (float): max threshold to close position
        low_exit (float): min threshold to close position, this parameter has to be positive
        open_in_list (list): list of strings ("down","up") to assess signals
        Returns
        -------
        if returns_fig is true, returns a matplotlib fig and list of dicts containing analysis
        """
        asset_1 = 'Close'
        up_signal, low_signal= f'signal_up_{self.feature_name}', f'signal_low_{self.feature_name}'
        signal_position = self.signal_position
        dft = self.df.iloc[-self.test_size:,:].reset_index(drop=True).copy()
        
        dft['lrets_bench'] = np.log(dft[asset_1]/dft[asset_1].shift(1))
        dft['bench_prod'] = dft['lrets_bench'].cumsum()
        dft['bench_prod_exp'] = np.exp(dft['bench_prod']) - 1

        map_ = {'down':'END LOW TREND', 'up': 'BEGINNING HIGH TREND'}

        open_in_list_items = len(open_in_list)
        fig, axs = plt.subplots(1,open_in_list_items, figsize = (7*open_in_list_items,6))
        messages = list()
        for i, open_in in enumerate(open_in_list):
            axs_ = axs if open_in_list_items == 1 else axs[i]
            if open_in == 'down':
                dft['open_long'] = np.where((dft.last_in_chain == True) & (dft.signal_type == 'down'), True, np.nan) # open strat
            elif open_in == 'up':
                dft['open_long'] = np.where((dft.first_in_chain == True) & (dft.signal_type == 'up'), True, np.nan) # open strat
                
            def chain_position(dft):
                dft['open_long_id'] = np.where(dft['open_long'] == True, dft.chain_id, np.nan)
                dft['open_long_id'] = dft['open_long_id'].fillna(method = 'ffill')
                dft['open_long_rn'] = dft.sort_values(['Date']).groupby(['open_long_id']).cumcount() + 1
                return dft
            
            if signal_position:
                dft['open_long'] = dft.sort_values(['Date'])['open_long'].shift(signal_position)
        
            dft = chain_position(dft)
            dft['flag'] = np.where(dft['open_long_rn'] < days_strategy, 1,0)
        
            if high_exit and low_exit:
                dft['open_strat'] = np.where(dft.open_long == True, dft.Open, np.nan) # open strat
                dft['open_strat'] = dft['open_strat'].fillna(method = 'ffill')
                dft['open_strat'] = np.where(dft.flag == 1, dft.open_strat, np.nan)
                dft['high_strat_ret'] = (dft['High']/dft['open_strat']-1)*100
                dft['low_strat_ret'] = (dft['Low']/dft['open_strat']-1)*100
                dft['max_step_chain'] = dft.groupby(['open_long_id'])['open_long_rn'].transform('max')
                dft['high_exit'] =  np.where(((dft['high_strat_ret'] >= high_exit) | (dft['open_long_rn'] == days_strategy) | (dft['max_step_chain'] == dft['open_long_rn'])), 1, np.nan)
                dft['low_exit'] =  np.where((dft['low_strat_ret'] <= low_exit), -1, np.nan)
        
                dft["exit_type"] = dft[["high_exit", "low_exit"]].max(axis=1)
                dft['exit_type'] = np.where(dft["exit_type"] == 1, 1, np.where(dft["exit_type"] == -1,-1,np.nan))
                dft['exit'] = np.where(dft['exit_type'].isnull(), np.nan, 1)
                dft['exit_order'] = dft.sort_values(['Date']).groupby(['open_long_id','exit']).cumcount() + 1
                dft['exit'] = np.where(dft['exit_order'] == 1, True, np.nan)
                dft = dft.drop(columns = ['exit_order'])
                ## if last signal is near
                max_id = dft.open_long_id.max()
                dft['max_internal_rn'] = dft.sort_values(['Date']).groupby(['open_long_id']).open_long_rn.transform('max')
                dft['exit'] = np.where((dft.open_long_id == max_id) & (dft.max_internal_rn < days_strategy) & (dft.max_internal_rn == dft.open_long_rn), 1, dft['exit'])
        
                dft['exit_step'] = np.where(dft.exit == 1, dft.open_long_rn, np.nan)
                dft['exit_step'] = dft.sort_values(['Date']).groupby(['open_long_id']).exit_step.transform('max')
        
                dft['flag'] = np.where(dft.open_long_rn <= dft.exit_step, 1, 0)
        
            dft['lrets_strat'] = np.log(dft[asset_1].shift(-1)/dft[asset_1]) * dft['flag']
            dft['lrets_strat'] = np.where(dft['lrets_strat'].isna(),-0.0,dft['lrets_strat'])
            dft['lrets_prod'] = dft['lrets_strat'].cumsum()
            dft['strat_prod_exp'] = np.exp(dft['lrets_prod']) - 1
        
            bench_rets = round(dft['bench_prod_exp'].values[-1]*100,1)
            strat_rets = round(dft['strat_prod_exp'].values[-1]*100,1)
        
            bench_sr = round(sharpe_ratio(dft.bench_prod_exp.dropna()),1)
            strat_sr = round(sharpe_ratio(dft.strat_prod_exp.dropna()),1)
        
            message1 = f'{bench_rets}%'
            message2 = f'{strat_rets}%'
        
            messages_ = {
                'type strategy':map_[open_in],
                'benchmark return:':message1,
                'benchmark sharpe ratio:': bench_sr,
                'strategy return:':message2,
                'strategy sharpe ratio:': strat_sr,
            }
            messages.append(messages_)
            if self.show_plot:
                print('----------------------------')
                print(messages_)
                print('----------------------------')
        
            
            axs_.plot(dft.bench_prod_exp.values, label = 'benchmark', color = 'steelblue')
            axs_.scatter(range(len(dft)),np.where(dft[low_signal] == 1,dft.bench_prod_exp.values,np.nan),color = 'red', label = 'signal')
            axs_.scatter(range(len(dft)),np.where(dft[up_signal] == 1,dft.bench_prod_exp.values,np.nan),color = 'green', label = 'signal')
            axs_.plot(dft.strat_prod_exp.values, label = 'strategy', color = 'darksalmon')
            axs_.set_xlabel("index")
            axs_.set_ylabel("comulative return")
            axs_.set_title(f'{map_[open_in]} strategy and cumulative returns')
            axs_.legend()
            
        if self.show_plot:
            plt.plot()

        if self.save_path:
            result_json_name = f'signals_strategy_return_{self.feature_name}.json'
            result_plot_name = f'signals_strategy_return_{self.feature_name}.png'

            plt.savefig(self.save_path+result_plot_name)

            with open(self.save_path+result_json_name, "w") as outfile:
                json.dump(messages, outfile)

        if self.save_path and self.save_aws:

            upload_file_to_aws(bucket = 'VIRGO_BUCKET', key = self.save_aws + result_json_name, input_path = self.save_path + result_json_name, aws_credentials = self.aws_credentials)
            upload_file_to_aws(bucket = 'VIRGO_BUCKET', key = self.save_aws + result_plot_name, input_path = self.save_path + result_plot_name, aws_credentials = self.aws_credentials)

        if not self.show_plot:
            plt.close()

        del dft

        if self.return_fig:
            return fig, messages
        
class IterateSignalAnalyse(SignalAnalyserObject):
    """
    object that is going to iterate backtest given a parameter space

    Attributes
    ----------
    test_data_size : int
    feature_name : str
    days_list: list
        list of integers that serve as time horizons
    arguments_to_test : dict
        paramter space
    method: str
        method to use
    object_stock: obj
        object containing data and methods
    plot: boolean
        show summary plot of median results
    best_result: float
        index of the best result, the index corresponds to the parameter space

    Methods
    -------
    execute(show_plot_iter=boolean):
        display plots for every iteration
    """
    def __init__(self, test_data_size, feature_name, days_list, arguments_to_test, method, object_stock, plot = False):
        """
        Parameters
        ----------
        test_data_size (int): size of the test data
        feature_name (str): name of the feature
        days_list (list): list of integers that serve as time horizons
        arguments_to_test (dict): paramter space
        method (str): method to use
        object_stock (obj): object containing data and methods
        plot (boolean): show summary plot of median results

        Returns
        -------
        None
        """
        self.test_data_size = test_data_size
        self.feature_name = feature_name
        self.days_list = days_list
        self.arguments_to_test = arguments_to_test
        self.method = method
        self.plot = plot
        self.object_stock = object_stock
        
    def execute(self,show_plot_iter = False):
        """
        Iterate backtest and compute median result for every iteration

        Parameters
        ----------
        show_plot_iter  (boolean): display plots for every iteration

        Returns
        -------
        None
        """
        results = list()
        for key in self.arguments_to_test.keys():
            configuration = self.arguments_to_test.get(key)
            getattr(self.object_stock, self.method)(**configuration)
            signal_assess = SignalAnalyserObject(self.object_stock.df, self.object_stock.stock_code, show_plot = show_plot_iter, test_size = self.test_data_size, feature_name = self.feature_name)
            signal_assess.signal_analyser(days_list = self.days_list)
            mean_median_return = signal_assess.median_return
            results.append(mean_median_return)
            
        df_result = pd.DataFrame({'keys':self.arguments_to_test.keys(),'results':results})
        if self.plot:
            plt.plot(df_result['keys'], df_result['results'])
            plt.scatter(df_result['keys'], df_result['results'])
            plt.title('simulation between configurations')
            plt.ylabel('median expected return')
            plt.show()
    
        best_result = df_result.sort_values('results',ascending = False)['keys'].values[0]
        self.best_result = best_result

def execute_signal_analyser(test_data_size, feature_name, days_list, configuration, method, object_stock, analyser_object, plot = False, backtest= False, exit_params = {}):
    '''
    code snippet that is going run backtest and display analysis messages and plots

            Parameters:
                    test_data_size (int): test data size
                    feature_name (str): name of the feature to assess
                    days_list (list): tome scope to assess the returns
                    configuration (dict): parameters of the method to run
                    object_stock (obj): object with data to assess
                    method (str): method to use
                    analyser_object (obj): signal_analyser object
                    plot (boolean): if true, plot results
                    backtest (boolean): if true, run backtest 
                    exit_params (dict): parameters of exit returns

            Returns:
                    None
    '''
    getattr(object_stock, method)(**configuration)
    signal_assess = analyser_object(object_stock.df,object_stock.stock_code,show_plot = plot, feature_name = feature_name, test_size = test_data_size)
    signal_assess.signal_analyser(days_list = days_list)
    signal_assess.create_backtest_signal(backtest, open_in_list = ['down','up'], **exit_params )