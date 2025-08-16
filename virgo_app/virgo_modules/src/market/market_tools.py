import gc

import pandas as pd
import numpy as np

from sklearn.linear_model import HuberRegressor
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from matplotlib import cm
import matplotlib.colors as mcolors

class MarketAnalysis:
    """
    Class that perform market analysis using robust linear regression

    Attributes
    ----------
    data : pd.DataFrame
        input data
    market_features : list
        list of market feature (log returns) to apply analysis
    return_cols: str
        main log return feature
    col_map: dict
        dictionary containing rename of market features

    Methods
    -------
    compute_beta(data=pd.DataFrame, feature_x=str, feature_y=str):
        compute betas given x and y using robust linear regression
    get_correlation(data=pd.DataFrame, feature_x=str, feature_y=str):
        compute correlation given x and y
    produce_beta_report(data=pd.DataFrame):
        produce beta report
    compute_general_report(sample_size=int, offset=int, index=str, subsample_ts=int, show_plot=bool):
        compute full report, global and latest window
    """

    def __init__(self, data, market_features, return_col, col_map=None):
        self.data = data.dropna()
        self.market_features = market_features
        self.return_cols = return_col
        self.col_map=col_map
        
    def compute_beta(self, data, feature_x, feature_y):
        """
        compute betas given x and y using robust linear regression

        Parameters
        ----------
        data (pd.DataFrame): input data containing analysis features
        feature_x (str): name of the feature x
        feature_y (str): name of the feature y

        Returns
        -------
        (beta(str), alpha(str))
        """
        x = data[feature_x].values.reshape(-1,1)
        y = data[feature_y].values.reshape(-1,1)
        huber_regr = HuberRegressor(fit_intercept = True)
        huber_regr.fit(x, y)
        beta, alpha = huber_regr.coef_[0], huber_regr.intercept_
        return beta, alpha

    def get_correlation(self, data, feature_x, feature_y):
        """
        compute correlation given x and y

        Parameters
        ----------
        data (pd.DataFrame): input data containing analysis features
        feature_x (str): name of the feature x
        feature_y (str): name of the feature y

        Returns
        -------
        r (float)
        """
        x = data[feature_x]
        y = data[feature_y]
        r = stats.mstats.pearsonr(x, y)[0]
        return r

    def produce_beta_report(self, data):
        """
        produce beta report

        Parameters
        ----------
        data (pd.DataFrame): input data containing analysis features

        Returns
        -------
        report (pd.DataFrame)
        """
        result = {
            "market_index": list(),
            "beta": list(),
            "alpha": list(),
            "r": list()
        }
        for index in self.market_features:
            beta, alpha = self.compute_beta( data, self.return_cols, index)
            r = self.get_correlation( data, self.return_cols, index)
            result["market_index"].append(index)
            result["beta"].append(beta)
            result["alpha"].append(alpha)
            result["r"].append(r)
        pd_result = pd.DataFrame(result)
        pd_result = pd_result.sort_values("r", ascending=False)
        if self.col_map:
            pd_result["map_market_index"] = pd_result.market_index.map(self.col_map)
        return pd_result
        
    def compute_general_report(self, sample_size, offset, index=False, subsample_ts=False, show_plot=True):
        """
        compute full report, global and latest window

        Parameters
        ----------
        sample_size (int): sample size for every beta computation
        offset (int): offset or overlap between samples
        index (str): if provided, bet fit index is taken
        subsample_ts (int): subsample for iterative beta calculation
        show_plot (bool): whether to show plot

        Returns
        -------
        (report (pd.DataFrame), latest_report (pd.DataFrame), figure (mtpl.plt))
        """
        general_report = self.produce_beta_report(self.data)
        current_report = self.produce_beta_report(self.data.iloc[sample_size:,:])
        if not index:
            index = general_report.head(1).market_index.values[0]
        b = general_report[general_report.market_index == index].beta.values
        a = general_report[general_report.market_index == index].alpha.values

        figure, ax = plt.subplot_mosaic(
            [["scatter_total", "scatter_sample",'ts','ts']],
            layout="constrained",
            figsize=(18, 5)
        )
        x = self.data[self.return_cols]
        y = self.data[index]
        ax['scatter_total'].scatter(x, y)
        ax['scatter_total'].plot(x, b*x+a, color='red')

        if subsample_ts:
            merger_df = self.data.iloc[-subsample_ts:,:].copy()
        else:
            merger_df = self.data.copy()
        ax['ts'].plot(merger_df.Date, merger_df.Close, color = 'grey', alpha = 0.3)
        b_array = list()
        for i in range(0,len(merger_df)-sample_size,offset):
            merger_ = merger_df.sort_values('Date', ascending = False).iloc[i:i+sample_size,:]
            b, a = self.compute_beta(merger_, self.return_cols, index)
            x = merger_[self.return_cols]
            y = merger_[index]
            normalize_ = mcolors.Normalize(vmin=-2.0, vmax=2.0)
            colormap_ = cm.jet
            ax['scatter_sample'].plot(x, y,'o', color = 'blue', alpha = 0.1)
            ax['scatter_sample'].plot(x, b*x+a, color=colormap_(normalize_(b)))
            ax['scatter_sample'].set_xlim(-0.08, 0.08)
            ax['scatter_sample'].set_ylim(-0.08, 0.08)
            plot = ax['ts'].scatter(merger_.Date, merger_.Close, color=colormap_(normalize_(b)), s = 10)
            b_array.append(b)
        normalize_ = mcolors.Normalize(vmin=np.min(b_array), vmax=np.max(b_array))
        colormap_ = cm.jet
        x_global = self.data[self.return_cols]
        scalarmappaple = cm.ScalarMappable(norm=normalize_, cmap=colormap_)
        scalarmappaple.set_array(x_global)
        if self.col_map:
            map_index = self.col_map.get(index)
            title = f'market analysis of {map_index}'
        else:
            title = f'market analysis'
        plt.title(title)
        plt.colorbar(scalarmappaple)
        del merger_df
        gc.collect()
        if show_plot:
            plt.show()
        else:
            plt.close() 
        return general_report, current_report, figure