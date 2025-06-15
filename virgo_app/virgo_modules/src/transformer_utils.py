from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

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
    
class InteractionFeatures(BaseEstimator, TransformerMixin):

    """
    Class that applies feature interaction.
    this class is compatible with scikitlearn pipeline

    Attributes
    ----------
    feature_list1 : list
        list of features to combine
    feature_list2 : list
        list of features to combine

    Methods
    -------
    fit(additional="", X=DataFrame, y=None):
        fit transformation.
    transform(X=DataFrame, y=None):
        apply feature transformation
    """

    def __init__(self, feature_list1, feature_list2):
        self.feature_list1 = feature_list1
        self.feature_list2 = feature_list2

    def fit(self, X, y=None):
        return self
    
    def simple_div_interaction(self, data, feature1, feature2, feature_name):
        data[feature_name] = data[feature1]*data[feature2]
        data[feature_name] = data[feature_name].replace([np.inf, -np.inf], 0)
        data[feature_name] = data[feature_name].fillna(0)
        return data

    def transform(self, X, y=None):
        for f1 in self.feature_list1:
            for f2 in self.feature_list2:
                fn = 'iterm_'+f1.replace("norm_","")+"_"+f2.replace("norm_","")
                X = self.simple_div_interaction(X, f1, f2, fn)
        return X
