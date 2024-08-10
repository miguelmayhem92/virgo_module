from hmmlearn.hmm import GaussianHMM

from sklearn.pipeline import Pipeline
from feature_engine.imputation import MeanMedianImputer
from virgo_modules.src.ticketer_source import FeatureSelector
from feature_engine.selection import DropCorrelatedFeatures
from sklearn.preprocessing import RobustScaler

import pandas as pd
import numpy as np
import random

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns; sns.set()

from virgo_modules.src.ticketer_source import FeatureSelector, states_relevance_score

def create_hmm_derived_features(df, lag_returns):
    """
    create features derived from hmm states features. Features are the index of the state, the duration of the state, chain raturn
    note: this is a copy of the method of the ticketer_object with the same name

    Parameters:
            df (pd.DataFrame): dataframe that must have hmm_feature columns
            lag_returns (int): lag paramter (not used)
    
    Returns:
            df (pd.DataFrame): dataframe with extra hmm features as columns
    """
    df = df.sort_values('Date')
    ## indexing chains
    df['lag_hmm_feature'] = df['hmm_feature'].shift(1)
    df['breack'] = np.where(df['lag_hmm_feature'] != df['hmm_feature'],1,0)
    df["chain_id"] = df.groupby("breack")["Date"].rank(method="first", ascending=True)
    df["chain_id"] = np.where(df['breack'] == 1,df["chain_id"],np.nan)
    df["chain_id"] = df["chain_id"].fillna(method='ffill')
    df["hmm_chain_order"] = df.groupby('chain_id')["Date"].rank(method="first", ascending=True)
     ### returns using the windowsseeds
    df['lag_chain_close'] = df.sort_values(by=["Date"]).groupby(['chain_id'])['Close'].shift(lag_returns)
    df['chain_return'] = (df['Close']/df['lag_chain_close'] -1) * 100
    df = df.drop(columns = ['breack'])
    return df

class trainer_hmm():
    """
    wrapper that gaussian model
    this class follows scikit learn practices

    Attributes
    ----------
    hmm_model: obj
        pipeline and model
    features_hmm: list
        list of features used to train the gaussian model
        
    Methods
    -------
    train():
        train pipeline given the parameters in the class initiliazation
    plot_training_results(lag_diff_returns=int):
        plot features and closing prices displaying the states
        plot the returns distribution by state given lag to calculate the returns in the chains
    """
    def __init__(self, data, features_hmm, n_clusters= 3, corr_thrshold = 0.65, seed = None):
        """
        Initialize object

        Parameters
        ----------
        data (pd.DataFrame): training data
        features_hmm (list): features to pass for modeling
        n_clusters (int): number or states to train
        corr_thrshold (float): correlation threhsold for initial feature selection
        seed (int): random state for model reproducibility

        Returns
        -------
        None
        """
        self.__data_train = data
        self.__features_hmm = features_hmm
        self.__n_clusters = n_clusters
        self.__corr_thrshold = corr_thrshold
        self.__seed = seed
    def train(self):
        """
        train pipeline and model

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        transform_pipe = Pipeline([
            ('selector', FeatureSelector(columns=self.__features_hmm)),
            ('fillna', MeanMedianImputer(imputation_method='median',variables=self.__features_hmm)),
            ('drop_correlated', DropCorrelatedFeatures(method='spearman',threshold=self.__corr_thrshold)),
        ])

        # features_hmm = list(transform_pipe.fit_transform(self.__data_train).columns)
        # n_features = len(features_hmm)
        # startprob_prior =  np.array([1/self.__n_clusters]*self.__n_clusters)
        transmat_prior = np.diag([0.70]*self.__n_clusters)
        # means_prior = np.array([1/n_features]*n_features)
        pipeline_hmm = Pipeline([
            ('transfrom_pipe', transform_pipe),
            ('scaler', RobustScaler()),
            ('hmm', GaussianHMM(
                n_components =  self.__n_clusters, covariance_type = 'spherical', 
                # startprob_prior = startprob_prior, 
                transmat_prior = transmat_prior, 
                # means_prior = means_prior,
                random_state = self.__seed,)
            )
        ])

        self.hmm_model = pipeline_hmm.fit(self.__data_train)
        self.features_hmm = [x for x in self.__features_hmm if x not in list(self.hmm_model[0][-1].features_to_drop_)]
        
    def plot_training_results(self, lag_diff_returns):
        """
        plot result as matplot figure

        Parameters
        ----------
        lag_diff_returns (int): lag or diff factor to calculate returns of chains

        Returns
        -------
        None
        """
        n_clusters = self.__n_clusters
        df_train = self.__data_train.copy()
        df_train['hmm_feature'] = self.hmm_model.predict(df_train)
        df_train = create_hmm_derived_features(df_train, lag_diff_returns,)
        n = len(self.features_hmm)+1
        fig, axs = plt.subplots(n, 1, figsize=(10, 3*n), sharex=True)
        for i,feature in enumerate(self.features_hmm):
            axs[i].plot(df_train.Date, df_train[feature])
            axs[i].set_title(feature)
            for s in range(n_clusters):
                df = df_train[df_train['hmm_feature'] == s]
                axs[i].scatter(df.Date, df[feature])
            
        axs[i+1].plot(df_train.Date, df_train.Close)
        axs[i+1].set_title('close price')
        for s in range(n_clusters):
            df = df_train[df_train['hmm_feature'] == s]
            axs[i+1].scatter(df.Date, df.Close)
        
        n = 1
        fig, axs = plt.subplots(n, 1, figsize=(10, 3*n), sharex=True)
        df_plot = df_train.dropna()
        sns.boxplot(data=df_plot, x="hmm_feature", y="chain_return", hue="hmm_feature", ax=axs)
        axs.axhline(0.5, linestyle='--')
        del df_train

def evaluate_model_chains(data, n_clusters, at_least_states, threshold_chain, at_least_length):
    """
    function that is going to assess chains or series of states given some sanity chekcs

    Parameters:
            data (pd.DataFrame): dataframe that must have hmm_feature and extra features
            n_clusters (int): n_clusters that are trainned, not observed
            at_least_states (int): number of states that should be ,at least, observed
            threshold_chain (int): number of times that a state should be , at least, observed
            at_least_length (int): minimal lenght that the states should have using a statical measure (median, q75, max, etc)
    
    Returns:
            result (boolean): true if the model complies with parameters
    """
    def q3(x):
        return x.quantile(0.75)
    tmp_df = data.groupby(['hmm_feature','chain_id'],as_index = False).agg(chain_lenght = ('hmm_chain_order','max'))
    tmp_df = tmp_df.groupby("hmm_feature", as_index = False).agg(count = ('chain_id','nunique'), median_length = ('chain_lenght','median'), q3_length = ('chain_lenght',q3))
    train_observedstates = len(tmp_df)
    
    states_under_threshold = list(tmp_df[tmp_df['count'] <= threshold_chain].hmm_feature)
    n_states_under_threshold = len(states_under_threshold)
    min_count = np.min(tmp_df[~tmp_df.hmm_feature.isin(states_under_threshold)]['count'].values) 
    med_length = np.min(tmp_df['q3_length'].values)
    
    condition_1 = threshold_chain <= min_count
    condition_2 = n_states_under_threshold <=  at_least_states
    condition_3 = at_least_length <= med_length
    condition_4 = (train_observedstates == n_clusters) 

    result = False

    if  condition_1 and condition_2 and condition_3 and condition_4:
        result = True
    else:
        result = False
    return result

def iterate_training(trials, train_params, relevance_params):
    """
    iterate valid training

    Parameters:
            trials (int): number of repetitions to iterate
            train_params (dict): dictionary containing training configurations
            relevance_params (dict): dictionary containing validation configurations
    
    Returns:
            results (list): list of valid relevance scores
            kept_model (obj): model (pipeling) that is kept, if it exists
    """
    results = list()
    kept_model=None
    for _ in range(trials):
        try:
            th = trainer_hmm(**train_params)
            th.train()
            result_model = th.hmm_model
            df_train_tmp = train_params.get('data')
            df_train_tmp['hmm_feature'] = result_model.predict(df_train_tmp)
            df_train_tmp = create_hmm_derived_features(df = df_train_tmp, lag_returns = relevance_params.get('lag'))
            relev, _, _ = states_relevance_score(df_train_tmp)
            relevance_hmm = evaluate_model_chains(data = df_train_tmp,
                                  n_clusters=train_params.get('n_clusters'),
                                  at_least_states=relevance_params.get('at_least_states'),
                                  threshold_chain=relevance_params.get('threshold_chain'),
                                  at_least_length=relevance_params.get('at_least_length'))
            if relevance_hmm:
                results.append(relev)
                kept_model = result_model
        except:
            pass
        del th
    if not kept_model:
        raise TypeError("no model was kept")
    return results, kept_model

class custom_hmm_permutation_importance():
    """
    class that is going to perform feature importance using feature permutation
    note: this method is inpired in the same method that is available in scikit-learn

    Attributes
    ----------
    n_repeats: int
        number of shufflings performed per feature 
    features: list
        list of features that is going to be tested, note that these features have to be the input of the model
    results: dict
        dictionary with the results containing feature and relevance scores per each iteration

    Methods
    -------
    fit():
        fit class
    """
    def __init__(self, model, X, n_repeats=5,random_state=False, features = list(), lag = 4):
        """
        Initialize object

        Parameters
        ----------
        model (obj): pipeline or model
        X (pd.DataFrame): input data to test feature permutation
        n_repeats (int): number or trials per feature
        random_state (bool): if true set a random state
        features (list): list of features to be tested. note that the features have to be input of the model
        lag (int): lag of diff factor to calculate chain returns

        Returns
        -------
        None
        """
        self.__model = model
        self.__X = X
        self.n_repeats = n_repeats
        self.__random_state = random_state
        self.features = features
        self.__lag = lag
    def __generate_seeds(self):
        """
        generate list of seeds

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.__random_state:
            self.__seeds = list()
            for _ in range(self.n_repeats):
                seed = np.random.randint(1,500)
                self.__seeds.append(seed)
    def fit(self):
        """
        fit class

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.__X['hmm_feature'] = self.__model.predict(self.__X)
        self.__X = create_hmm_derived_features(df=self.__X, lag_returns=self.__lag)
        init_relevance, _, _  = states_relevance_score(self.__X)
        self.results = {feature: list() for feature in self.features}
        if self.__random_state:
            self.__generate_seeds()
        for feature in self.features:
            X_ = self.__X.dropna().reset_index(drop = True).copy()
            for j in range(self.n_repeats):
                if self.__random_state:
                    seed = self.__seeds[j]
                    np.random.seed(seed)
                else:
                    seed = None
                shuffled = X_[feature].sample(frac=1, random_state = seed, replace = True).reset_index(drop=True)
                X_[feature] = shuffled
                X_['hmm_feature'] = self.__model.predict(X_)
                X_ = create_hmm_derived_features(df=X_, lag_returns=self.__lag)
                
                tmp_df = X_.groupby(['hmm_feature','chain_id'],as_index = False).agg(chain_lenght = ('hmm_chain_order','max'))
                tmp_df = tmp_df.groupby("hmm_feature", as_index = False).agg(count = ('chain_id','nunique'), median_length = ('chain_lenght','median')).copy()
                mean_relevance, _, _ = states_relevance_score(X_)
                self.results[feature].append(mean_relevance - init_relevance)
            del X_

def hmm_feature_selection(max_features, trials, train_params, relevance_params):
    """
    wrapper function that is going to use permutation importance to select features

    Parameters:
            ax_features (int): target to number of features
            trials (int): training iterations 
            train_params (dict): dictionary containing training configurations
            relevance_params (dict): dictionary containing validation configurations
    
    Returns:
            results (pd.DataFrame): summary relevace score per excluded feature 
    """
    results = {'index':list(),'feature_to_drop':list(), 'median relevance excluding feature':list()}
    i=0
    init_numfeatures = len(train_params.get('features_hmm'))
    while max_features <= init_numfeatures:
        print(init_numfeatures)
        if i==0:
            exclude = None
            r,model= iterate_training(trials, train_params, relevance_params)
            for ri in r:
                results['index'].append(0)
                results['feature_to_drop'].append('full')
                results['median relevance excluding feature'].append(ri)
        data_train = train_params.get('data')
        chmm_pi = custom_hmm_permutation_importance(model, data_train,random_state=5, features = train_params.get('features_hmm'), lag = relevance_params.get('lag'))
        chmm_pi.fit()
        results_fp = pd.DataFrame(chmm_pi.results)
        feature_deltas = results_fp.median(axis = 0)
        feature_deltas = feature_deltas.sort_values(ascending = False)
        feature_to_drop = feature_deltas.index[0]
        print(f'excluding {feature_to_drop}')
        
        train_params['features_hmm'].remove(feature_to_drop)
        print(train_params['features_hmm'])
        r,model = iterate_training(trials, train_params, relevance_params)
        for ri in r:
            results['index'].append(i+1)
            results['feature_to_drop'].append(feature_to_drop)
            results['median relevance excluding feature'].append(ri)
        init_numfeatures = len(model[:-2].transform(data_train).columns)
        i+=1
    return pd.DataFrame(results)


def seed_finder(train_params, relevance_params, n_seed = 100,max_results =5):
    """
    iterate valid training finding best starter seed

    Parameters:
            train_params (dict): dictionary containing training configurations
            relevance_params (dict): dictionary containing validation configurations
            n_seed (int): number of iterations
            max_results (int): number of max results to keep and stop the iteration
    
    Returns:
            df_results (pd.DataFrame): summary table of seed and relevance score
    """
    seeds = list()
    i_ = 0
    while len(seeds) < max_results and i_ < n_seed:
        # print(i_)
        if i_ >= (n_seed*0.5) and len(seeds) == 0:
            i_ += 10
        
        seed = random.randint(50, 10000)
        train_params['seed'] = seed
        try:
            th = trainer_hmm(**train_params)
            th.train()
            result_model = th.hmm_model
            df_train_tmp = train_params.get('data')
            df_train_tmp['hmm_feature'] = result_model.predict(df_train_tmp)
            df_train_tmp = create_hmm_derived_features(df = df_train_tmp, lag_returns = relevance_params.get('lag'))
            relev, _, _ = states_relevance_score(df_train_tmp)
            relevance_hmm = evaluate_model_chains(data = df_train_tmp,
                                  n_clusters=train_params.get('n_clusters'),
                                  at_least_states=relevance_params.get('at_least_states'),
                                  threshold_chain=relevance_params.get('threshold_chain'),
                                  at_least_length=relevance_params.get('at_least_length'))
            if relevance_hmm:
                print('new model candidate was found, seed saved')
                seeds.append(seed)
            i_ += 1
        except:
            i_ += 1
    print('best seeds', seeds)
    ## searching the best seed
    results = {'seed' : list(),'train_relevance': list()}
    
    for seed_x in seeds:
        train_params['seed'] = seed_x
        th = trainer_hmm(**train_params)
        th.train()
        result_model = th.hmm_model
        df_train_tmp = train_params.get('data')
        df_train_tmp['hmm_feature'] = result_model.predict(df_train_tmp)
        df_train_tmp = create_hmm_derived_features(df = df_train_tmp, lag_returns = relevance_params.get('lag'))
        relev, _, _ = states_relevance_score(df_train_tmp)
    
        results['seed'].append(seed_x)
        results['train_relevance'].append(relev)

    df_results = pd.DataFrame(results).sort_values(['train_relevance'], ascending = [False])
    return df_results