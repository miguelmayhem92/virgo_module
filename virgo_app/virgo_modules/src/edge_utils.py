import numpy as np
import itertools

from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from feature_engine.selection import DropFeatures, DropCorrelatedFeatures
from feature_engine.imputation import  MeanMedianImputer
from virgo_modules.src.ticketer_source import FeatureSelector
from feature_engine.discretisation import EqualWidthDiscretiser
from feature_engine.datetime import DatetimeFeatures

from .ticketer_source import VirgoWinsorizerFeature, InverseHyperbolicSine

class produce_model_wrapper:
    """
    class that wraps a pipeline and a machine learning model. it also provides data spliting train/validation

    Attributes
    ----------
    data : pd.DataFrame
        list of features to apply the transformation
    X_train : pd.DataFrame
    y_train : pd.DataFrame
    X_val : pd.DataFrame
    y_val : pd.DataFrame
    self.pipeline: obj
        sklearn pipeline including model and pipleline

    Methods
    -------
    preprocess(validation_size=int, target=list):
        ingest data and split data between train and validation data and X and Y data
    train_model(pipe=obj, model=obj, cv_=boolean):
        merge and train pipeline and machine learning model
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
    
    def preprocess(self, validation_size, target):
        """
        ingest data and split data between train and validation data and X and Y data

        Parameters
        ----------
        validation_size (int): validation data size, the remaining is taken as training data
        target (list): target column list

        Returns
        -------
        None
        """
        val_date = self.data.groupby('Date', as_index = False).agg(target_down = (target[0],'count')).sort_values('Date').iloc[-validation_size:,].head(1)['Date'].values[0]
        
        train_data = self.data[self.data['Date'] < val_date].dropna()
        val_data = self.data[self.data['Date'] >= val_date].dropna()

        columns = [ x for x in train_data.columns if x not in target ]
        X_train, y_train = train_data[columns], train_data[target]
        X_val, y_val = val_data[columns], val_data[target]
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
    
    def train_model(self, pipe, model, cv_ = False):
        """
        merge and train pipeline and machine learning model

        Parameters
        ----------
        pipe (int): sklearn pipeline object
        model (list): model

        Returns
        -------
        None
        """
        self.model = model
        self.pipe_transform = pipe
        self.pipeline = Pipeline([('pipe_transform',self.pipe_transform), ('model',self.model)])
        self.features_to_model = self.pipe_transform.fit_transform(self.X_train).columns
        self.pipeline.fit(self.X_train, self.y_train)

class register_results():
    """
    class that collects model metrics

    Attributes
    ----------
    model_name : str
        model name
    metric_logger : diot
        dictionary that collect model metrics

    Methods
    -------
    eval_metrics(pipeline=obj, X=pd.DataFrame, y=pd.DataFrame, type_data=str, phase=str):
        register model metrics
    print_metric_logger():
        print logger results
    """
    def __init__(self, model_name):
        """
        Initialize object

        Parameters
        ----------
        model_name (str): model name

        Returns
        -------
        None
        """
        self.model_name = model_name
        self.metric_logger = dict()
    def eval_metrics(self, pipeline, X, y, type_data, phase):
        """
        register model metrics

        Parameters
        ----------
        pipeline (obj): model pipeline
        X (pd.DataFrame): input data
        Y (pd.DataFrame): target data
        type_data (str): data type, either train, test or validation
        phase (str): model phase, either baseline, feature selection, tunned model

        Returns
        -------
        None
        """
        preds_proba = pipeline.predict_proba(X)
        preds = pipeline.predict(X)
    
        if type(preds_proba) == list:
            preds_proba = np.array([ x[:,1]  for x in preds_proba]).T

        roc = roc_auc_score(y,preds_proba, average=None)
        precision = precision_score(y,preds, average=None)
        recall = recall_score(y,preds, average=None)
        
        self.metric_logger[f'{phase}//{self.model_name}//{type_data}'] = {'roc':roc, 'precision':precision, 'recall':recall}

    def print_metric_logger(self):
        """
        print logger results

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        parts = list(self.metric_logger.keys())
        phase_parts = [ x.split('//')[0] for x in parts]
    
        parts = list(self.metric_logger)
        phase_parts = [ x.split('//')[0] for x in parts]
        
        init_phase = phase_parts[0]
        print(f'---{init_phase}--')
        for phase,val in zip(phase_parts,self.metric_logger):
            stage = val.split('//')[2]
            if init_phase != phase:
                print(f'---{phase}--')
                init_phase = phase
            for metric in self.metric_logger[val]:
                print(stage, metric,self.metric_logger[val][metric])


def eval_metrics(pipeline, X, y, type_data, model_name):
    '''
    print metrics from a model pipeline

            Parameters:
                    pipeline (obj): model pipeline
                    X (pd.DataFrame): input data
                    Y (pd.DataFrame): target data
                    type_data (str): data type, either train, test or validation
                    model_name (str): model name

            Returns:
                    objects (dict): that contains ml artifacts, data , configs and models
    '''
    preds_proba = pipeline.predict_proba(X)
    preds = pipeline.predict(X)

    if type(preds_proba) == list:
        preds_proba = np.array([ x[:,1]  for x in preds_proba]).T
            
    print(f'--{type_data} - {model_name}--')
    print('--target: down, up--')
    print('--roc-auc--')
    print(roc_auc_score(y,preds_proba, average=None))
    print('--precision--')
    print(precision_score(y,preds, average=None))
    print('--recall--')
    print(recall_score(y,preds, average=None))


def data_processing_pipeline_classifier(
        features_base,features_to_drop = False, winsorizer_conf = False, discretize_columns = False,
        bins_discretize = 10, correlation = 0.85, fillna = True,
        invhypervolsin_features = False,
        date_features_list = False,
        pipeline_order = 'selector//winzorizer//discretizer//median_inputer//drop//correlation'
        ):

    '''
    pipeline builder

            Parameters:
                    features_base (list): model pipeline
                    features_to_drop (list): features to drop list
                    winsorizer_conf (dict): winsorising configuration dictionary
                    discretize_columns (list): feature list to discretize
                    bins_discretize (int): number of bins to discretize
                    correlation (float): correlation threshold to discard correlated features
                    fillna (boolean): if true to fill na features
                    invhypervolsin_features (list): list of features to apply inverse hyperbolic sine
                    date_features_list (list): list of features to compute from Date field. (list of features from feature_engine)
                    pipeline_order (str): custom pipeline order eg. selector//winzorizer//discretizer//median_inputer//drop//correlation
            Returns:
                    pipe (obj): pipeline object
    '''
    select_pipe = [('selector', FeatureSelector(features_base))] if features_base else []
    winzorizer_pipe = [('winzorized_features', VirgoWinsorizerFeature(winsorizer_conf))] if winsorizer_conf else []
    drop_pipe = [('drop_features' , DropFeatures(features_to_drop=features_to_drop))] if features_to_drop else []
    discretize = [('discretize',EqualWidthDiscretiser(discretize_columns, bins = bins_discretize ))] if discretize_columns else []
    drop_corr = [('drop_corr', DropCorrelatedFeatures(threshold=correlation, method = 'spearman'))] if correlation else []
    median_imputer_pipe = [('median_imputer', MeanMedianImputer())] if fillna else []
    invhypersin_pipe = [('invhypervolsin scaler', InverseHyperbolicSine(features = invhypervolsin_features))] if invhypervolsin_features else []
    datetimeFeatures_pipe = [('date features', DatetimeFeatures(features_to_extract = date_features_list, variables = 'Date', drop_original = False))] if date_features_list else []

    pipe_dictionary = {
        'selector': select_pipe,
        'winzorizer':winzorizer_pipe,
        'drop':drop_pipe,
        'discretizer': discretize,
        'correlation': drop_corr,
        'median_inputer':median_imputer_pipe,
        'arcsinh_scaler': invhypersin_pipe,
        'date_features': datetimeFeatures_pipe,
    }

    pipeline_steps = pipeline_order.split('//')
    ## validation
    for step in pipeline_steps:
        if step not in pipe_dictionary.keys():
            raise Exception(f'{step} step not in list of steps, the list is: {list(pipe_dictionary.keys())}')
        
    pipeline_args = [ pipe_dictionary[step] for step in pipeline_steps]
    pipeline_args = list(itertools.chain.from_iterable(pipeline_args))
    pipe = Pipeline(pipeline_args)

    return pipe


class ExpandingMultipleTimeSeriesKFold:
    """
    class that creates a custom cv schema that is compatible with sklearn cv arguments.

    Attributes
    ----------
    df : pd.DataFrame
        dataset
    number_window : int
        number of train splits
    window_size : int
        window size data
    overlap_size : int 
        overlap size

    Methods
    -------
    split(X=pd.DataFrame, y=pd.DataFrame, groups=boolean):
        custom split procedure
    get_n_splits(X=pd.DataFrame, y=pd.DataFrame, groups=boolean):
        get number of splits
    """
    
    def __init__(self, df, window_size = 100, number_window=3, overlap_size = 0):
        """
        Initialize object

        Parameters
        ----------
        df (pd.DataFrame): dataset
        number_window (int): number of train splits
        window_size (int): window size data
        overlap_size (int): overlap size

        Returns
        -------
        None
        """
        self.df = df
        self.number_window = number_window
        self.window_size = window_size
        self.overlap_size = overlap_size
        
    def split(self, X, y, groups=None):
        """
        custom split procedure

        Parameters
        ----------
        X (pd.DataFrame): input data (required for sklearn classes)
        y (pd.DataFrame): target data (required for sklearn classes)
        groups (boolean): to apply groups (required for sklearn classes)

        Returns
        -------
        None
        """
        if 'Date_i' not in self.df.index.names or 'i' not in self.df.index.names:
            raise Exception('no date and/or index in the index dataframe')
        
        if self.overlap_size > self.window_size:
            raise Exception('overlap can not be higher than the window size')

        unique_dates = list(self.df.index.get_level_values('Date_i').unique())
        unique_dates.sort()
    
        total_test_size = self.window_size * self.number_window
        total_test_size = total_test_size - (self.number_window - 1)*self.overlap_size
        
        if total_test_size > len(unique_dates):
            raise Exception('test size is higher than the data length')

        cut = total_test_size
        for fold in range(self.number_window):
            
            topcut = cut-self.window_size
            train_dates = unique_dates[:-cut]
            test_dates = unique_dates[-cut:-topcut]
            
            if topcut == 0:
                test_dates = unique_dates[-cut:]
        
            max_train_date = max(train_dates)
            min_test_date, max_test_date = min(test_dates), max(test_dates)
            
            cut = cut - (self.window_size - self.overlap_size) 
        
            train_index = self.df[self.df.index.get_level_values('Date_i') <= max_train_date].index.get_level_values('i')
            test_index = self.df[(self.df.index.get_level_values('Date_i') >= min_test_date) & (self.df.index.get_level_values('Date_i') <= max_test_date)].index.get_level_values('i')
        
            yield train_index, test_index

    def get_n_splits(self, X, y, groups=None):
        """
        get number of splits

        Parameters
        ----------
        X (pd.DataFrame): input data (required for sklearn classes)
        y (pd.DataFrame): target data (required for sklearn classes)
        groups (boolean): to apply groups (required for sklearn classes)

        Returns
        -------
        number_window (int): number of splits
        """
        return self.number_window