import numpy as np
import itertools

from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from feature_engine.selection import DropFeatures, DropCorrelatedFeatures
from feature_engine.imputation import  MeanMedianImputer
from virgo_modules.src.ticketer_source import FeatureSelector
from feature_engine.discretisation import EqualWidthDiscretiser

from .ticketer_source import VirgoWinsorizerFeature

class produce_model_wrapper:
    def __init__(self,data):
        self.data = data.copy()
    
    def preprocess(self, validation_size, target):
        
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
        self.model = model
        self.pipe_transform = pipe
        self.pipeline = Pipeline([('pipe_transform',self.pipe_transform), ('model',self.model)])
        self.features_to_model = self.pipe_transform.fit_transform(self.X_train).columns
        self.pipeline.fit(self.X_train, self.y_train)

class register_results():
    def __init__(self, model_name):
        self.model_name = model_name
        self.metric_logger = dict()
    def eval_metrics(self, pipeline, X, y, type_data, phase):
        
        preds_proba = pipeline.predict_proba(X)
        preds = pipeline.predict(X)
    
        if type(preds_proba) == list:
            preds_proba = np.array([ x[:,1]  for x in preds_proba]).T

        roc = roc_auc_score(y,preds_proba, average=None)
        precision = precision_score(y,preds, average=None)
        recall = recall_score(y,preds, average=None)
        
        self.metric_logger[f'{phase}//{self.model_name}//{type_data}'] = {'roc':roc, 'precision':precision, 'recall':recall}

    def print_metric_logger(self):
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


def data_processing_pipeline_classifier(features_base,features_to_drop = False, winsorizer_conf = False, discretize_columns = False,
                                         bins_discretize = 10, correlation = 0.85, fillna = True,
                                        pipeline_order = 'selector//winzorizer//discretizer//median_inputer//drop//correlation'):


    select_pipe = [('selector', FeatureSelector(features_base))] if features_base else []
    winzorizer_pipe = [('winzorized_features', VirgoWinsorizerFeature(winsorizer_conf))] if winsorizer_conf else []
    drop_pipe = [('drop_features' , DropFeatures(features_to_drop=features_to_drop))] if features_to_drop else []
    discretize = [('discretize',EqualWidthDiscretiser(discretize_columns, bins = bins_discretize ))] if discretize_columns else []
    drop_corr = [('drop_corr', DropCorrelatedFeatures(threshold=correlation, method = 'spearman'))] if correlation else []
    median_imputer_pipe = [('median_imputer', MeanMedianImputer())] if fillna else []

    pipe_dictionary = {
        'selector': select_pipe,
        'winzorizer':winzorizer_pipe,
        'drop':drop_pipe,
        'discretizer': discretize,
        'correlation': drop_corr,
        'median_inputer':median_imputer_pipe,
    }

    pipeline_steps = pipeline_order.split('//')
    ## validation
    for step in pipeline_steps:
        if step not in pipe_dictionary.keys():
            raise Exception(f'{step} step not in list of steps, the list is: {list(pipe_dictionary.keys())}')
        
    pipeline_args = [ pipe_dictionary[step] for step in pipeline_steps]
    pipeline_args = list(itertools.chain.from_iterable(pipeline_args))
    pipe = Pipeline(pipeline_args)
    # pipe = Pipeline(
    #     select_pipe + \
    #     winzorizer_pipe + \
    #     discretize + \
    #     median_imputer_pipe + \
    #     drop_pipe + \
    #     drop_corr
    # )
    return pipe


class ExpandingMultipleTimeSeriesKFold:
    """increasing training window where the test can be overlap"""
    def __init__(self, df, window_size = 100, number_window=3, overlap_size = 0):
        self.df = df
        self.number_window = number_window
        self.window_size = window_size
        self.overlap_size = overlap_size
        
    def split(self, X, y, groups=None):
        
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
        return self.number_window