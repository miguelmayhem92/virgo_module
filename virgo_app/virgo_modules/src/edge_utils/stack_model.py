import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin

class MyStackingClassifierMultiClass(BaseEstimator, ClassifierMixin):
    def __init__(self,  estimators, meta_estimators,targets,perc=None,stack_size=None, **kwargs):
        self.estimators = estimators
        self.meta_estimators = meta_estimators
        self.targets = targets
        if stack_size and perc:
            raise Exception('just one option')
        if not stack_size and not perc:
            raise Exception('set one option')
        self.stack_size = stack_size
        self.perc = perc
    def get_index_training(self, X):
        if self.stack_size:
            unique_dates = list(X.index.get_level_values('Date_i').unique())
            unique_dates.sort()
            stack_chunk = unique_dates[-self.stack_size:]
            base_indexes = X[~X.index.get_level_values('Date_i').isin(stack_chunk)].index.get_level_values('i')
            meta_indexes = X[X.index.get_level_values('Date_i').isin(stack_chunk)].index.get_level_values('i')
        elif self.perc:
            meta_indexes = X.sample(frac = self.perc).index.get_level_values('i')
            base_indexes = X[~X.index.get_level_values('i').isin(meta_indexes)].index.get_level_values('i')
        else:
            raise Exception("error", self.stack_size, self.perc)
        return base_indexes, meta_indexes
    def train_base_learner(self, classifier, X, y,indexes):
        base_X = X[X.index.get_level_values('i').isin(indexes)]
        base_y = y[y.index.get_level_values('i').isin(indexes)]
        classifier.fit(base_X, base_y)
    def fit(self, X, y):
        # #base learners
        base_indexes, meta_indexes = self.get_index_training(X)
        for name,estimator in self.estimators:
            self.train_base_learner(estimator,X, y, base_indexes)
    
        #stack meta learner
        metas_pred = dict()
        for i,cont in enumerate(self.estimators, start=1):
            _,estimator = cont
            meta_pred = estimator.predict_proba(X[X.index.get_level_values('i').isin(meta_indexes)])
            metas_pred[f"meta{i}0"] = meta_pred[0][:,1]
            metas_pred[f"meta{i}1"] = meta_pred[1][:,1]
        meta_preds_df = pd.DataFrame(metas_pred)
    
        for i,metaest in enumerate(self.meta_estimators,start=0):
            _,metaest = metaest
            metacols = [f"meta{j}{i}" for j in range(1,len(self.estimators)+1)]
            metaest.fit(
                meta_preds_df[metacols],
                y[X.index.get_level_values('i').isin(meta_indexes)][self.targets[i]]
            )
        
    def predict_proba(self, X):
        metas_pred = dict()
        for i,cont in enumerate(self.estimators, start=1):
            _,estimator = cont
            meta_pred = estimator.predict_proba(X)
            metas_pred[f"meta{i}0"] = meta_pred[0][:,1]
            metas_pred[f"meta{i}1"] = meta_pred[1][:,1]
        self.meta_preds_df__ = pd.DataFrame(metas_pred)

        prediction_vector = list()
        for i,cont in enumerate(self.meta_estimators, start=0):
            _,estimator = cont
            metacols = [f"meta{j}{i}" for j in range(1,len(self.estimators)+1)]
            preds = estimator.predict_proba(self.meta_preds_df__[metacols].values)
            prediction_vector.append(preds)
        return prediction_vector
        
    def predict(self, X):
        prediction_vector = list()
        _ = self.predict_proba(X)
        for i,cont in enumerate(self.meta_estimators, start=0):
            _,estimator = cont
            metacols = [f"meta{j}{i}" for j in range(1,len(self.estimators)+1)]
            preds = estimator.predict(self.meta_preds_df__[metacols].values)
            prediction_vector.append(preds) 
        
        p = np.array(tuple(prediction_vector))
        return p.reshape((p.shape[1],p.shape[0]))
        
    def get_params(self, deep=True):
        return {k:v for k, v in self.__dict__.items()}
        
    def set_params(self, **parms):
        for k,v in parms.items():
            setattr(self,k,v)