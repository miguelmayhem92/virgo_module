import random

from numpy.random import choice
import numpy as np
from scipy import stats
from sklearn.feature_selection import RFE

class StackRFE:
    def __init__(self, model, n_features, batch_elim, step_elim, cv, max_iterations, manual_pipe=list(), importance_callable="auto"):
        """
        n_features: number of features to select in RFE
        batch_elim: select n features as suggestion
        step_elim: number of features to iter in RFE
        manual_pipe: list of pipeline to suggest features to pass to RFE
        importance_callable: function to calculate feature importance
        """
        self.model = model
        self.n_features = n_features
        self.batch_elim = batch_elim
        self.step_elim = step_elim
        self.cv = cv
        self.max_iterations = max_iterations
        self.manual_pipe = manual_pipe
        self.importance_callable=importance_callable

    def _suggest_elimination(self, uniform=False):
        """
        suggest based on mean ranking, lower the mean rank higher the prob to be selected
        """
        ds = self.feature_rankings
        ds_mean = {k:np.mean(ds.get(k)) for k in ds}
        max_ = np.max([x for x in ds_mean.values()])
        ds_weight = {k: (max_-v+1) for k,v in ds_mean.items()}
        sum_ = np.sum([x for x in ds_weight.values()])
        ds_prob = {k: v/sum_ for k,v in ds_weight.items()}
        result = list(choice(list(ds_prob.keys()), self.batch_elim,p=list(ds_prob.values()), replace=False))
        if uniform:
            features = list(ds_prob.keys())
            random.shuffle(features)            
            result = features[0:self.batch_elim]
        return result
        
    def fit(self, X, y):
        features = list(X.columns).copy()
        self.feature_rankings = {f:[1] for f in features}
        for iteration in range(self.max_iterations):
            # shuffling
            if random.random() > 0.5:
                batch_features = self._suggest_elimination()
            else:
                batch_features = self._suggest_elimination(uniform=True)
                
            if len(self.manual_pipe)>0:
                batch_features = self.manual_pipe.pop(0)
            # selector and elimination
            tmp_feature_ranking = {k: list() for k in batch_features}
            selector = RFE(self.model, n_features_to_select=self.n_features, step=self.step_elim, importance_getter=self.importance_callable)
            for train_index, test_index in self.cv.split(X, y):
                X_ = X[X.index.get_level_values('i').isin(train_index)][batch_features]
                y_ = y[y.index.get_level_values('i').isin(train_index)]
                selector = selector.fit(X_, y_)
                for k,r in zip(tmp_feature_ranking.keys(), selector.ranking_):
                    tmp_feature_ranking[k].append(r)
            rankings = [np.median(v) for v in tmp_feature_ranking.values()]
            for f,r in zip(batch_features, rankings):
                self.feature_rankings[f].append(r)