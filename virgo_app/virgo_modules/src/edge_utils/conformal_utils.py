from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.pipeline import Pipeline
import mlflow
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from mapie.classification import SplitConformalClassifier

class ConformalStack(mlflow.pyfunc.PythonModel):
    def __init__(self, model,targets, alphas):
        self.model = model
        self.targets = targets
        self.alphas = alphas
    def fit(self, data):
        self.classifiers = dict()
        for i,target in enumerate(self.targets):
            st = SingleStack(self.model["model"],i)
            st.fit()
            seg_model = Pipeline([ 
                ('pipe',self.model['pipe_transform']),
                ('modelbase',st)
            ])
            mapie_class = SplitConformalClassifier(seg_model, prefit=True, random_state=123, conformity_score="lac", confidence_level=1-np.array(self.alphas))
            mapie_class.conformalize(data, data[self.targets[i]].values)
            self.classifiers[target] = mapie_class
    def predict_conformal(self, data, ):
        for target in self.targets:
            prefix = target+"_conf"
            _, y_pis = self.classifiers[target].predict_set(data)
            for i,alpha in enumerate(self.alphas):
                data[f'{prefix}-{alpha}'] = y_pis[:,1,i]
                data[f'{prefix}-{alpha}'] = np.where(data[f'{prefix}-{alpha}'] == True,alpha,0)
        return data
    

class SingleStack(ClassifierMixin, BaseEstimator):
    def __init__(self, model, estimator_index):
        self.model = model
        self.estimator_index = estimator_index
        
    def fit(self):
        self._is_fitted = True
        self.classes_ = [0,1]

    def predict_proba(self, X):
        metas_pred = dict()
        for i,cont in enumerate(self.model.estimators, start=1):
            _,estimator = cont
            meta_pred = estimator.predict_proba(X)
            metas_pred[f"meta{i}0"] = meta_pred[0][:,1]
            metas_pred[f"meta{i}1"] = meta_pred[1][:,1]
        self.meta_preds_df__ = pd.DataFrame(metas_pred)

        prediction_vector = list()
        for i,cont in enumerate(self.model.meta_estimators, start=0):
            _,estimator = cont
            metacols = [f"meta{j}{i}" for j in range(1,len(self.model.estimators)+1)]
            preds = estimator.predict_proba(self.meta_preds_df__[metacols].values)
            prediction_vector.append(preds)
        return prediction_vector[self.estimator_index]
        
    def predict(self, X):
        prediction_vector = list()
        _ = self.predict_proba(X)
        for i,cont in enumerate(self.model.meta_estimators, start=0):
            _,estimator = cont
            metacols = [f"meta{j}{i}" for j in range(1,len(self.model.estimators)+1)]
            preds = estimator.predict(self.meta_preds_df__[metacols].values)
            prediction_vector.append(preds) 
        
        p = np.array(tuple(prediction_vector))
        return p.reshape((p.shape[1],p.shape[0]))[:,self.estimator_index]
    
    def __sklearn_is_fitted__(self):
        return hasattr(self, "_is_fitted") and self._is_fitted

def edge_conformal_lines(data, alphas,threshold = 0.6, plot = False, look_back = 750, offset = 0.08):
    ### corect labels ####
    df = data.sort_values('Date').iloc[-look_back:]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df.Date, y=df.Close,mode='lines+markers',marker = dict(color = 'grey'),line = dict(color = 'grey'),name='Close price'))
    fig.add_trace(go.Scatter(x=df.Date, y=df.proba_target_up,mode='lines',marker = dict(color = 'blue'),showlegend=True,legendgroup='go up', name='go up'),secondary_y=True)
    fig.add_trace(go.Scatter(x=df.Date, y=df.proba_target_down,mode='lines',marker = dict(color = 'coral'),showlegend=True,legendgroup='go down',name='go down'),secondary_y=True)
    for i,alpha in enumerate(alphas, start=1):
        try:
            col_alpha = [x for x in df.columns if str(alpha) in x and 'target_up' in x][0]
            df_ = df[df[col_alpha] != 0]
            fig.add_trace(go.Scatter(x=df_.Date, y=df_.proba_target_up + (offset*i),mode='markers',marker = dict(opacity=0.7,size=10, color = 'blue')
                                     ,showlegend=False,legendgroup='go up',name='go up', text=df_[col_alpha],textposition="bottom center")
                                     , secondary_y=True)
        except:
            pass
        try:
            col_alpha = [x for x in df.columns if str(alpha) in x and 'target_down' in x][0]
            df_ = df[df[col_alpha] != 0]
            fig.add_trace(go.Scatter(x=df_.Date, y=df_.proba_target_down + (offset*i),mode='markers',marker = dict(opacity=0.7,size=10, color = 'coral')
                                     ,showlegend=False,legendgroup='go down', name='go down',text=df_[col_alpha].astype(str),textposition="bottom center")
                                     , secondary_y=True)
        except:
            pass
    fig.add_shape(type="line", xref="paper", yref="y2",x0=0.02, y0=threshold, x1=0.9, y1=threshold,line=dict(color="red",dash="dash"))
    fig.update_layout(title_text="sirius - edge probabilities conformal",width=1200,height = 500)
    if plot:
        fig.show()
    return fig