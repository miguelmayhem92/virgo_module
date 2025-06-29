import shap
import mlflow
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

class StackInterpretor(mlflow.pyfunc.PythonModel):
    def __init__(self, model, targets):
        self.base_estimators = model.estimators_
        self.targets = targets
    def fit_interpretor(self, data):
        interpretors = {}
        for label, predictor in zip(self.targets,self.base_estimators):
            explainer = shap.Explainer(predictor, data)
            interpretors[label] = explainer
        self.interpretors = interpretors
    def get_shap_values(self, data):
        shap_values = dict()
        for label, interpretor in self.interpretors.items():
            shap_value = interpretor(data)
            shap_values[label] = shap_value
        return shap_values
    def register_map(self, mapping):
        self.mapping = mapping

def mean_shap(data, explainers, pipe_transform):
    t_data = pipe_transform.transform(data)
    input_features = t_data.columns
    shap_results = explainers.get_shap_values(t_data)
    dict_shap_values = explainers.mapping
    arrays_ = list()
    for k,_ in shap_results.items():
        arrays_.append(shap_results.get(k).values)
    shap_results_mean = np.mean(np.array(arrays_), axis = 0)
    df_shap = pd.DataFrame(shap_results_mean, columns=input_features, index=data.index)
    df_shap['Close'] = data['Close']
    df_shap['Date'] = data['Date']
    df_shap = df_shap[['Date','Close']+list(dict_shap_values.keys())]
    df_shap = df_shap.rename(columns =dict_shap_values)
    return df_shap

def edge_shap_lines(data, plot = False, look_back = 750):
    ### corect labels ####
    shap_cols = [col for col in data.columns if col not in ['Date','Close']]
    df = data.sort_values('Date').iloc[-look_back:]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df.Date, y=df.Close,mode='lines+markers',marker = dict(color = 'grey'),line = dict(color = 'grey'),name='Close price'))
    for col in shap_cols:
        fig.add_trace(go.Scatter(x=df.Date, y=df[col],mode='lines+markers',name=col),secondary_y=True)
    fig.update_layout(title_text="sirius - feature power",width=1200,height = 500)
    if plot:
        fig.show()
    return fig