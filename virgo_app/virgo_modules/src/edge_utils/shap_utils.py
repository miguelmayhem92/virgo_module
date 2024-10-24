import shap
import mlflow
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def get_explainers(model, data):
    explainers = list()
    for i, _ in enumerate(model['model'].estimators_):
        transf_data = model['pipe_transform'].transform(data)
        predictor = model['model'].estimators_[i]
        explainer= shap.Explainer(predictor, transf_data)
        explainers.append(explainer)
    return explainers

def log_explainer(runid, classifiers):
    with mlflow.start_run(run_id=runid) as run:
        for i,classifier in enumerate(classifiers):
            mlflow.sklearn.log_model(classifier,f"explainer/explainer-{i}")
        print('models were logged')

def load_explainer(runid, target_variables):
    explainers = list()
    for i in range(len(target_variables)):
        folder = f"explainer/explainer-{i}"
        model = mlflow.sklearn.load_model(f"runs:/{runid}/{folder}")
        explainers.append(model)
    return explainers

def get_shapvalues(explainers, data):
    shap_values = {}
    for i,explainer in enumerate(explainers):
        shap_value_i = explainer(data)
        shap_values[i] = shap_value_i
    return shap_values

def get_explainerclusters(model, data, targets):
    clustermodels = list()
    for i, _ in enumerate(model['model'].estimators_):
        transf_data = model['pipe_transform'].transform(data)
        Y = data[targets[i]]
        cluster_model = shap.utils.hclust(transf_data, Y)
        clustermodels.append(cluster_model)
    return clustermodels

def mean_shap(data, explainers, pipe_transform):
    t_data = pipe_transform.transform(data)
    input_features = t_data.columns
    shap_results = get_shapvalues(explainers,t_data)
    arrays_ = list()
    for k,_ in shap_results.items():
        arrays_.append(shap_results.get(k).values)
    shap_results_mean = np.mean(np.array(arrays_), axis = 0)
    df_shap = pd.DataFrame(shap_results_mean, columns=input_features, index=data.index)
    df_shap['Close'] = data['Close']
    df_shap['Date'] = data['Date']
    return df_shap

def edge_shap_lines(data, dict_shap_values, plot = False, look_back = 750):
    ### corect labels ####
    df = data.sort_values('Date').iloc[-look_back:]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df.Date, y=df.Close,mode='lines+markers',marker = dict(color = 'grey'),line = dict(color = 'grey'),name='Close price'))
    top_features_shap = dict_shap_values.keys()
    for col in top_features_shap:
        fig.add_trace(go.Scatter(x=df.Date, y=df[col],mode='lines+markers',name=dict_shap_values.get(col).get('tag')),secondary_y=True)
    fig.update_layout(title_text="sirius - feature power",width=1200,height = 500)
    if plot:
        fig.show()
    return fig