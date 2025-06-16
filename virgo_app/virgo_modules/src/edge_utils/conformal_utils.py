from plotly.subplots import make_subplots
import plotly.graph_objects as go
from mapie.classification import MapieClassifier
from sklearn.pipeline import Pipeline
import mlflow
import numpy as np


def get_conformal_classifiers(model, data, targets):
    classfiers = list()
    for i, _ in enumerate(model['model'].estimators_):
        seg_model = Pipeline([ 
            ('pipe',model['pipe_transform']),
            ('model',model['model'].estimators_[i])
        ])
        mapie_class = MapieClassifier(seg_model, cv='prefit', random_state=123, method="lac")
        mapie_class.fit(data, data[targets[i]].values)
        classfiers.append(mapie_class)
    return classfiers

def log_confmodels(runid, classifiers):
    with mlflow.start_run(run_id=runid) as run:
        for i,classifier in enumerate(classifiers):
            mlflow.sklearn.log_model(classifier,name = f"conformal_model-{i}")
        print('models were logged')

def load_confmodel(runid, target_variables):
    classifiers = list()
    for i in range(len(target_variables)):
        folder = f"conformal_model-{i}"
        model = mlflow.sklearn.load_model(f"runs:/{runid}/{folder}",)
        classifiers.append(model)
    return classifiers


def get_conformal_prediction(classifier, alphas, data, prefix='conf'):
    _, y_pis = classifier.predict(data, alpha=alphas)
    for i,alpha in enumerate(alphas):
        data[f'{prefix}-{alpha}'] = y_pis[:,1,i]
        data[f'{prefix}-{alpha}'] = np.where(data[f'{prefix}-{alpha}'] == True,alpha,0)
    return data

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