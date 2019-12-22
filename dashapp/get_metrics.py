# -*- coding: utf-8 -*-
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.figure_factory as ff

import numpy as np
import pandas as pd
import logging
import os
import cloudstorage as gcs
import webapp3
from google.cloud import storage
from io import BytesIO
import gcsfs

import socket
import psycopg2

import get_metrics

global metrics_df

#######################################################################
max_rows=10
left_margin = 200
right_margin = 100
#######################################################################
##reading from sql
try:
    conn = psycopg2.connect(host='34.85.27.74', port='5432', database='Embryonics_DB', user='postgres', password = 'password')
    c = conn.cursor()
    c.execute('select DISTINCT "model_id" FROM "Fact_Model_Results"  limit all;')
    rows = c.fetchall()
    models_list= [row[0] for row in rows]
    c.close()

except socket.herror:
        print (u"Unknown host")

#######################################################################
def generate_table(dataframe, max_rows=10):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )
#######################################################################

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
#######################################################################

app.layout = html.Div([

    ### title & subtitle
    html.Div(children=[
        html.H1(
            children='Welcome to Embryonics dashbodrad', style={'textAlign': 'center'}
            , className="row"),
        html.Div(children='this is just me trying stuff on dash.',
                 style={'textAlign': 'center','font-size': '26px'})
    ],className="row"),

    ### models selection row
    html.Div([

        ### ROC models selection
        html.Div([
            dcc.Dropdown(
                id='mult_model_drop_down',
                options=[{'label': '{}'.format(model), 'value': '{}'.format(model)} for model in models_list],
                value='1',
                multi=True,
                placeholder="Select models",
                style={'height': '40px', 'width': '300px', 'margin-left': '100px',
                       'margin-top': '20px',},
            ),
            html.Div(id='model_select_text',
                     style={'height': '50px', 'width': '400px', 'margin-left': '200px',
                            'margin-top': '20px','font-size': '26px'},
                     )], className="six columns"),

        ### metrics model selection
        html.Div([
            dcc.Dropdown(
                id='model_show_drop_down',
                value='1',
                multi=False,
                placeholder="Select model for info",
                style={'height': '40px', 'width': '300px', 'margin-left': '300px',
                       'margin-top': '20px','margin-bottom': '5px'},
            )], className="six columns"),
    ], className="row"),

    ### hole page
    html.Div([

        ###ROC
        html.Div([
            dcc.Graph(
                id='ROC',
                style={'height': '200%', 'width': '100%', 'display': 'inline-block',
                       'vertical-align': 'middle', 'margin-left': '100px', 'font-size': '26px'},
            ),
        ], className="six columns"),
        ### add metric nex to ROC

    ], className="row"),

    ### matrix and slider row
    html.Div([

        ### slider notoation
        html.Div(id='slider_notation',
                 style={'width': '20%', 'display': 'inline-block',
                        'margin-top': '50px', 'margin-left': '22%'},
                 className="six columns"),

        ###confusion matrix
        html.Div([dcc.Graph(id='heatmap_output')],
                 style={'width': '50%', 'display': 'inline-block',
                        'margin-top': '10px', 'margin-left': '10px'},
                 className="six columns"),

        ### threshold slider
        html.Div([
            dcc.Slider(
                id='slider',
                min=0,
                max=1,
                step=0.01,
                value=0.5,
                marks={i / 10: '{}'.format(i / 10) for i in range(11)}
            )
        ], style={'width': '800px', 'display': 'inline-block',
                  'margin-top': '10px', 'margin-left': '10px'},
            className="six columns"),

    ],className="row"),

])


#######################################################################
#
#       selecting model for plot
#
@app.callback(
    [Output('model_select_text', 'children'),
     Output('ROC', 'figure'),
     Output('slider_notation', 'children'),
     Output('model_show_drop_down', 'options')],
    [Input('mult_model_drop_down', 'value'),
     Input('slider', 'value')])
def update_output(models_selected,slider_value):
    if models_selected:
        query = 'SELECT * FROM "Fact_Model_Results" WHERE "model_id" IN ({});'.format(','.join(models_selected))
        metrics_df = create_metrics_df(models_selected, conn,query)
        return 'You have selected model/s "{}"'.format(models_selected),\
            create_ROC_graph(models_selected,metrics_df),\
            'Selected threshold value = {}'.format(slider_value), \
            [{'label': '{}'.format(model), 'value': '{}'.format(model)} for model in models_selected]

#######################################################################

@app.callback(
     Output('heatmap_output', 'figure'),
    [Input('model_show_drop_down', 'value'),
     Input('slider', 'value')])
def update_output(selected_model, slider_value):
    query = 'SELECT * FROM "Fact_Model_Results" WHERE "model_id" IN ({});'.format(','.join(selected_model))
    metrics_df = create_metrics_df(selected_model, conn, query)
    return creating_conf_fig(slider_value, metrics_df, selected_model)

#######################################################################
#
#       creating table based on chosen models
#
def creating_conf_fig(value,df,selected_model):
    dff = create_conf_mat(df, value, selected_model)
    z =  dff.values.T.tolist()
    x =  dff.index.tolist()
    y = dff.columns.tolist()
    z_text = [['tp = {}'.format(z[0][0]), 'fn = {}'.format(z[0][1])],
              ['fp = {}'.format(z[1][0]), 'tn = {}'.format(z[1][1])]]
    fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text,
                                       colorscale='Viridis', reversescale=True)
    fig['layout']['yaxis']['autorange'] = "reversed"
    return fig
#######################################################################
def create_conf_mat (df,thresh,model):
    out_df = df.loc[df['threshs'] == thresh, ['model_id', 'tp', 'tn', 'fp', 'fn']]
    out_df = out_df.loc[out_df['model_id'] == str(model)]
    d = {'P': [int(out_df['tp']),
               int(out_df['fn'])],
         'N': [int(out_df['fp']),
               int(out_df['tn'])]}
    conf_mat = pd.DataFrame(data=d,index=['P', 'N'])
    return conf_mat
#######################################################################
#
#       creating table based on chosen models
#
def create_metrics_df(value,conn,query):
    if value:
        try:
            c2 = conn.cursor()
            c2.execute(query)
            df = pd.DataFrame(data=c2.fetchall(), columns=[desc[0] for desc in c2.description])
            c2.close()
            #####
            gt = np.random.rand(1, df.count(axis='rows')[1])
            #####
            df.insert(4, 'gt', gt.T)
            metrics_df=get_metrics.metrics(df)
        except socket.herror:
            print(u"Unknown host")
    return metrics_df
#######################################################################
#
#       func ROC graph
#
def create_ROC_graph(value,df):
    return({
         'data': [
             dict(
                 x=df[df['model_id'] == i]['fpr'],
                 y=df[df['model_id'] == i]['tpr'],
                 text=['acc', df[df['model_id'] == i]['acc']],
                 mode='line',
                 opacity=0.7,
                 marker={
                     'size': 30,
                     'line': {'width': 0.5, 'color': 'white'}
                 },
                 name=i
             ) for i in value
         ],
         'layout': dict(
             xaxis={'title': 'fpr'},
             yaxis={'title': 'tpr'},
             margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
             legend={'x': 0, 'y': 1},
             hovermode='closest'
         )
    })
#######################################################################
if __name__ == '__main__':
    app.run_server(debug=True)
    #app.run_server(dev_tools_hot_reload=False)