from datetime import datetime
#import sanity_tests
import sys, os, glob
import csv
import math
#import cv2
#import h5py
import numpy as np
import pandas as pd
import plotly.figure_factory as ff

from pandas import read_csv
import os
from pandas import ExcelWriter

import socket
import psycopg2


from google.cloud import storage
import logging
import os
import cloudstorage as gcs
#import webapp2




def calc_metrics(pred,gt,thresh):
    epsilon=1e-7
    tp = sum(np.where(pred > thresh, 1, 0) & gt)
    fp = sum(np.where(pred > thresh, 1, 0) & (1- gt))
    tn = sum(np.where(pred < thresh, 1, 0) & gt)
    fn = sum(np.where(pred < thresh, 1, 0) & (1-gt))
    tpr = recall = tp/(tp+fn+epsilon)#recall/sensitivity
    fpr = fp/(tn+fp+epsilon)
    acc=(tn+tp)/len(pred)
    precision=tp/(tp+fp+epsilon)
    f1=2*(precision*recall)/(precision+recall+epsilon)
    return tp, tn, fp, fn, tpr, fpr, acc, precision, f1
#######################################################################
def metrics(df):
    output_headers = ['model_id', 'threshs', 'tp', 'tn', 'fp', 'fn', 'tpr', 'fpr', 'acc', 'precision', 'f1']
    #'AUC', 'AP@k-1','AP@k-2', 'AP@k-3']
    n_thresh=51
    threshs = np.array(np.array(np.linspace(0, 1, n_thresh)), ndmin=2).transpose()
    output_df=pd.DataFrame(columns=output_headers)
    model_ids=df.model_id.unique()
    for model in model_ids:
        inter_df = pd.DataFrame(np.zeros((threshs.size, len(output_headers))), index=None, columns=output_headers,
                                 dtype=None, copy=False)
        pred = df[df['model_id'] == model]['prediction_value'].to_numpy()
        gt = np.where(df[df['model_id'] == model]['gt'].to_numpy() > 0.5, 1, 0)

        for idx, thresh in enumerate(threshs):
                [tp, tn, fp, fn, tpr, fpr, acc, precision, f1] =calc_metrics(pred,gt,float(thresh))
                inter_df.iloc[idx] = [str(model), thresh, tp, tn, fp, fn, tpr, fpr, acc, precision, f1]
                                       #0, AP1, AP2, AP3]
        output_df = output_df.append(inter_df)
    return output_df
#######################################################################
#
#       creating table based on chosen models
#
def creating_conf_fig(value,df,selected_model):
    dff = create_conf_mat(df, value, selected_model)
    z = dff.values.T.tolist()
    x = dff.index.tolist()
    y = dff.columns.tolist()
    z_text = [['tp = {}'.format(int(z[0][0])), 'fn = {}'.format(int(z[0][1]))],
              ['fp = {}'.format(int(z[1][0])), 'tn = {}'.format(int(z[1][1]))]]
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
    conf_mat = pd.DataFrame(data=d, index=['P', 'N'])
    return conf_mat
#######################################################################
#
#       creating table based on chosen models
#
def create_metrics_df(value,query):
    if value:
        try:
            conn = psycopg2.connect(host='34.85.27.74', port='5432',
                                    database='Embryonics_DB', user='postgres',password='password')
            c2 = conn.cursor()
            c2.execute(query)
            df = pd.DataFrame(data=c2.fetchall(), columns=[desc[0] for desc in c2.description])
            #####
            # nan_value = float("NaN")
            # df["gt"] = nan_value

            uniq_gt_df = get_gt(df.embryo_id.unique(), conn)
            merged = df.merge(uniq_gt_df, on='embryo_id', how='left')
            #####
            metrics_df = metrics(merged)
            c2.close()

            return metrics_df
        except socket.herror:
            print(u"Unknown host")
    return
#######################################################################

def get_gt(embryo_id,conn):
    lst = [str(i) for i in embryo_id.tolist()]
    query = 'SELECT embryo_id, value FROM "Fact_Embryos_Ground_Truth" WHERE "embryo_id" IN ({});'.format(','.join(lst))
    try:
        c3 = conn.cursor()
        c3.execute(query)
        gt_df = pd.DataFrame(data=c3.fetchall(), columns=[desc[0] for desc in c3.description])
        gt_df.rename(columns={'value': 'gt'}, inplace=True)
        c3.close()
        #####
        return gt_df
    except socket.herror:
        print(u"Unknown host")
    return

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

