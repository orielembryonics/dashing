# -*- coding: utf-8 -*-

import dash
import dash_html_components as html




external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
#######################################################################

app.layout = html.Div(children=[
        html.H1(
            children='Welcome to Embryonics dashbodrad', style={'textAlign': 'center'}
            , className="row"),
        html.Div(children='this is just me trying stuff on dash.',
                 style={'textAlign': 'center','font-size': '26px'})
            ],className="row")


#######################################################################

if __name__ == '__main__':
    app.run_server(debug=True)
    #app.run_server(dev_tools_hot_reload=False)