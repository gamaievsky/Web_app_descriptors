# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
from dash import dcc, html, dash_table, Input, Output, State, MATCH, ALL
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd


app = dash.Dash(__name__)

# df = pd.DataFrame({
#     # "Onsets": [0., 0.75, 2.5, 3.],
#     "audio":['Audio 1','Audio 1','Audio 1','Audio 1','Audio 2','Audio 2','Audio 2','Audio 2'],
#     "index":[1,2,3,4,1,2,3,4],
#     # 'durations':[1000, 500, 2000, 300],
#     "harmonicity": [0.5, 0.75, 0.1, 1, 0.3, 0.1, 0.9, 0.8],
#     "roughness": [0.8, 0.45, 0.2, 1, 0.7,0.6,0.2,0.5]
# })
#
#
# fig = px.scatter(df, x="harmonicity", y="roughness", text="index", color='audio', range_x=[0,1.1], range_y=[0,1.1])
# fig.update_traces(textposition="bottom right")
#
# app.layout = html.Div([
#     html.Big('Test Graphique'),
#     dcc.Graph(
#         id='example-graph',
#         figure=fig,
#         style={
#             'width': '600px',
#             'height': '500px',
#             'lineHeight': '30px'
#         },
#     )
# ])

app.layout = html.Div([
    html.Div([
        dcc.Input(
            id={'type': 'name_elt', 'temp':k},
            value=name,
            style={'display': 'inline-block'}

        ),
        dcc.RadioItems(
            id={'type': 'show','temp': k+1},
            options=[
                {'label': 'Show', 'value': 'show'},
                {'label': 'Hide', 'value': 'hide'},
            ],
            value='show',
            labelStyle={'display': 'inline-block'},
            style={'display': 'inline-block'}
        )
    ])
    for (k, name) in enumerate(['Accord 1', 'Accord 2'])
])


# fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 1000
# fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 5
#
if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_hot_reload=False)
