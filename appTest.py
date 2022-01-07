# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
from dash import dcc, html, dash_table, Input, Output, State, MATCH, ALL
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd


app = dash.Dash(__name__)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
# df = pd.DataFrame({
#     "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
#     "Amount": [4, 1, 2, 2, 4, 5],
#     "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
# })
#
# fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

# df = pd.DataFrame({
#     # "Onsets": [0., 0.75, 2.5, 3.],
#     "index":[1,2,2,4],
#     'durations':[1000, 500, 2000, 300],
#     "harmonicity": [0.5, 0.75, 0.1, 1],
#     "roughness": [0.8, 0.45, 0.2, 1]
# })
#
# fig = px.scatter(df, x="harmonicity", y="roughness", text="index", animation_frame="index", range_x=[0,1.1], range_y=[0,1.1])
# fig.update_traces(textposition="bottom right")
# fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 1000
# fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 5


app.layout = html.Div(
    children=[
        html.H3("Input files"),
        html.Br(),
        html.Div(id='tabs_input_files'),
        dcc.Tabs(id='tabs', children=[]),
        html.Button(id='add_audio_input', n_clicks=0, children='Add Audio input'),
        html.Button(id='delete_audio_input', n_clicks=0, children='Delete Audio input')
    ]
)

@app.callback(
    Output('tabs','children'),
    Output('tabs','value'),
    Output('add_audio_input', 'n_clicks'),
    Input('add_audio_input', 'n_clicks'),
    Input('delete_audio_input', 'n_clicks'),
    State('tabs','children'))
def add_del_tab(add_audio, del_audio, children):
    ctx = dash.callback_context
    new_tab = dcc.Tab(
        label='Audio {}'.format(add_audio + 1),
        value='Audio {}'.format(add_audio + 1),
        id={'type': 'audio', 'index': add_audio + 1},
        # children='Input audio n°{}'.format(add_audio + 1)
        children=[
            html.Div([
                html.Big('Main sound file {}'.format(add_audio + 1)),
                dcc.Upload(
                    id={'type': 'main_sound', 'index': add_audio + 1},
                    children=html.Div(['Drag and drop or ', html.A('Select File')]),
                    style={'width': '30%','height': '40px','lineHeight': '30px','borderWidth': '1px','borderStyle': 'dashed','borderRadius': '5px','textAlign': 'center','margin': '10px','display': 'inline-block'},

                    multiple=False,
                    filename=''
                ),
                html.Div(id={'type': 'input1', 'index': add_audio + 1})
            ]),

            html.Div([
                html.Big('Separated audio tracks'),
                dcc.Upload(
                    id={'type': 'separated_tracks', 'index': add_audio + 1},
                    children=html.Div(['Drag and drop or ', html.A('Select File')]),
                    style={'width': '30%','height': '40px','lineHeight': '30px','borderWidth': '1px','borderStyle': 'dashed','borderRadius': '5px','textAlign': 'center','margin': '10px','display': 'inline-block'},
                    # Allow multiple files to be uploaded
                    multiple=True
                ),
                html.Div(id={'type': 'input2', 'index': add_audio + 1})
            ]),

            html.Div([
                html.Big('Onsets'),
                dcc.Upload(
                    id={'type': 'onsets', 'index': add_audio + 1},
                    children=html.Div(['Drag and drop or ', html.A('Select File')]),
                    style={'width': '30%','height': '40px','lineHeight': '30px','borderWidth': '1px','borderStyle': 'dashed','borderRadius': '5px','textAlign': 'center','margin': '10px','display': 'inline-block'},
                    multiple=False
                ),
                html.Div(id={'type': 'input3', 'index': add_audio + 1})
            ]),
            html.Div(id='sound')
        ]
    )

    if add_audio==0 or ctx.triggered[0]['prop_id']=='add_audio_input.n_clicks':
        print('create')
        children.append(new_tab)
        return children, 'Audio {}'.format(add_audio + 1), add_audio
    else:
        print('pop')
        children.pop()
        return children, 'Audio {}'.format(add_audio) ,(add_audio - 1)
    # if 'delete_audio_input' in ctx.triggered:
    #     if del_click > 0:
    #         children.pop()
    #         return children



@app.callback(
    Output('delete_audio_input','style'),
    Input('add_audio_input', 'n_clicks'))
def cache_button(add_audio):
    if add_audio==0:
        return {'display':'none'}
    else:
        return {'display':'inline-block'}

# @app.callback(
#     Output('tabs_input_files','children'),
#     Input('tabs', 'value'))
#     Input('tabs', 'children')
# def cache_button(add_audio):
#     if add_audio==0:
#         return {'display':'none'}
#     else:
#         return {'display':'inline-block'}



if __name__ == '__main__':
    app.run_server(debug=True)

            # dcc.Graph(
            #     id='example-graph',
            #     figure=fig,
            #     style={
            #         'width': '600px',
            #         'height': '500px',
            #         'lineHeight': '30px'
            #     },
            # )
