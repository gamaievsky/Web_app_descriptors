# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go
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

df = pd.DataFrame({
    # "Onsets": [0., 0.75, 2.5, 3.],
    "number":[1,2,3,4],
    "harmonicity": [0.5, 0.75, 0.1, 1],
    "roughness": [0.8, 0.45, 0.2, 1],
})

fig = px.line(df, x="harmonicity", y="roughness", text="number")
fig.update_traces(textposition="bottom right")


app.layout = html.Div(children=[
    html.H1(children='Test de graphique avec PyPlot'),

    html.Div(children='''
        2-dimension harmonic trajectories
    '''),

    dcc.Graph(
        id='example-graph',
        figure=fig,
        style={
            'width': '600px',
            'height': '500px',
            'lineHeight': '30px'
        },
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
