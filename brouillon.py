# import dash
# import dash_core_components as dcc
# import dash_html_components as html
# from dash.dependencies import Input, Output, State, MATCH, ALL
#
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#
# app = dash.Dash(__name__, external_stylesheets = external_stylesheets)
#
# app.layout = html.Div([
#     dcc.Slider(
#         id='my-slider',
#         min=0,
#         max=20,
#         step=0.5,
#         value=10,
#     ),
# ])
#
# if __name__ == '__main__':
#    app.run_server(debug = True)


import librosa
print(librosa.midi_to_note(range(12,132)))
