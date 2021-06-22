import base64
import datetime
import io

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
# import dash_table

import pandas as pd
import plotly.express as px
# import plotly.graph_objs as go


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-data-upload'),
])


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
            # Can't handle more than 1 column for graphing
            if len(df.columns) > 1:
                raise

    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    fig = px.scatter(df)
    fig.update_traces(mode='lines+markers')

    graphData = [
            html.H5(filename),
            html.H6(datetime.datetime.fromtimestamp(date)),

            # Graph of csv file
            dcc.Graph(
                figure=fig),
            dcc.Dropdown(
                id='dropdown_tags',
                options=[
                    {'label': 'Arc', 'value': 'Arc'},
                    {'label': 'Peaks', 'value': 'Peaks'},
                    {'label': 'Rings', 'value': 'Rings'},
                    {'label': 'Rods', 'value': 'Rods'}],
                multi=True,
                placeholder="Select Tags"),
            html.Button(id='save-labels', children='Save Labels to Disk'),
            html.Div(id='saved-response'),

            html.Hr(),  # horizontal line

            # For debugging, display the raw contents provided by the
            # web browser
            html.Div('Raw Content'),
            html.Pre(contents[0:200] + '...', style={
                'whiteSpace': 'pre-wrap',
                'wordBreak': 'break-all'
            })]

    return html.Div(graphData)


@app.callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children


@app.callback(Output('saved-response', 'children'),
              Input('save-labels', 'n_clicks'),
              State('dropdown_tags', 'value'))
def save_labels(n_clicks, list_of_tags):
    if n_clicks is not None and list_of_tags is not None:
        children = [html.Div('Saved')]
        return children


if __name__ == '__main__':
    app.run_server(debug=True)
