import base64
import datetime
import io
import os

import dash
from dash.dependencies import Input, Output, State, MATCH
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


def parseXDI(csvFile):
    # parse the csvFile as a file.  While it isn't actually a file, it's saved
    # in memory so it can be accessed like one. The goal here is to find the
    # '----' instance before the data so we can grab it for use in a graph

    return None


def parse_contents(contents, filename, date, index):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        # Different if statments to hopefully handel the files types needed
        # when graphing 1D data
        if filename.endswith('.csv'):
            # The user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
            # Can't handle anything other than 3 columns for graphing
            if len(df.columns) != 3:
                raise
        if filename.endswith('.xdi'):
            # The user uploaded a XDI file
            df = parseXDI(io.StringIO(decoded.decode('utf-8')))

    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    fig = px.scatter(x=df.energy, y=df.mutrans/df.i0)
    fig.update_traces(mode='lines+markers')

    graphData = [
            html.H5(
                id={'type': 'filename', 'index': index},
                children=filename),
            html.H6(datetime.datetime.fromtimestamp(date)),

            # Graph of csv file
            dcc.Graph(
                figure=fig),
            dcc.Dropdown(
                id={'type': 'dropdown_tags', 'index': index},
                options=[
                    {'label': 'Arc', 'value': 'Arc'},
                    {'label': 'Peaks', 'value': 'Peaks'},
                    {'label': 'Rings', 'value': 'Rings'},
                    {'label': 'Rods', 'value': 'Rods'}],
                multi=True,
                placeholder="Select Tags"),
            html.Button(
                id={'type': 'save-labels', 'index': index},
                children='Save Labels to Disk'),
            dcc.Download(id={
                'type': 'download-csv-tags',
                'index': index}),
            html.Div(id={'type': 'saved-response', 'index': index}),

            html.Hr(),  # horizontal line

            # For debugging, display the raw contents provided by the
            # web browser
            html.Div('Raw Content'),
            html.Pre(contents[0:200] + '...', style={
                'whiteSpace': 'pre-wrap',
                'wordBreak': 'break-all'
            })]

    return html.Div(graphData)


@app.callback(
        Output('output-data-upload', 'children'),
        Input('upload-data', 'contents'),
        State('upload-data', 'filename'),
        State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = []
        for i in range(len(list_of_contents)):
            c = list_of_contents[i]
            n = list_of_names[i]
            d = list_of_dates[i]
            children.append(parse_contents(c, n, d, i))
        return children


@app.callback(
        Output({'type': 'download-csv-tags', 'index': MATCH}, 'data'),
        Input({'type': 'save-labels', 'index': MATCH}, 'n_clicks'),
        State({'type': 'dropdown_tags', 'index': MATCH}, 'value'),
        State({'type': 'filename', 'index': MATCH}, 'children'),
        prevent_initial_call=True)
def save_labels(n_clicks, list_of_tags, file_name):
    if n_clicks is not None:
        if list_of_tags is not None:
            # getting rid of .csv from file name to add _tags.csv to end
            tags_file = file_name[:-4]+'_tags.csv'
            f = open("./"+tags_file, "w")
            for i in list_of_tags:
                f.write(i+"\n")
            f.close()

            res = dcc.send_file('./'+tags_file)
            os.remove('./'+tags_file)
            return res


if __name__ == '__main__':
    app.run_server(debug=True)
