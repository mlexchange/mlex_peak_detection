import base64
import datetime
import io
import os

import dash
from dash.dependencies import Input, Output, State, MATCH
import dash_core_components as dcc
import dash_html_components as html

import pandas as pd
import plotly.express as px

# Imports for interacting with splash-ml api
import urllib.request
import requests
import json


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Setting up initial webpage layout
app.layout = html.Div([
        dcc.Upload(
            id='upload_data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')]),
            style={
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'},
            # Allow multiple files to be uploaded
            multiple=True
        ),
        html.Div(
            children='Or',
            style={'textAlign': 'center'}),
        html.Div(
            children=html.Button(
                id='splash_ml_data',
                children='Splash ML'),
            style={'textAlign': 'center'}),

        html.Div(id='output_data_upload'),
        html.Div(id='output_data_splash')])


# splash-ml GET request with default parameters.  parameters might be an option
# to look into but currently splash just returns the first 10 datasets in the
# database.
def splash_GET_call():
    url = 'http://127.0.0.1:8000/api/v0/datasets'
    response = urllib.request.urlopen(url)
    data = json.loads(response.read())
    file_info = []
    for i in data:
        file_info.append((i['uri'], i['uid'], i['type']))

    return file_info


# Takes tags applied to data along wtih the UID of the splash-ml dataset. With
# those tags and UID it POSTS to the database with the api.
def splash_POST_call(list_of_tags, uid):
    url = 'http://127.0.0.1:8000/api/v0/datasets/'+uid[5:]+'/tags'
    data = []
    for i in list_of_tags:
        data.append({'name': 'label', 'value': i})
    return requests.post(url, json=data).status_code


# Handles .xdi files and returns a dataframe of the data in it
def parseXDI(csvFile):
    # parse the csvFile as a file.  While it isn't actually a file, it's saved
    # in memory so it can be accessed like one.
    last_header_line = None
    data = []
    for line in csvFile.readlines():
        if line.startswith('#'):
            last_header_line = line
        else:
            vals = line.split()
            vals = [float(i) for i in vals]
            data.append(vals)

    return pd.DataFrame(data, columns=last_header_line.split()[1:])


# parsing splash-ml files found.  Changes download tags button to an upload
# button that applies these tags to splash-ml
def parse_splash_ml(contents, filename, uid, index):

    try:
        # Different if statments to hopefully handel the files types needed
        # when graphing 1D data
        if filename.endswith('.csv'):
            # The user uploaded a CSV file
            df = pd.read_csv(contents)
            # Can't handle anything other than 3 columns for graphing
            if len(df.columns) != 3:
                raise
        if filename.endswith('.xdi'):
            # The user uploaded a XDI file
            df = parseXDI(contents)

    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    # Only handels df data that has columns of energy, itrans, and i0
    fig = px.scatter(x=df.energy, y=df.itrans/df.i0)
    fig.update_traces(mode='lines+markers')

    graphData = [
            html.H5(
                id={'type': 'splash_location', 'index': index},
                children=filename),
            html.H6(
                id={'type': 'splash_uid', 'index': index},
                children='uid: '+uid),

            # Graph of csv file
            dcc.Graph(
                figure=fig),
            dcc.Dropdown(
                id={'type': 'splash_tags', 'index': index},
                options=[
                    {'label': 'Arc', 'value': 'Arc'},
                    {'label': 'Peaks', 'value': 'Peaks'},
                    {'label': 'Rings', 'value': 'Rings'},
                    {'label': 'Rods', 'value': 'Rods'}],
                multi=True,
                placeholder="Select Tags"),
            html.Button(
                id={'type': 'upload_splash_tags', 'index': index},
                children='Save Labels to Splash-ML'),
            html.Div(id={'type': 'splash_response', 'index': index})]

    return html.Div(graphData)


# Parsing uploaded files to display graphically on the website
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

    # Only handels df data that has columns of energy, itrans, and i0
    fig = px.scatter(x=df.energy, y=df.itrans/df.i0)
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
                id={'type': 'save_labels', 'index': index},
                children='Save Labels to Disk'),
            dcc.Download(id={
                'type': 'download_csv_tags',
                'index': index}),
            html.Div(id={'type': 'saved_response', 'index': index}),

            html.Hr(),  # horizontal line

            # For debugging, display the raw contents provided by the
            # web browser
            html.Div('Raw Content'),
            html.Pre(contents[0:200] + '...', style={
                'whiteSpace': 'pre-wrap',
                'wordBreak': 'break-all'
            })]

    return html.Div(graphData)


# Takes the tags and converts them to a .csv format to save locally for the
# user on the webpage
def save_local_file(list_of_tags, file_name):
    # getting rid of .csv from file name to add _tags.csv to end
    tags_file = file_name[:-4]+'_tags.csv'
    f = open("./"+tags_file, "w")
    for i in list_of_tags:
        f.write(i+"\n")
    f.close()

    res = dcc.send_file('./'+tags_file)
    os.remove('./'+tags_file)
    return res


@app.callback(
        Output('output_data_upload', 'children'),
        Input('upload_data', 'contents'),
        State('upload_data', 'filename'),
        State('upload_data', 'last_modified'))
def update_output_local(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = []
        for i in range(len(list_of_contents)):
            c = list_of_contents[i]
            n = list_of_names[i]
            d = list_of_dates[i]
            children.append(parse_contents(c, n, d, i))
        return children


@app.callback(
        Output('output_data_splash', 'children'),
        Input('splash_ml_data', 'n_clicks'),
        prevent_initial_call=True)
def update_output_splash(n_clicks):
    if n_clicks is not None:
        file_info = splash_GET_call()
        children = []
        for i in range(len(file_info)):
            if file_info[i][2] == 'file':
                c = open(file_info[i][0], 'r')
                n = file_info[i][0]
                d = file_info[i][1]
                children.append(parse_splash_ml(c, n, d, i))
            elif file_info[i][2] == 'dbroker':
                # grab file from dbroker, atm this doesnt exist though
                print('work in progress')
            elif file_info[i][2] == 'web':
                # grab file from web??? this probably exists, not sure how to
                # work this without splash-ml example
                print('work in progress')
        return children


@app.callback(
        Output({'type': 'download_csv_tags', 'index': MATCH}, 'data'),
        Input({'type': 'save_labels', 'index': MATCH}, 'n_clicks'),
        State({'type': 'dropdown_tags', 'index': MATCH}, 'value'),
        State({'type': 'filename', 'index': MATCH}, 'children'),
        prevent_initial_call=True)
def save_tags_button(n_clicks, list_of_tags, file_name):
    if n_clicks and list_of_tags:
        return save_local_file(list_of_tags, file_name)


@app.callback(
        Output({'type': 'saved_response', 'index': MATCH}, 'children'),
        Input({'type': 'save_labels', 'index': MATCH}, 'n_clicks'),
        State({'type': 'dropdown_tags', 'index': MATCH}, 'value'),
        prevent_initial_call=True)
def save_no_tags(n_clicks, list_of_tags):
    if n_clicks and list_of_tags:
        return html.Div('Downloading Tags')
    else:
        return html.Div('No Tags Selected')


@app.callback(
        Output({'type': 'splash_response', 'index': MATCH}, 'children'),
        Input({'type': 'upload_splash_tags', 'index': MATCH}, 'n_clicks'),
        State({'type': 'splash_tags', 'index': MATCH}, 'value'),
        State({'type': 'splash_uid', 'index': MATCH}, 'children'),
        prevent_initial_call=True)
def upload_tags_button(n_clicks, list_of_tags, uid):
    if n_clicks and list_of_tags:
        code_response = splash_POST_call(list_of_tags, uid)
        # 200 for OK, 422 for validation error, 500 for server error
        return html.Div('Uploading Tags: '+str(code_response))
    else:
        return html.Div('No Tags Selected')


if __name__ == '__main__':
    app.run_server(debug=True)
