import base64
import datetime
import io
import os

import dash
from dash.dependencies import Input, Output, State, MATCH
import dash_core_components as dcc
import dash_html_components as html
import dash_table

import pandas as pd
import plotly.express as px
# import plotly.graph_objects as go

# Imports for interacting with splash-ml api
import urllib.request
import requests
import json

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

ANNOTATION_OPTIONS = [
        {'label': 'Arc', 'value': 'Arc'},
        {'label': 'Rods', 'value': 'Rods'},
        {'label': 'Peaks', 'value': 'Peaks'},
        {'label': 'Rings', 'value': 'Rings'}]

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
        'position': 'fixed',
        'top': 0,
        'left': 0,
        'bottom': 0,
        'width': '22rem',
        'padding': '2rem 1rem',
        'background-color': '#f8f9fa'}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
        'margin-left': '24rem',
        'margin-right': '2rem',
        'padding': '2rem 1rem'}

sidebar = html.Div(
        id='sidebar',
        children=[
            html.H2("1D XRD", className="display-4"),
            html.Hr(),
            html.P(
                "A simple labeler using Splash-ML and Local Files",
                className="lead"),
            html.Div(
                children=[
                    dcc.Input(
                        id='GET_uri',
                        placeholder='Pick URI',
                        type='text'),
                    dcc.Dropdown(
                        id='GET_tag',
                        options=ANNOTATION_OPTIONS,
                        placeholder='Pick Tags',
                        multi=True,
                        style={
                            'textAlign': 'center'})],
                style={
                    'lineHeight': '60px',
                    'textAlign': 'center'}),
            html.Div(
                children=html.Button(
                    id='splash_ml_data',
                    children='Query Splash-ML'),
                style={
                    'textAlign': 'center'})],

        style=SIDEBAR_STYLE)


content = html.Div([
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
            multiple=True),
        html.Div(
            children='Or Query Splash-ML',
            style={'textAlign': 'center'}),
        html.Div(id='output_data_upload')],
    style=CONTENT_STYLE)


# Setting up initial webpage layout
app.layout = html.Div([sidebar, content])


# splash-ml GET request with default parameters.  parameters might be an option
# to look into but currently splash just returns the first 10 datasets in the
# database.
def splash_GET_call(uri, tags, offset, limit):
    url = 'http://127.0.0.1:8000/api/v0/datasets?'
    if uri:
        url += ('&uri='+uri)
    if tags:
        for tag in tags:
            url += ('&tags='+tag)
    if offset:
        url += ('&page%5Boffset%5D='+str(offset))
    if limit:
        url += ('&page%5Blimit%5D='+str(limit))
    response = urllib.request.urlopen(url)
    data = json.loads(response.read())
    return data


# Takes tags applied to data along wtih the UID of the splash-ml dataset. With
# those tags and UID it PATCHs to the database with the api.
def splash_PATCH_call(tag, uid, domain):
    url = 'http://127.0.0.1:8000/api/v0/datasets/'+uid[5:]+'/tags'
    data = []
    data.append({
        'name': tag,
        'locator': 'xaxis.range: '+str(domain[0])+', '+str(domain[1])})
    return requests.patch(url, json=data).status_code


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


# Handles the creation and format of the graph component in this webpage
def get_Graph(df, name, index):
    fig = px.line(x=df.energy, y=df.itrans/df.i0,
                  labels={
                      'x': 'Energy',
                      'y': 'iTrans/i0'})
    fig.update_layout(
             xaxis=dict(
                rangeslider=dict(
                        visible=True),
                type='linear'))

    graph = dcc.Graph(
            id={'type': name, 'index': index},
            figure=fig,
            config={
                'displayModeBar': True,
                'displaylogo': False})
    return graph


# parsing splash-ml files found.  Changes download tags button to an upload
# button that applies these tags to splash-ml
def parse_splash_ml(contents, filename, uid, tags, index):

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
    if tags is None:
        tags = []
    graph = get_Graph(df, 'graph', index)

    tags_data = []
    for i in tags:
        x1 = i['locator'].split()[1][:-1]
        x2 = (i['locator'].split())[2]
        temp = dict(Tag=i['name'], Domain=str(x1+', '+x2))
        tags_data.append(temp)

    graphData = [
            html.H5(
                id={'type': 'splash_location', 'index': index},
                children='uri: '+filename),
            html.H6(
                id={'type': 'splash_uid', 'index': index},
                children='uid: '+uid),

            # Graph of csv file
            graph,
            html.Div(id={'type': 'domain', 'index': index}),
            html.Div(
                children=[
                    html.H6('Select domain to apply tag to'),
                    dcc.Dropdown(
                        id={'type': 'splash_tags', 'index': index},
                        options=ANNOTATION_OPTIONS,
                        placeholder='Select Tag'),
                    html.Button(
                        id={'type': 'upload_splash_tags', 'index': index},
                        children='Save Tag to Splash-ML'),
                    html.Div(id={'type': 'splash_response', 'index': index})],
                style={
                    'width': '30rem',
                    'padding': '3rem 3rem',
                    'float': 'left'}),
            html.Div(
                children=[
                    html.H5('Current Tags'),
                    dash_table.DataTable(
                        id={'type': 'splash_tag_table', 'index': index},
                        columns=[
                            {'name': 'Tag', 'id': 'Tag'},
                            {'name': 'Domain', 'id': 'Domain'}],
                        data=tags_data,
                        style_cell={'padding': '1rem'})],
                style={
                    'margin-left': '33rem',
                    'margin-right': '2rem',
                    'padding': '3rem 3rem'})]

    return html.Div(
            children=graphData,
            style={
                'box-shadow': '0 4px 8px 0 rgba(0,0,0,0.2)',
                'border-radius': '20px',
                'padding': '16px 16px',
                'margin': '10px'})


# Parsing uploaded files to display graphically on the website
def parse_contents(contents, filename, date, index):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    content = io.StringIO(decoded.decode('utf-8'))
    try:
        # Different if statments to hopefully handel the files types needed
        # when graphing 1D data
        if filename.endswith('.csv'):
            # The user uploaded a CSV file
            df = pd.read_csv(content)
            # Can't handle anything other than 3 columns for graphing
            if len(df.columns) != 3:
                raise
        if filename.endswith('.xdi'):
            # The user uploaded a XDI file
            df = parseXDI(content)

    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    # Only handels df data that has columns of energy, itrans, and i0
    graph = get_Graph(df, 'graph', index)

    graphData = [
            html.H5(
                id={'type': 'filename', 'index': index},
                children=filename),
            html.H6(datetime.datetime.fromtimestamp(date)),

            # Graph of csv file
            graph,
            html.Div(id={'type': 'domain', 'index': index}),
            html.Div(
                children=[
                    html.H6('Select domain to apply tag to'),
                    dcc.Dropdown(
                        id={'type': 'dropdown_tags', 'index': index},
                        options=ANNOTATION_OPTIONS,
                        placeholder='Select Tags'),
                    html.Button(
                        id={'type': 'apply_labels', 'index': index},
                        children='Save Label to Table')],
                style={
                    'width': '30rem',
                    'padding': '3rem 3rem',
                    'float': 'left'}),
            html.Div(
                children=[
                    html.H5('Current Tags'),
                    dash_table.DataTable(
                        id={'type': 'tag_table', 'index': index},
                        columns=[
                            {'name': 'Tag', 'id': 'Tag'},
                            {'name': 'Domain', 'id': 'Domain'}],
                        style_cell={'padding': '1rem'}),
                    html.Button(
                        id={'type': 'save_labels', 'index': index},
                        children='Save Table of Tags'),
                    dcc.Download(id={
                        'type': 'download_csv_tags',
                        'index': index}),
                    html.Div(id={'type': 'saved_response', 'index': index})],
                style={
                    'margin-left': '33rem',
                    'margin-right': '2rem',
                    'padding': '3rem 3rem'})]

    return html.Div(
            children=graphData,
            style={
                'box-shadow': '0 4px 8px 0 rgba(0,0,0,0.2)',
                'border-radius': '20px',
                'padding': '16px 16px',
                'margin': '10px'})


# Takes the tags and converts them to a .csv format to save locally for the
# user on the webpage
def save_local_file(rows_of_tags, file_name):
    # getting rid of .csv from file name to add _tags.csv to end
    tags_file = file_name[:-4]+'_tags.csv'
    f = open('./'+tags_file, 'w')
    for i in rows_of_tags:
        x1 = i['Domain'].split()[0][:-1]
        x2 = i['Domain'].split()[1]
        f.write(i['Tag']+','+x1+','+x2+'\n')
    f.close()

    res = dcc.send_file('./'+tags_file)
    os.remove('./'+tags_file)
    return res


@app.callback(
        Output('output_data_upload', 'children'),
        Output('upload_data', 'contents'),
        Output('splash_ml_data', 'n_clicks'),
        Input('upload_data', 'contents'),
        Input('splash_ml_data', 'n_clicks'),
        State('upload_data', 'filename'),
        State('upload_data', 'last_modified'),
        State('GET_uri', 'value'),
        State('GET_tag', 'value'),
        prevent_initial_call=True)
def update_output(
        list_of_contents,
        n_clicks,
        list_of_names,
        list_of_dates,
        uri,
        list_of_tags):
    children = []
    if list_of_contents:
        for i in range(len(list_of_contents)):
            c = list_of_contents[i]
            n = list_of_names[i]
            d = list_of_dates[i]
            children.append(parse_contents(c, n, d, i))
    elif n_clicks:
        offset = 0
        limit = 10
        file_info = splash_GET_call(uri, list_of_tags, offset, limit)
        children = []
        for i in range(len(file_info)):
            f_type = file_info[i]['type']
            if f_type == 'file':
                c = open(file_info[i]['uri'], 'r')
                n = file_info[i]['uri']
                d = file_info[i]['uid']
                t = file_info[i]['tags']
                children.append(parse_splash_ml(c, n, d, t, i))
            elif f_type == 'dbroker':
                # grab file from dbroker, atm this doesnt exist though
                print('work in progress')
            elif f_type == 'web':
                # grab file from web??? this probably exists, not sure how to
                # work this without splash-ml example
                print('work in progress')
            else:
                children.append(html.Div('Invalid file type from splash-ml'))
    return children, [], 0


@app.callback(
        Output({'type': 'tag_table', 'index': MATCH}, 'data'),
        Input({'type': 'apply_labels', 'index': MATCH}, 'n_clicks'),
        State({'type': 'tag_table', 'index': MATCH}, 'data'),
        State({'type': 'dropdown_tags', 'index': MATCH}, 'value'),
        State({'type': 'graph', 'index': MATCH}, 'figure'),
        prevent_initial_call=True)
def apply_tags_table(n_clicks, rows, tag, figure):
    if n_clicks and tag:
        x1 = figure['layout']['xaxis']['range'][0]
        x2 = figure['layout']['xaxis']['range'][1]
        temp = dict(Tag=tag, Domain=str(x1)+', '+str(x2))
        if rows:
            rows.append(temp)
        else:
            rows = [temp]
    return rows


@app.callback(
        Output({'type': 'download_csv_tags', 'index': MATCH}, 'data'),
        Output({'type': 'saved_response', 'index': MATCH}, 'children'),
        Input({'type': 'save_labels', 'index': MATCH}, 'n_clicks'),
        State({'type': 'tag_table', 'index': MATCH}, 'data'),
        State({'type': 'filename', 'index': MATCH}, 'children'),
        prevent_initial_call=True)
def save_tags_button(n_clicks, rows, file_name):
    if n_clicks and rows:
        save = save_local_file(rows, file_name)
        return save, html.Div('Downloading Tags')
    else:
        return None, html.Div('No Tags Selected')


@app.callback(
        Output({'type': 'splash_response', 'index': MATCH}, 'children'),
        Output({'type': 'splash_tag_table', 'index': MATCH}, 'data'),
        Input({'type': 'upload_splash_tags', 'index': MATCH}, 'n_clicks'),
        State({'type': 'splash_tag_table', 'index': MATCH}, 'data'),
        State({'type': 'splash_tags', 'index': MATCH}, 'value'),
        State({'type': 'splash_uid', 'index': MATCH}, 'children'),
        State({'type': 'graph', 'index': MATCH}, 'figure'),
        prevent_initial_call=True)
def upload_tags_button(n_clicks, rows, tag, uid, figure):
    if n_clicks and tag:
        code_response = splash_PATCH_call(
                tag,
                uid,
                figure['layout']['xaxis']['range'])
        # 200 for OK, 422 for validation error, 500 for server error
        if code_response == 200:
            x1 = figure['layout']['xaxis']['range'][0]
            x2 = figure['layout']['xaxis']['range'][1]
            temp = dict(Tag=tag, Domain=str(x1)+', '+str(x2))
            if rows:
                rows.append(temp)
            else:
                rows = [temp]
        return html.Div('Uploading Tags: '+str(code_response)), rows
    else:
        return html.Div('No Tags Selected'), rows


@app.callback(
        Output({'type': 'domain', 'index': MATCH}, 'children'),
        Input({'type': 'graph', 'index': MATCH}, 'relayoutData'),
        State({'type': 'graph', 'index': MATCH}, 'figure'))
def post_graph_scale(action_dict, figure):
    return 'Domain: '+str(figure['layout']['xaxis']['range'])


if __name__ == '__main__':
    app.run_server(debug=True)
