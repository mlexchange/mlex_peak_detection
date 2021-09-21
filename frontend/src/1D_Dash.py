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
import plotly.graph_objects as go
import numpy as np

# Imports for interacting with splash-ml api
import urllib.request
import requests
import json

from packages.helpers import get_peaks, peak_helper

# Imported code by Ronald Pandolfi
from packages.targeted_callbacks import targeted_callback


class Tag():
    def __init__(self, tag_name, peak_x, peak_y, fwhm, uid='TBD'):
        self.Tag = tag_name
        self.Peak = str(peak_x) + ', ' + str(peak_y)
        self.FWHM = str(fwhm)
        self.tag_uid = uid


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Figure of graph when there are unfit and fit curves added on.
global stash_figure
stash_figure = None

global DATA_DIR
DATA_DIR = os.environ['DATA_DIR']

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
        'position': 'fixed',
        'top': 0,
        'left': 0,
        'bottom': 0,
        'padding': '2rem 1rem',
        'background-color': '#f8f9fa',
        'width': '15%'}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
        'margin-left': '17%',
        'padding': '2rem 1rem'}

#
TABLE_STYLE = [
{   'if': {'column_id': ''},
    'width': '2%'},
    {'if': {'column_id': 'Tag'},
    'width': '10%'},
    {'if': {'column_id': 'Peak'},
    'width': '35%'},
    {'if': {'column_id': 'FWHM'},
     'width': '35%'},
    {'if': {'column_id': 'COLOR'},
     'width': '10%'},
    {'if': {'column_id': 'tag_uid'},
     'width': '8%'},
]

# Sidebar content, this includes the titles with the splash-ml entry and query
# button
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
                    dcc.Input(
                        id='GET_tag',
                        placeholder='Pick Tag')],
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

# Content section to the right of the sidebar.  This includes the upload bar
# and graphs to be tagged once loaded into app.
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


# splash-ml GET request with uri and tag paramaters.  The offset and limit
# values are hard coded at the moment as splash-ml integration and use case
# isnt fully explored
def splash_GET_call(uri, tags, offset, limit):
    url = 'http://splash:8000/api/v0/datasets?'
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
def splash_PATCH_call(uid, tags2add, tags2remove):
    url = 'http://splash:8000/api/v0/datasets/' + uid + '/tags'
    data = {'add_tags': tags2add, 'remove_tags': list(tags2remove)}
    print(data)
    return requests.patch(url, json=data).status_code


# Handles .xdi files and returns a dataframe of the data in it
def parseXDI(xdiFile):
    # parse the xdiFile as a file.  While it isn't actually a file, it's saved
    # in memory so it can be accessed like one.
    last_header_line = None
    data = []
    for line in xdiFile.readlines():
        if line.startswith('#'):
            last_header_line = line
        else:
            vals = line.split()
            vals = [float(i) for i in vals]
            data.append(vals)

    return pd.DataFrame(data, columns=last_header_line.split()[1:])


# Handles the creation and format of the figure graph component in this webpage
def get_fig(x, y):
    if len(y) > 0:
        fig = go.Figure(
            go.Scatter(x=x, y=y, name='XRD Data'))
    else:
        fig = px.line(x)
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(
                visible=True),
                type='linear'),
        yaxis=dict(
            autorange=False,
            range=[np.amin(x)-30, np.amax(y)+30],
            fixedrange=False),
        margin=dict(l=20, r=20, t=30, b=20),
        height=300
    )

    return fig


# Applying tags to graph along with baselines or fit curves from the peak
# fitting calls
def update_annotation_helper(rows, x, y, unfit_list=None, fit_list=None,
                             residual=None, base_list=None):
    figure = get_fig(x, y)
    if rows:
        for i in rows:
            pos_x = i['Peak'].split()[0][:-1]
            name = i['Tag']
            figure.add_vline(
                    x=float(pos_x),
                    line_width=1,
                    line_color='purple',
                    annotation_text=name)
    if base_list:
        figure.add_trace(
                go.Scatter(
                    x=base_list[0],
                    y=base_list[1],
                    mode='lines',
                    name='baseline'))
        if unfit_list is not None:
            figure.add_trace(
                    go.Scatter(
                        x=unfit_list[0],
                        y=(np.array(unfit_list[1]) + np.array(base_list[1])),
                        mode='lines',
                        name='unfit'))
        if fit_list is not None:
            figure.add_trace(
                    go.Scatter(
                        x=fit_list[0],
                        y=(np.array(fit_list[1]) + np.array(base_list[1])),
                        mode='lines',
                        name='fit'))
        if residual is not None:
            figure.add_trace(
                    go.Scatter(
                        x=residual[0],
                        y=(np.array(residual[1]) + np.array(base_list[1])),
                        mode='lines',
                        name='residual'))
    else:
        if unfit_list is not None:
            figure.add_trace(
                    go.Scatter(
                        x=unfit_list[0],
                        y=unfit_list[1],
                        mode='lines',
                        name='unfit'))
        if fit_list is not None:
            figure.add_trace(
                    go.Scatter(
                        x=fit_list[0],
                        y=fit_list[1],
                        mode='lines',
                        name='fit'))
        if residual is not None:
            figure.add_trace(
                    go.Scatter(
                        x=residual[0],
                        y=residual[1],
                        mode='lines',
                        name='residual'))

    return figure


# parsing splash-ml files found and upload contents to content section of
# webpage
def parse_splash_ml(contents, filename, uid, tags, index):
    try:
        print(filename)
        # Different if statements to hopefully handle the files types needed
        # when graphing 1D data
        if filename.endswith('.csv'):
            # The user uploaded a CSV file
            df = pd.read_csv(contents)
            # Can't handle anything other than 3 columns for graphing
            if len(df.columns) != 2:
                raise
        if filename.endswith('.xdi'):
            # The user uploaded a XDI file
            df = parseXDI(contents)
        if filename.endswith('.npy'):
            # The user uploaded a numpy file
            npyArr = np.load(filename)
            df = pd.DataFrame({'Column1': npyArr})

    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    # Makes sure tags doesnt break with None type
    if tags is None:
        tags = []

    # Building the splash-ml table from tags already in the database
    tags_data = []
    for i in tags:
        arr = i['locator'].split()

        if len(arr) == 3:
            x = arr[0][:-1]
            y = arr[1][:-1]
            fwhm = arr[2]
            temp = Tag(i['name'], x, y, fwhm, i['uid'])
            tags_data.append(temp.__dict__)
        else:
            temp = Tag(i['name'])
            print('MISSING TAG VALUES')
            tags_data.append(temp.__dict__)

    # split down data to work with new get_fig function
    data = pd.DataFrame.to_numpy(df)
    x = data[:, 0]
    if len(df.columns) == 2:
        y = data[:, 1]
    else:
        y = []
    # graphData = generate_graph(x, y, index, filename, tags_data) # MERGING EVERYTHING
    # building graph object outside of get_fig() as get_fig() is used in other
    # locations
    graph = dcc.Graph(
            id={'type': 'splash_graph', 'index': index},
            figure=update_annotation_helper(tags_data, x, y),
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': [
                    'pan2d',
                    'resetScale2d',
                    'toggleSpikelines',
                    'hoverCompareCartesian',
                    'zoomInGeo',
                    'zoomOutGeo']})
 #   color = random.sample(range(0, 0xFFFFFF),tags_data.shape[0])
 #   color = list(map(str,color))
 #   list_colors = ['#'+shade for shade in color]
 #   tags_data['COLOR'] = list_colors
    graphData = [
            html.H5(
                id={'type': 'splash_location', 'index': index},
                children='uri: '+filename),
             html.H6(
                id={'type': 'splash_uid', 'index': index},
                children='uid: '+uid),
            # Graph of csv file
            graph,
            # Auto resize Y-Axis
            dcc.Checklist(
                id={'type': 'splash_resize', 'index': index},
                options=[
                    {'label': 'Autoscale Y-Axis',
                        'value': 'Autoscale Y-Axis'}],
                value=['Autoscale Y-Axis'],
                style={
                    'padding-left': '3rem',
                    'width': 'fit-content'}),
            dcc.Checklist(
                id={'type': 'baseline', 'index': index},
                options=[
                    {'label': 'Apply Baseline to Peak Fitting',
                        'value': 'Apply Baseline to Data'}],
                style={
                    'padding-left': '3rem',
                    'width': 'fit-content'}),
            html.H6(children=['Only select peak number if you tag window'],
                    style={'padding-left': '3rem'}),
            html.Div(
                children=[
                    html.H6('Select peaks to find'),
                    html.Div(id={'type': 'splash_domain', 'index': index}),
                    dcc.Input(
                        id={'type': 'splash_peaks', 'index': index},
                        type='number',
                        placeholder='Number of Peaks'),
                    dcc.Input(
                        id={'type': 'splash_tags', 'index': index},
                        type='text',
                        min=0,
                        placeholder='Tag Name'),
                    dcc.Dropdown(
                        id={'type': 'splash_shape', 'index': index},
                        options=[
                            {'label': 'Gaussian', 'value': 'Gaussian'},
                            {'label': 'Voigt', 'value': 'Voigt'}
                        ],
                        placeholder='Peak Shape'),
                    html.Button(
                        id={'type': 'add_splash_tags', 'index': index},
                        children='Tag Window',
                        style={
                            'padding-bottom': '3rem'}),
                    html.Button(
                        id={'type': 'block_splash_tags', 'index': index},
                        children='Tag w/ Blocks',
                        style={
                            'padding-bottom': '3rem'}),
                    html.Div(id={'type': 'splash_response', 'index': index})],
                style={
                    'width': '20%',
                    #'width': '30rem',
                    'padding': '3rem 3rem 3rem 3rem',
                    'float': 'left'}),
            html.Div(
                children=[
                    html.H5('Current Tags'),
                    dash_table.DataTable(
                        id={'type': 'splash_tag_table', 'index': index},
                        columns=[
                            {'name': 'Tag', 'id': 'Tag'},
                            {'name': 'Peak', 'id': 'Peak'},
                            {'name': 'FWHM', 'id': 'FWHM'},
                            {'name': 'COLOR', 'id': 'COLOR'},
                            {'name': 'UID', 'id': 'tag_uid'} #, 'hideable': True}
                        ],
                        data=tags_data,
                        style_cell_conditional=TABLE_STYLE,
#                        style_data_conditional=[{'if': {'row_index': i, 'column_id': 'COLOR'}, 'background-color': tags_data['COLOR'][i], 'color': tags_data['COLOR'][i]} for i in range(tags_data.shape[0])],
                        style_cell={'padding': '1rem',
                                    'textOverflow': 'ellipsis',
                                    'overflow': 'hidden'},
                                    #'maxWidth': 0},
                        style_table={'overflowX': 'auto'},
                        hidden_columns=['tag_uid'],
                        row_deletable=True),
                    html.Button(
                        id={'type': 'save_splash', 'index': index},
                        children='Save Table to Splash',
                        style={
                            'padding-bottom': '3rem'})],
                style={
                    'margin-left': '27%',
                    #'margin-left': '33rem',
                    'width': '70%',
                    #'margin-right': '2rem',
                    'padding': '3rem 3rem 10rem 3rem'})]
    return html.Div(
            children=graphData,
            style={
                'box-shadow': '0 4px 8px 0 rgba(0,0,0,0.2)',
                'border-radius': '20px',
                'padding': '30px 30px',
                'margin': '10px'})


# Parsing uploaded files to display in content section of the website
def parse_contents(contents, filename, date, index):
    content_type, content_string = contents.split(',')

    npyArr = 'Error'
    decoded = base64.b64decode(content_string)
    try:
        # Different if statments to hopefully handel the files types needed
        # when graphing 1D data
        if filename.endswith('.csv'):
            content = io.StringIO(decoded.decode('utf-8'))
            # The user uploaded a CSV file
            df = pd.read_csv(content)
            # Can't handle anything other than 3 columns for graphing
            if len(df.columns) != 2:
                raise
        if filename.endswith('.xdi'):
            content = io.StringIO(decoded.decode('utf-8'))
            # The user uploaded a XDI file
            df = parseXDI(content)
        if filename.endswith('.npy'):
            # The user uploaded a numpy file
            content = io.BytesIO(decoded)
            npyArr = np.load(content)
            df = pd.DataFrame({'Column1': npyArr})

    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file. '+str(npyArr)
        ])

    # split down data to work with new get_fig() function
    data = pd.DataFrame.to_numpy(df)
    x = data[:, 0]
    if len(df.columns) == 2:
        y = data[:, 1]
    else:
        y = []
    # Graph build outside of get_fig() to accomodate for other calls to this
    # function
    graphData = generate_graph(x, y, index, filename, [], date)

    return html.Div(
            children=graphData,
            style={
                'box-shadow': '0 4px 8px 0 rgba(0,0,0,0.2)',
                'border-radius': '20px',
                'padding': '30px 30px',
                'margin': '10px'})


def generate_graph(x, y, index, filename, tags_data, date=None):
    if len(tags_data) == 0:
        fig = get_fig(x, y)
    else:
        fig = update_annotation_helper(tags_data, x, y)
    graph = dcc.Graph(
            id={'type': 'graph', 'index': index},
            figure=fig,
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': [
                    'pan2d',
                    'resetScale2d',
                    'toggleSpikelines',
                    'hoverCompareCartesian',
                    'zoomInGeo',
                    'zoomOutGeo']})

    if date is None:
        date = 1

    graphData = [
        html.H5(
            id={'type': 'filename', 'index': index},
            children=filename),
        html.H6(datetime.datetime.fromtimestamp(date)),
        # Graph of csv file
        graph,
        # Auto resize the Y-Axis
        dcc.Checklist(
            id={'type': 'resize', 'index': index},
            options=[
                {'label': 'Autoscale Y-Axis',
                 'value': 'Autoscale Y-Axis'}],
            value=['Autoscale Y-Axis'],
            style={
                'padding-left': '3rem',
                'width': 'fit-content'}),
        dcc.Checklist(
            id={'type': 'baseline', 'index': index},
            options=[
                {'label': 'Apply Baseline to Peak Fitting',
                 'value': 'Apply Baseline to Data'}],
            style={
                'padding-left': '3rem',
                'width': 'fit-content'}),
        html.H6(children=['Only select peak number if you tag window'],
                style={'padding-left': '3rem'}),
        html.Div(
            children=[
                html.H6('Select peaks to find'),
                html.Div(id={'type': 'domain', 'index': index}),
                dcc.Input(
                    id={'type': 'input_peaks', 'index': index},
                    type='number',
                    placeholder='Number of Peaks'),
                dcc.Input(
                    id={'type': 'input_tags', 'index': index},
                    type='text',
                    min=0,
                    placeholder='Tag Name'),
                dcc.Dropdown(
                    id={'type': 'input_shape', 'index': index},
                    options=[
                        {'label': 'Gaussian', 'value': 'Gaussian'},
                        {'label': 'Voigt', 'value': 'Voigt'}
                    ],
                    placeholder='Peak Shape'),
                html.Button(
                    id={'type': 'apply_labels', 'index': index},
                    children='Tag Window',
                    style={
                        'padding-bottom': '3rem'}),
                html.Button(
                    id={'type': 'block_tag', 'index': index},
                    children='Tag w/ Blocks',
                    style={'padding-left': '3rem'})],
            style={
                'width': '20%',
                'padding': '3rem 3rem 3rem 3rem',
                'float': 'left'}),
        html.Div(
            children=[
                html.H5('Current Tags'),
                dash_table.DataTable(
                    id={'type': 'tag_table', 'index': index},
                    columns=[
                        {'name': 'Tag', 'id': 'Tag'},
                        {'name': 'Peak', 'id': 'Peak'},
                        {'name': 'FWHM', 'id': 'FWHM'},
                        {'name': 'UID', 'id': 'tag_uid'}
                    ],
                    style_cell={'padding': '1rem',
                                'textOverflow': 'ellipsis',
                                'overflow': 'hidden'},
                    # 'maxWidth': 0},
                    data=tags_data,
                    style_table={'overflowX': 'auto'},
                    hidden_columns=['tag_uid'],
                    row_deletable=True),
                html.Button(
                    id={'type': 'save_labels', 'index': index},
                    children='Save Table of Tags'),
                dcc.Download(id={
                    'type': 'download_csv_tags',
                    'index': index}),
                html.Div(id={'type': 'saved_response', 'index': index})],
            style={
                'margin-left': '27%',
                'width': '70%',
                'padding': '3rem 3rem 10rem 3rem'})
    ]
    return graphData

# Takes the tags and converts them to a .csv format to save locally for the
# user on the webpage
def save_local_file(rows_of_tags, file_name):
    # getting rid of .csv from file name to add _tags.csv to end
    tags_file = file_name[:-4]+'_tags.csv'
    f = open('/app/tmp/'+tags_file, 'w')
    for i in rows_of_tags:
        x1 = i['Peak'].split()[0][:-1]
        x2 = i['Peak'].split()[1]
        f.write(i['Tag']+','+x1+','+x2+','+i['FWHM']+'\n')
    f.close()

    res = dcc.send_file('/app/tmp/'+tags_file)
    os.remove('/app/tmp/'+tags_file)
    return res


# A callback function that automatically scales the range of the interactive
# graph.  This allows users to zoom in on the x scale and get a y scale that
# allows for a better view of the data
def zoom(change):
    input_states = dash.callback_context.states
    state_iter = iter(input_states.values())
    check = next(state_iter)
    figure = next(state_iter)
    if check:
        y_range = figure['layout']['yaxis']['range']
        x_range = figure['layout']['xaxis']['range']
        x_data = figure['data'][0]['x']
        y_data = figure['data'][0]['y']
        start = 0
        end = len(x_data) - 1
        for i in range(len(x_data)):
            if x_data[i] >= x_range[0] and start == 0:
                start = i
            if x_data[i] >= x_range[1]:
                end = i+1
                break
        in_view = y_data[start:end]
        in_view = np.array(in_view)
        y_range = [np.amin(in_view)-30, np.amax(in_view)+30]
        figure['layout']['yaxis']['range'] = y_range
    return figure


# Targets the upload part of website
targeted_callback(
        zoom,
        Input({'type': 'graph', 'index': MATCH}, 'relayoutData'),
        Output({'type': 'graph', 'index': MATCH}, 'figure'),
        State({'type': 'resize', 'index': MATCH}, 'value'),
        State({'type': 'graph', 'index': MATCH}, 'figure'),
        app=app)


# Targets the upload part of website
targeted_callback(
        zoom,
        Input({'type': 'resize', 'index': MATCH}, 'value'),
        Output({'type': 'graph', 'index': MATCH}, 'figure'),
        State({'type': 'resize', 'index': MATCH}, 'value'),
        State({'type': 'graph', 'index': MATCH}, 'figure'),
        app=app)


# Targets the splash-ml part of website
targeted_callback(
        zoom,
        Input({'type': 'splash_graph', 'index': MATCH}, 'relayoutData'),
        Output({'type': 'splash_graph', 'index': MATCH}, 'figure'),
        State({'type': 'splash_resize', 'index': MATCH}, 'value'),
        State({'type': 'splash_graph', 'index': MATCH}, 'figure'),
        app=app)


# Targets the splash-ml part of website
targeted_callback(
        zoom,
        Input({'type': 'splash_resize', 'index': MATCH}, 'value'),
        Output({'type': 'splash_graph', 'index': MATCH}, 'figure'),
        State({'type': 'splash_resize', 'index': MATCH}, 'value'),
        State({'type': 'splash_graph', 'index': MATCH}, 'figure'),
        app=app)


# Combined the upload and splash-ml data graphs into one callback to simplify
# the whole process (could be split into targeted callbacks for organization)
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
    # If local upload needs to populate the page
    if list_of_contents:
        for i in range(len(list_of_contents)):
            c = list_of_contents[i]
            n = list_of_names[i]
            d = list_of_dates[i]
            children.append(parse_contents(c, n, d, i))
    # If splash-ml needs to populate the page
    elif n_clicks:
        offset = 0
        limit = 10
        file_info = splash_GET_call(uri, list_of_tags, offset, limit)
        children = []
        for i in range(len(file_info)):
            f_type = file_info[i]['type']
            if f_type == 'file':
                try:
                    c = open(file_info[i]['uri'], 'r')
                except Exception as e:
                    print(e)
                    children.append(
                            html.Div('File path not found from splash-ml'))
                    continue
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
    if len(children) == 0:
        children.append(html.Div('No data found to graph'))
    return children, [], 0


# Tag callback for when the graph is from uploaded data.  Updates the tag table
# and saves it in current session. Handles window tagging
def single_tags_table(n_clicks):
    input_states = dash.callback_context.states
    state_iter = iter(input_states.values())
    baseline = next(state_iter)
    rows = next(state_iter)
    tag = next(state_iter)
    num_peaks = next(state_iter)
    peak_shape = next(state_iter)
    figure = next(state_iter)
    if n_clicks and tag:
        x1 = figure['layout']['xaxis']['range'][0]
        x2 = figure['layout']['xaxis']['range'][1]
        x_data = figure['data'][0]['x']
        y_data = figure['data'][0]['y']
        start, end = 0, len(x_data)
        for i in range(len(x_data)):
            if x_data[i] >= x1 and start == 0:
                start = i
            if x_data[i] >= x2:
                end = i+1
                break

        if baseline is None or len(baseline) == 0:
            peak_info, unfit_list, fit_list, residual, base_list = get_peaks(
                    x_data[start:end],
                    y_data[start:end],
                    num_peaks,
                    peak_shape)
        else:
            peak_info, unfit_list, fit_list, residual, base_list = get_peaks(
                    x_data[start:end],
                    y_data[start:end],
                    num_peaks,
                    peak_shape,
                    baseline=True)

        for i in peak_info:
            index = i['index']+start
            x, y = x_data[index], y_data[index]
            fwhm = i['FWHM']
            if i['flag'] == 1:
                temp = Tag('(F)'+tag, x, y, fwhm)
                temp = temp.__dict__
            else:
                temp = Tag(tag, x, y, fwhm)
                temp = temp.__dict__
            if rows:
                rows.append(temp)
            else:
                rows = [temp]

        x_data = figure['data'][0]['x']
        y_data = figure['data'][0]['y']

        figure = update_annotation_helper(
                rows,
                x_data,
                y_data,
                unfit_list,
                fit_list,
                residual,
                base_list)

        global stash_figure
        stash_figure = figure

    return rows


targeted_callback(
        single_tags_table,
        Input({'type': 'apply_labels', 'index': MATCH}, 'n_clicks'),
        Output({'type': 'tag_table', 'index': MATCH}, 'data'),
        State({'type': 'baseline', 'index': MATCH}, 'value'),
        State({'type': 'tag_table', 'index': MATCH}, 'data'),
        State({'type': 'input_tags', 'index': MATCH}, 'value'),
        State({'type': 'input_peaks', 'index': MATCH}, 'value'),
        State({'type': 'input_shape', 'index': MATCH}, 'value'),
        State({'type': 'graph', 'index': MATCH}, 'figure'),
        app=app)


# Tag callback for when the graph is from uploaded data.  Updates the tag table
# and saves it in current session.  Handles block tagging
def multi_tags_table(n_clicks):
    input_states = dash.callback_context.states
    state_iter = iter(input_states.values())
    baseline = next(state_iter)
    rows = next(state_iter)
    tag = next(state_iter)
    # I try deleting this and the state that goes with it, but it breaks the
    # function each time... I dont know why
    num_peaks = next(state_iter)
    peak_shape = next(state_iter)
    figure = next(state_iter)
    num_peaks = 1       # for block detection, the number of peaks is set to 1
    if n_clicks and tag:
        x1 = figure['layout']['xaxis']['range'][0]
        x2 = figure['layout']['xaxis']['range'][1]
        x_data = figure['data'][0]['x']
        y_data = figure['data'][0]['y']
        start, end = 0, len(x_data)
        for i in range(len(x_data)):
            if x_data[i] >= x1 and start == 0:
                start = i
            if x_data[i] >= x2:
                end = i+1
                break

        if baseline is None or len(baseline) == 0:
            peak_info, unfit_list, fit_list, residual, base_list = get_peaks(
                    x_data[start:end],
                    y_data[start:end],
                    num_peaks,
                    peak_shape,
                    block=True)
        else:
            peak_info, unfit_list, fit_list, residual, base_list = get_peaks(
                    x_data[start:end],
                    y_data[start:end],
                    num_peaks,
                    peak_shape,
                    baseline=True,
                    block=True)

        for i in peak_info:
            index = i['index']+start
            x, y = x_data[index], y_data[index]
            fwhm = i['FWHM']
            if i['flag'] == 1:
                temp = Tag('(F)'+tag, x, y, fwhm)
            else:
                temp = Tag(tag, x, y, fwhm)
            if rows:
                rows.append(temp.__dict__)
            else:
                rows = [temp.__dict__]

        x_data = figure['data'][0]['x']
        y_data = figure['data'][0]['y']

        figure = update_annotation_helper(
                rows,
                x_data,
                y_data,
                unfit_list,
                fit_list,
                residual,
                base_list)

        global stash_figure
        stash_figure = figure
    num_peaks = num_peaks

    return rows


targeted_callback(
        multi_tags_table,
        Input({'type': 'block_tag', 'index': MATCH}, 'n_clicks'),
        Output({'type': 'tag_table', 'index': MATCH}, 'data'),
        State({'type': 'baseline', 'index': MATCH}, 'value'),
        State({'type': 'tag_table', 'index': MATCH}, 'data'),
        State({'type': 'input_tags', 'index': MATCH}, 'value'),
        # State({'type': 'input_peaks', 'index': MATCH}, 'value'),
        State({'type': 'input_shape', 'index': MATCH}, 'value'),
        State({'type': 'graph', 'index': MATCH}, 'figure'),
        app=app)


# Tag table callback for when the graph is from uploaded data.  Downloads the
# tag table data into a csv file with format of tag,x1,x2 where x1 and x2 are
# the domain of tag
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


targeted_callback(
        single_tags_table,
        Input({'type': 'add_splash_tags', 'index': MATCH}, 'n_clicks'),
        Output({'type': 'splash_tag_table', 'index': MATCH}, 'data'),
        State({'type': 'baseline', 'index': MATCH}, 'value'),
        State({'type': 'splash_tag_table', 'index': MATCH}, 'data'),
        State({'type': 'splash_tags', 'index': MATCH}, 'value'),
        State({'type': 'splash_peaks', 'index': MATCH}, 'value'),
        State({'type': 'splash_shape', 'index': MATCH}, 'value'),
        State({'type': 'splash_graph', 'index': MATCH}, 'figure'),
        app=app)


targeted_callback(
        multi_tags_table,
        Input({'type': 'block_splash_tags', 'index': MATCH}, 'n_clicks'),
        Output({'type': 'splash_tag_table', 'index': MATCH}, 'data'),
        State({'type': 'splash_tag_table', 'index': MATCH}, 'data'),
        State({'type': 'splash_tags', 'index': MATCH}, 'value'),
        # State({'type': 'splash_peaks', 'index': MATCH}, 'value'),
        State({'type': 'splash_shape', 'index': MATCH}, 'value'),
        State({'type': 'baseline', 'index': MATCH}, 'value'),
        State({'type': 'splash_graph', 'index': MATCH}, 'figure'),
        app=app)


# Handles upload of table data to splash-ml.  It first does a GET call, and
# then compares to figure out what data needs to be uploaded.  Currently no
# way to really compare tag to tag, so I just compare size of tag list
def update_splash_data(n_clicks):
    input_states = dash.callback_context.states
    state_iter = iter(input_states.values())
    rows = next(state_iter)
    uri = next(state_iter)
    uid = next(state_iter)
    uri = uri[5:]
    uid = uid[5:]
    splash_data = splash_GET_call(uri, None, 0, 1)
    splash_tags = splash_data[0]['tags']
    splash_tags_uid = [tag['uid'] for tag in splash_tags]
    current_tag_uid = [row['tag_uid'] for row in rows]
    tags2remove = np.setdiff1d(splash_tags_uid, current_tag_uid)
    row_idx_add = [i for i, e in enumerate(current_tag_uid) if e == 'TBD']
    tags2add = []
    for idx in row_idx_add:
        i = rows[idx]
        tags2add.append({'name': i['Tag'],
                         'locator': i['Peak']+', '+str(i['FWHM'])
                         })
    response = splash_PATCH_call(uid, tags2add, tags2remove)
    if response != 200:
        return html.Div('Response: '+str(response))
    return html.Div('Response: 200')


targeted_callback(
        update_splash_data,
        Input({'type': 'save_splash', 'index': MATCH}, 'n_clicks'),
        Output({'type': 'splash_response', 'index': MATCH}, 'children'),
        State({'type': 'splash_tag_table', 'index': MATCH}, 'data'),
        State({'type': 'splash_location', 'index': MATCH}, 'children'),
        State({'type': 'splash_uid', 'index': MATCH}, 'children'),
        app=app)


# Displays the current domain of graph for local upload.
@app.callback(
        Output({'type': 'domain', 'index': MATCH}, 'children'),
        Input({'type': 'graph', 'index': MATCH}, 'relayoutData'),
        State({'type': 'graph', 'index': MATCH}, 'clickData'),
        State({'type': 'graph', 'index': MATCH}, 'figure'))
def post_graph_scale(action_dict, data, figure):
    domain = figure['layout']['xaxis']['range']
    x1 = domain[0]
    x2 = domain[1]
    return html.Div(children=[
        'Domain: ['+str(round(x1, 2))+', '+str(round(x2, 2))+']'])


# Displays the current domain of graph for splash-ml upload.
@app.callback(
        Output({'type': 'splash_domain', 'index': MATCH}, 'children'),
        Input({'type': 'splash_graph', 'index': MATCH}, 'relayoutData'),
        State({'type': 'splash_graph', 'index': MATCH}, 'clickData'),
        State({'type': 'splash_graph', 'index': MATCH}, 'figure'))
def splash_graph_scale(action_dict, data, figure):
    domain = figure['layout']['xaxis']['range']
    x1 = domain[0]
    x2 = domain[1]
    return html.Div(children=[
        'Domain: ['+str(round(x1, 2))+', '+str(round(x2, 2))+']'])


# populates the graph with tags and the peak locations for splash upload
# Optimization Issue with deleting lots of tags... Code runs, but the graph
# updates really slowly when the whole graph is updated
def update_graph_annotation(rows):
    global stash_figure
    if stash_figure:
        fig = stash_figure
        stash_figure = None
        return fig
    input_states = dash.callback_context.states
    for i in iter(input_states):
        if str(i).endswith('.figure'):
            figure = dash.callback_context.states[i]
    x_data = figure['data'][0]['x']
    y_data = figure['data'][0]['y']
    return update_annotation_helper(rows, x_data, y_data)


targeted_callback(
        update_graph_annotation,
        Input({'type': 'tag_table', 'index': MATCH}, 'data'),
        Output({'type': 'graph', 'index': MATCH}, 'figure'),
        State({'type': 'graph', 'index': MATCH}, 'figure'),
        app=app)


# populates the graph with tags and the peak locations for splash upload
def update_splash_annotation(rows):
    input_states = dash.callback_context.states
    global stash_figure
    if stash_figure:
        fig = stash_figure
        stash_figure = None
        return fig
    for i in iter(input_states):
        if str(i).endswith('.figure'):
            figure = dash.callback_context.states[i]
    x_data = figure['data'][0]['x']
    y_data = figure['data'][0]['y']
    return update_annotation_helper(rows, x_data, y_data)


targeted_callback(
    update_splash_annotation,
    Input({'type': 'splash_tag_table', 'index': MATCH}, 'data'),
    Output({'type': 'splash_graph', 'index': MATCH}, 'figure'),
    State({'type': 'splash_graph', 'index': MATCH}, 'figure'),
    app=app)


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')
