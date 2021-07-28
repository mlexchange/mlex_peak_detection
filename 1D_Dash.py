import base64
import datetime
import io
import os
import math

import dash
from dash.dependencies import Input, Output, State, MATCH
import dash_core_components as dcc
import dash_html_components as html
import dash_table

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy import signal
from astropy.modeling import models, fitting

# Imports for interacting with splash-ml api
import urllib.request
import requests
import json

# Imported code by Ronald Pandolfi
from packages.targeted_callbacks import targeted_callback
# Imported code by Robert Tang-Kong
from packages.hitp import bayesian_block_finder

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Figure of graph when there are unfit and fit curves added on.
global stash_figure
stash_figure = None

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
def splash_PATCH_call(tag, uid, x, y, fwhm):
    url = 'http://127.0.0.1:8000/api/v0/datasets/'+uid+'/tags'
    data = []
    data.append({
        'name': tag,
        'locator': str(x)+', '+str(y)+', '+str(fwhm)})
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
                 range=[np.amin(x)-20, np.amax(y)+20],
                 fixedrange=False))

    return fig


# Applying tags to graph along with baselines or fit curves from the peak
# fitting calls
def update_annotation_helper(rows, x, y, g_unfit=None, g_fit=None,
                             baseline=None):
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
    if baseline is not None:
        figure.add_trace(
                go.Scatter(x=x, y=baseline(x), mode='lines', name='baseline'))
        if g_unfit is not None:
            figure.add_trace(
                    go.Scatter(
                        x=x,
                        y=g_unfit(x)+baseline(x),
                        mode='lines',
                        name='unfit'))
        if g_fit is not None:
            figure.add_trace(
                    go.Scatter(
                        x=x,
                        y=g_fit(x)+baseline(x),
                        mode='lines',
                        name='fit'))
    else:
        if g_unfit is not None:
            figure.add_trace(
                    go.Scatter(x=x, y=g_unfit(x), mode='lines', name='unfit'))
        if g_fit is not None:
            figure.add_trace(
                    go.Scatter(x=x, y=g_fit(x), mode='lines', name='fit'))

    return figure


# parsing splash-ml files found and upload contents to content section of
# webpage
def parse_splash_ml(contents, filename, uid, tags, index):

    try:
        # Different if statments to hopefully handel the files types needed
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
            temp = dict(Tag=i['name'], Peak=str(x+', '+y), FWHM=fwhm)
            tags_data.append(temp)
        else:
            temp = dict(Tag=i['name'])
            tags_data.append(temp)

    # split down data to work with new get_fig function
    data = pd.DataFrame.to_numpy(df)
    x = data[:, 0]
    if len(df.columns) == 2:
        y = data[:, 1]
    else:
        y = []
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

    graphData = [
            html.H5(
                id={'type': 'splash_location', 'index': index},
                children='uri: '+filename),
            html.H6(
                id={'type': 'splash_uid', 'index': index},
                children='uid: '+uid),

            # Graph of csv file
            graph,
            dcc.Checklist(
                id={'type': 'baseline', 'index': index},
                options=[
                    {'label': 'Apply Baseline to Peak Fitting',
                        'value': 'Apply Baseline to Data'}],
                style={'padding-left': '3rem'}),
            html.H6(children=['Only select peak number if you tag window'],
                    style={'padding-left': '3rem'}),
            html.Div(
                children=[
                    html.H6('Select peaks to find in'),
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
                    'width': '30rem',
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
                            {'name': 'FWHM', 'id': 'FWHM'}],
                        data=tags_data,
                        style_cell={'padding': '1rem'}),
                    html.Button(
                        id={'type': 'save_splash', 'index': index},
                        children='Save Table to Splash',
                        style={
                            'padding-bottom': '3rem'})],
                style={
                    'margin-left': '33rem',
                    'margin-right': '2rem',
                    'padding': '8rem 3rem 3rem 3rem'})]

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

    npyArr = 'idk man'
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
    graph = dcc.Graph(
            id={'type': 'graph', 'index': index},
            figure=get_fig(x, y),
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

    graphData = [
            html.H5(
                id={'type': 'filename', 'index': index},
                children=filename),
            html.H6(datetime.datetime.fromtimestamp(date)),

            # Graph of csv file
            graph,
            dcc.Checklist(
                id={'type': 'baseline', 'index': index},
                options=[
                    {'label': 'Apply Baseline to Peak Fitting',
                        'value': 'Apply Baseline to Data'}],
                style={'padding-left': '3rem'}),
            html.H6(children=['Only select peak number if you tag window'],
                    style={'padding-left': '3rem'}),
            html.Div(
                children=[
                    html.H6('Select peaks to find in'),
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
                    'width': '30rem',
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
                            {'name': 'FWHM', 'id': 'FWHM'}],
                        style_cell={'padding': '1rem'},
                        row_deletable=True),
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
                    'padding': '8rem 3rem 3rem 3rem'})]

    return html.Div(
            children=graphData,
            style={
                'box-shadow': '0 4px 8px 0 rgba(0,0,0,0.2)',
                'border-radius': '20px',
                'padding': '30px 30px',
                'margin': '10px'})


# Takes the tags and converts them to a .csv format to save locally for the
# user on the webpage
def save_local_file(rows_of_tags, file_name):
    # getting rid of .csv from file name to add _tags.csv to end
    tags_file = file_name[:-4]+'_tags.csv'
    f = open('./'+tags_file, 'w')
    for i in rows_of_tags:
        x1 = i['Peak'].split()[0][:-1]
        x2 = i['Peak'].split()[1]
        f.write(i['Tag']+','+x1+','+x2+','+i['FWHM']+'\n')
    f.close()

    res = dcc.send_file('./'+tags_file)
    os.remove('./'+tags_file)
    return res


# The actual peak finding based off of x and y data provided
def peak_helper(x_data, y_data, num_peaks):
    flag_list = []
    total_p = signal.find_peaks_cwt(y_data, 1)
    total_img = signal.cwt(y_data, signal.ricker, list(range(1, 10)))
    total_img = np.log(total_img+1)
    if len(total_p) == 0:
        return [], [], None, None
    temp_list = []
    return_p = []
    if len(total_p > num_peaks):
        for i in total_p:
            temp_list.append((y_data[i], i))
        temp_list.sort()
        temp_list = temp_list[-num_peaks:]
        for i in temp_list:
            return_p.append(i[1])
        return_p.sort()

    g_unfit = None
    g_fit = None
    difference = x_data[1] - x_data[0]
    for i in return_p:
        largest_width = 0
        for i_img in range(len(total_img)):
            if total_img[i_img][i] > largest_width:
                largest_width = total_img[i_img][i]
        stddev = (largest_width*difference)/(2*math.sqrt(2*math.log(2)))

        g_init = models.Gaussian1D(
                amplitude=y_data[i],
                mean=x_data[i],
                stddev=stddev)
        g_init.mean.min = float(x_data[0])
        g_init.mean.max = float(x_data[-1])
        g_init.amplitude.min = 0
        if g_unfit is None:
            g_unfit = g_init
        else:
            g_unfit = g_unfit+g_init
    fit_g = fitting.SimplexLSQFitter()
    if len(return_p) == 1:
        fit_g = fitting.LevMarLSQFitter()
    g_fit = fit_g(g_unfit, x_data, y_data)

    FWHM_list = []
    if len(return_p) == 1:
        FWHM_list.append(g_fit.stddev.value)
    else:
        for i in range(len(return_p)):
            FWHM_list.append(getattr(g_fit, f"stddev_{i}"))
            FWHM_list[-1] = FWHM_list[-1].value
    residual = 0
    fit_total = 0
    y_total = 0
    for i in range(len(x_data)):
        fit_total += g_fit(x_data[i])
        y_total += y_data[i]
    residual = fit_total/y_total
    residual = abs(1-residual)
    if residual > 0.15:
        for i in return_p:
            flag_list.append(1)
    else:
        for i in return_p:
            flag_list.append(0)

    return return_p, FWHM_list, flag_list, g_unfit, g_fit


# The logic on fitting peaks to data split up by blocks, or all the data
# together.  If data is split by blocks, we attempt to fit 3 peaks to the
# section
def get_peaks(x_data, y_data, num_peaks, baseline=None, block=None):
    base_model = None
    g_unfit = None
    g_fit = None
    # Linear Model from data on the left wall of the window to data on the
    # right wall of the window
    if baseline:
        slope = (y_data[-1] - y_data[0])/(x_data[-1] - x_data[0])
        intercept = y_data[0] - (slope * x_data[0])
        base_model = models.Linear1D(slope=slope, intercept=intercept)
        for i in range(len(y_data)):
            y_data[i] = y_data[i] - base_model(x_data[i])

    FWHM_list = []
    peak_list = []
    flag_list = []

    if block:
        boundaries = bayesian_block_finder(np.array(x_data), np.array(y_data))
        for bound_i in range(len(boundaries)):
            lower = int(boundaries[bound_i])
            if bound_i == (len(boundaries)-1):
                upper = len(x_data)
            else:
                upper = int(boundaries[bound_i+1])
            temp_x = x_data[lower:upper]
            temp_y = y_data[lower:upper]
            temp_peak, temp_FWHM, unfit, temp_flag, fit = peak_helper(
                    temp_x,
                    temp_y,
                    3)
            temp_peak = [i+lower for i in temp_peak]
            flag_list.extend(temp_flag)
            FWHM_list.extend(temp_FWHM)
            peak_list.extend(temp_peak)
    else:
        peak_list, FWHM_list, flag_list, g_unfit, g_fit = peak_helper(
                        x_data,
                        y_data,
                        num_peaks)

    return_list = []
    for i in range(len(peak_list)):
        diction = {}
        diction['index'] = peak_list[i]
        diction['FWHM'] = FWHM_list[i]
        diction['flag'] = flag_list[i]
        return_list.append(diction)

    return return_list, g_unfit, g_fit, base_model


# A callback function that automatically scales the range of the interactive
# graph.  This allows users to zoom in on the x scale and get a y scale that
# allows for a better view of the data
def zoom(change):
    input_states = dash.callback_context.states
    state_iter = iter(input_states.values())
    figure = next(state_iter)
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
    y_range = [np.amin(in_view)-20, np.amax(in_view)+20]
    figure['layout']['yaxis']['range'] = y_range
    return figure


# Targets the upload part of website
targeted_callback(
        zoom,
        Input({'type': 'graph', 'index': MATCH}, 'relayoutData'),
        Output({'type': 'graph', 'index': MATCH}, 'figure'),
        State({'type': 'graph', 'index': MATCH}, 'figure'),
        app=app)


# Targets the splash-ml part of website
targeted_callback(
        zoom,
        Input({'type': 'splash_graph', 'index': MATCH}, 'relayoutData'),
        Output({'type': 'splash_graph', 'index': MATCH}, 'figure'),
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
            peak_info, g_unfit, g_fit, base_model = get_peaks(
                    x_data[start:end],
                    y_data[start:end],
                    num_peaks)
        else:
            peak_info, g_unfit, g_fit, base_model = get_peaks(
                    x_data[start:end],
                    y_data[start:end],
                    num_peaks,
                    baseline=True)

        for i in peak_info:
            index = i['index']+start
            x, y = x_data[index], y_data[index]
            fwhm = i['FWHM']
            temp = dict(
                    Tag=tag,
                    Peak=str(x)+', '+str(y),
                    FWHM=str(fwhm))
            if rows:
                rows.append(temp)
            else:
                rows = [temp]

        x_data = figure['data'][0]['x']
        y_data = figure['data'][0]['y']

        figure = update_annotation_helper(
                rows, x_data, y_data, g_unfit, g_fit, base_model)

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
            peak_info, g_unfit, g_fit, base_model = get_peaks(
                    x_data[start:end],
                    y_data[start:end],
                    3,
                    block=True)
        else:
            peak_info, g_unfit, g_fit, base_model = get_peaks(
                    x_data[start:end],
                    y_data[start:end],
                    3,
                    baseline=True,
                    block=True)

        for i in peak_info:
            index = i['index']+start
            x, y = x_data[index], y_data[index]
            fwhm = i['FWHM']
            temp = dict(
                    Tag=tag,
                    Peak=str(x)+', '+str(y),
                    FWHM=str(fwhm))
            if rows:
                rows.append(temp)
            else:
                rows = [temp]

        x_data = figure['data'][0]['x']
        y_data = figure['data'][0]['y']

        figure = update_annotation_helper(
                rows, x_data, y_data, g_unfit, g_fit, base_model)

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
        State({'type': 'input_peaks', 'index': MATCH}, 'value'),
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


# Order of the table and splash-ml might be different but that shouldn't matter
# for the current usecase.  Upload to splash-ml happens later, this only
# handles adding window tagged data to table
def single_tags_splash(n_clicks):
    input_states = dash.callback_context.states
    state_iter = iter(input_states.values())
    rows = next(state_iter)
    tag = next(state_iter)
    num_peaks = next(state_iter)
    baseline = next(state_iter)
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
            peak_info, g_unfit, g_fit, base_model = get_peaks(
                    x_data[start:end],
                    y_data[start:end],
                    num_peaks)
        else:
            peak_info, g_unfit, g_fit, base_model = get_peaks(
                    x_data[start:end],
                    y_data[start:end],
                    num_peaks,
                    baseline=True)

        for i in peak_info:
            index = i['index']+start
            x, y = x_data[index], y_data[index]
            fwhm = i['FWHM']
            temp = dict(
                    Tag=tag,
                    Peak=str(x)+', '+str(y),
                    FWHM=str(fwhm))
            if rows:
                rows.append(temp)
            else:
                rows = [temp]

        x_data = figure['data'][0]['x']
        y_data = figure['data'][0]['y']

        figure = update_annotation_helper(
                rows, x_data, y_data, g_unfit, g_fit, base_model)

        global stash_figure
        stash_figure = figure

    return rows


targeted_callback(
        single_tags_splash,
        Input({'type': 'add_splash_tags', 'index': MATCH}, 'n_clicks'),
        Output({'type': 'splash_tag_table', 'index': MATCH}, 'data'),
        State({'type': 'splash_tag_table', 'index': MATCH}, 'data'),
        State({'type': 'splash_tags', 'index': MATCH}, 'value'),
        State({'type': 'splash_peaks', 'index': MATCH}, 'value'),
        State({'type': 'baseline', 'index': MATCH}, 'children'),
        State({'type': 'splash_graph', 'index': MATCH}, 'figure'),
        app=app)


# Order of the table and splash-ml might be different but that shouldn't matter
# for the current usecase.  Upload to splash-ml happens later, this only
# handles adding block tagged data to table
def multi_tags_splash(n_clicks):
    input_states = dash.callback_context.states
    state_iter = iter(input_states.values())
    rows = next(state_iter)
    tag = next(state_iter)
    num_peaks = next(state_iter)
    baseline = next(state_iter)
    figure = next(state_iter)
    num_peaks = num_peaks
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
            peak_info, g_unfit, g_fit, base_model = get_peaks(
                    x_data[start:end],
                    y_data[start:end],
                    3,
                    block=True)
        else:
            peak_info, g_unfit, g_fit, base_model = get_peaks(
                    x_data[start:end],
                    y_data[start:end],
                    3,
                    baseline=True,
                    block=True)

        for i in range(len(peak_info)):
            index = i['index']+start
            x, y = x_data[index], y_data[index]
            fwhm = i['FWHM']
            temp = dict(
                    Tag=tag,
                    Peak=str(x)+', '+str(y),
                    FWHM=str(fwhm))
            if rows:
                rows.append(temp)
            else:
                rows = [temp]

        x_data = figure['data'][0]['x']
        y_data = figure['data'][0]['y']

        figure = update_annotation_helper(
                rows, x_data, y_data, g_unfit, g_fit, base_model)

        global stash_figure
        stash_figure = figure
    return rows


targeted_callback(
        multi_tags_splash,
        Input({'type': 'block_splash_tags', 'index': MATCH}, 'n_clicks'),
        Output({'type': 'splash_tag_table', 'index': MATCH}, 'data'),
        State({'type': 'splash_tag_table', 'index': MATCH}, 'data'),
        State({'type': 'splash_tags', 'index': MATCH}, 'value'),
        State({'type': 'splash_peaks', 'index': MATCH}, 'value'),
        State({'type': 'splash_uid', 'index': MATCH}, 'children'),
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
    if splash_data:
        offset = len(splash_data[0]['tags'])
        rows = rows[offset:len(rows)]
    for i in rows:
        x = i['Peak'].split()[0][:-1]
        y = i['Peak'].split()[1]
        response = splash_PATCH_call(
                i['Tag'],
                uid,
                x,
                y,
                i['FWHM'])
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
def update_graph_annotation(rows):
    global stash_figure
    if stash_figure:
        fig = stash_figure
        stash_figure = None
        return fig
    input_states = dash.callback_context.states
    figure = next(iter(input_states.values()))
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
    global stash_figure
    if stash_figure:
        fig = stash_figure
        stash_figure = None
        return fig
    input_states = dash.callback_context.states
    figure = next(iter(input_states.values()))
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
    app.run_server(debug=True)
