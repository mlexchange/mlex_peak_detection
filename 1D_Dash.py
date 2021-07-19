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

from packages.targeted_callbacks import targeted_callback

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

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
# values are set to the default values of 0 and 10 at the moment as splash-ml
# integration and use case isnt fully explored
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
    url = 'http://127.0.0.1:8000/api/v0/datasets/'+uid[5:]+'/tags'
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
                type='linear'))

    return fig


# Tagging graph code used by both splash-ml and local callbacks
def update_annotation_helper(rows, x, y, g_unfit=None, g_fit=None):
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
    if g_unfit is not None:
        figure.add_trace(
                go.Scatter(x=x, y=g_unfit(x), mode='lines', name='unfit'))
    if g_fit is not None:
        figure.add_trace(
                go.Scatter(x=x, y=g_fit(x), mode='lines', name='fit'))

    return figure


# parsing splash-ml files found.
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
                        id={'type': 'upload_splash_tags', 'index': index},
                        children='Save Tag to Splash-ML',
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
                        style_cell={'padding': '1rem'})],
                style={
                    'margin-left': '33rem',
                    'margin-right': '2rem',
                    'margin-bottom': '4rem',
                    'padding': '3rem 3rem'})]

    return html.Div(
            children=graphData,
            style={
                'box-shadow': '0 4px 8px 0 rgba(0,0,0,0.2)',
                'border-radius': '20px',
                'padding': '30px 30px',
                'margin': '10px'})


# Parsing uploaded files to display graphically on the website
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
                        children='Add Tag',
                        style={
                            'padding-bottom': '3rem'})],
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
                    'padding': '3rem 3rem'})]

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


# Takes x,y data, find the peaks and fits Gassuian curves to them.
# Returns location of peaks and full width half max
# Takes x,y data, find the peaks and fits Gassuian curves to them.
# Returns location of peaks and full width half max
def get_peaks(x_data, y_data, num_peaks):
    total_p = signal.find_peaks_cwt(y_data, 1)
    total_img = signal.cwt(y_data, signal.ricker, list(range(1, 10)))
    total_img = np.log(total_img+1)
    if len(total_p) == 0:
        return [], []
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
    # If all else isnt working, check this again it might be an issue
    fit_g = fitting.SimplexLSQFitter()
    if len(return_p) == 1:
        fit_g = fitting.LevMarLSQFitter()
    g_fit = fit_g(g_unfit, x_data, y_data)

    print(g_fit)

    FWHM_list = []
    if len(return_p) == 1:
        FWHM_list.append(g_fit.stddev)
    else:
        for i in range(len(return_p)):
            FWHM_list.append(getattr(g_fit, f"stddev_{i}"))
            FWHM_list[-1] = FWHM_list[-1].value

    return return_p, FWHM_list, g_unfit, g_fit, total_img


# Combined the upload and splash-ml data graphs into one callback to simplify
# the whole process
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
# and saves it in current session
@app.callback(
        Output({'type': 'tag_table', 'index': MATCH}, 'data'),
        Output({'type': 'graph', 'index': MATCH}, 'figure'),
        Input({'type': 'apply_labels', 'index': MATCH}, 'n_clicks'),
        State({'type': 'tag_table', 'index': MATCH}, 'data'),
        State({'type': 'input_tags', 'index': MATCH}, 'value'),
        State({'type': 'input_peaks', 'index': MATCH}, 'value'),
        State({'type': 'graph', 'index': MATCH}, 'figure'),
        prevent_initial_call=True)
def apply_tags_table(n_clicks, rows, tag, num_peaks, figure):
    peaks = None
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

        peaks, peak_fits, g_unfit, g_fit, cwt_img = get_peaks(
                x_data[start:end],
                y_data[start:end],
                num_peaks)

        for i in range(len(peaks)):
            index = peaks[i]+start
            x, y = x_data[index], y_data[index]
            fwhm = peak_fits[i]
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

        figure = update_annotation_helper(rows, x_data, y_data, g_unfit, g_fit)

    return rows, figure


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


# Uploads tag to splash-ml along with adding to the tag table.  Order of the
# table and splash-ml might be different but that shouldn't matter for the
# current usecase.  This can be fixed by not adding a new tag right to the
# table and instead doing a GET call to grab the tag order from splash-ml
@app.callback(
        Output({'type': 'splash_response', 'index': MATCH}, 'children'),
        Output({'type': 'splash_tag_table', 'index': MATCH}, 'data'),
        Input({'type': 'upload_splash_tags', 'index': MATCH}, 'n_clicks'),
        State({'type': 'splash_tag_table', 'index': MATCH}, 'data'),
        State({'type': 'splash_tags', 'index': MATCH}, 'value'),
        State({'type': 'splash_peaks', 'index': MATCH}, 'value'),
        State({'type': 'splash_uid', 'index': MATCH}, 'children'),
        State({'type': 'splash_graph', 'index': MATCH}, 'figure'),
        prevent_initial_call=True)
def upload_tags_button(n_clicks, rows, tag, num_peaks, uid, figure):
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

        peaks, peak_fits = get_peaks(
                x_data[start:end],
                y_data[start:end],
                num_peaks)

        for i in range(len(peaks)):
            index = peaks[i]+start
            x, y = x_data[index], y_data[index]
            fwhm = peak_fits[i]
            code_response = splash_PATCH_call(
                    tag,
                    uid,
                    x,
                    y,
                    fwhm)
            # 200 for OK, 422 for validation error, 500 for server error
            if code_response == 200:
                temp = dict(
                        Tag=tag,
                        Peak=str(x)+', '+str(y),
                        FWHM=str(fwhm))
                if rows:
                    rows.append(temp)
                else:
                    rows = [temp]
            else:
                break
        return html.Div('Uploading Tags: '+str(code_response)), rows
    else:
        return html.Div('No Tags Selected'), rows


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


# Populates the graph with tags and the peak locations for splash upload
def update_splash_annotation(rows):
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
