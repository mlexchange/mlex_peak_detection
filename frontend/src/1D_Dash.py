import base64
import datetime
import io
import os
import pathlib
import zipfile

import dash
from dash.dependencies import Input, Output, State, MATCH
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import dash_uploader as du

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Imports for interacting with splash-ml api
import urllib.request
import random
import requests
import json

from packages.helpers import get_peaks, add_paths_from_dir

# Imported code by Ronald Pandolfi
from packages.targeted_callbacks import targeted_callback


class Tag():
    def __init__(self, tag_name, peak_x, peak_y, fwhm, flag, uid='TBD', color='TBD'):
        self.Tag = tag_name
        self.Peak = str(peak_x) + ', ' + str(peak_y)
        self.FWHM = str(fwhm)
        self.flag = flag
        self.tag_uid = uid
        self.COLOR = color


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, "../assets/mlex-style.css"]) #, 'https://codepen.io/chriddyp/pen/bWLwgP.css'])

# Figure of graph when there are unfit and fit curves added on.
global stash_figure
stash_figure = None

global DATA_DIR
DATA_DIR = os.environ['DATA_DIR']
UPLOAD_FOLDER_ROOT = os.environ['UPLOAD_FOLDER_ROOT']
SPLASH_URL = 'http://splash:80/api/v0'

du.configure_upload(app, UPLOAD_FOLDER_ROOT, use_upload_id=False)

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

# header
header = dbc.Navbar(
    dbc.Container([
        dbc.Row(
            [
                dbc.Col(
                    html.Img(id="logo",
                             src='assets/mlex.png',
                             height="60px"),
                    md="auto"),
                dbc.Col(
                    [html.Div(children=html.H3("MLExchange | Peak Detection"),
                              id="app-title")],
                    md=True,
                    align="center",
                )
            ],
            align="center",
        ),
        dbc.Row([
            dbc.Col([dbc.NavbarToggler(id="navbar-toggler")],
                    md=2)],
            align="center"),
    ], fluid=True),
    dark=True,
    color="dark",
    sticky="top",
)


# Sidebar content, this includes the titles with the splash-ml entry and query
# button
sidebar = dbc.Card(
    id='sidebar',
    children=[
        dbc.CardHeader(dbc.Label('Query from Splash-ML', className='mr-2')),
        dbc.CardBody([
            dcc.Input(id='GET_uri',
                      placeholder='Pick URI',
                      type='text',
                      style={'width': '100%', 'marginBottom': '5px'}),
            dcc.Input(id='GET_tag',
                      placeholder='Pick Tag',
                      type='text',
                      style={'width': '100%', 'marginBottom': '5px'}),
            dbc.Button('QUERY',
                       id='splash_ml_data',
                       className="ms-auto",
                       n_clicks=0,
                       style={'width': '100%'})
        ])
    ],
    style={'margin-left': '1rem', 'margin-right': '0rem'}
)


# Content section to the right of the sidebar.  This includes the upload bar
# and graphs to be tagged once loaded into app.
content = html.Div([
    du.Upload(
        id="upload_data",
        max_file_size=1800,  # 1800 Mb
        cancel_button=True,
        pause_button=True
    ),
    html.Div(
        children='Or Query Splash-ML',
        style={'textAlign': 'center'}),
    html.Div(id='output_data_upload')],
    style={'margin-left': '0rem', 'margin-top': '1rem', 'margin-right': '1rem'}
)


# Setting up initial webpage layout
app.layout = html.Div(children=[header,
                                dbc.Row(children=[dbc.Col(sidebar, width=3),
                                                  dbc.Col(content, width=9)],
                                        justify='center')
                                ]
                      )


# splash-ml GET request with uri and tag paramaters.  The offset and limit
# values are hard coded at the moment as splash-ml integration and use case
# isnt fully explored
def splash_GET_call(uri, tag, offset, limit):
    url = f'{SPLASH_URL}/datasets?'
    params = {}
    if uri:
        params['uris'] = [uri]
    if tag:
        params['tags'] = [tag]
    if offset:
        params['page[offset]'] = offset
    if limit:
        params['page[limit]'] = limit
    data = requests.get(url, params=params).json()
    return data


# Takes tags applied to data along with the UID of the splash-ml dataset. With
# those tags and UID it PATCHs to the database with the api.
def splash_PATCH_call(uid, tags2add, tags2remove):
    url = f'{SPLASH_URL}/datasets/' + uid + '/tags'
    data = {'add_tags': tags2add, 'remove_tags': list(tags2remove)}
    return requests.patch(url, json=data).status_code


# Takes tags applied to data along with the dataset filename. With those tags
# and URI it POSTs to the database with the api.
def splash_POST_call(uri, tags):
    url = f'{SPLASH_URL}/datasets/'
    dataset = {'type': 'file',
               'uri': uri,
               'tags': tags}
    return requests.post(url, json=dataset).status_code


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
                    line_color=i['COLOR'],
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

    color = random.sample(range(0, 0xFFFFFF), len(tags))
    list_colors = ['#' + ''.join('%06x' % shade) for shade in color]

    # Building the splash-ml table from tags already in the database
    tags_data = []
    for i, tag in enumerate(tags):
        if tag['locator']['path']['Xpeak']:
            x = tag['locator']['path']['Xpeak']
            y = tag['locator']['path']['Ypeak']
            fwhm = tag['locator']['path']['fwhm']
            temp = Tag(tag['name'], x, y, fwhm, tag['locator']['path']['flag'], tag['uid'], list_colors[i])
            tags_data.append(temp.__dict__)
        else:
            temp = Tag(tag['name'])
            print('MISSING TAG VALUES')
            tags_data.append(temp.__dict__)

    # split down data to work with new get_fig function
    data = pd.DataFrame.to_numpy(df)
    x = data[:, 0]
    if len(df.columns) == 2:
        y = data[:, 1]
    else:
        y = []

    graph_data = generate_graph(x, y, index, filename, tags_data, uid) # MERGING EVERYTHING

    return html.Div(
            children=graph_data,
            style={
                'box-shadow': '0 4px 8px 0 rgba(0,0,0,0.2)',
                'border-radius': '20px',
                'padding': '30px 30px',
                'margin': '10px'})


# Parsing uploaded files to display in content section of the website
def parse_contents(filename, index):
    npyArr = 'Error'
    try:
        # Different if statements to hopefully handle the files types needed
        # when graphing 1D data
        if filename.endswith('.csv'):
            # The user uploaded a CSV file
            df = pd.read_csv(filename)
            # Can't handle anything other than 3 columns for graphing
            if len(df.columns) != 2:
                raise
        if filename.endswith('.xdi'):
            # The user uploaded a XDI file
            df = parseXDI(filename)
        if filename.endswith('.npy'):
            # The user uploaded a numpy file
            npyArr = np.load(filename)
            df = pd.DataFrame({'Column1': npyArr})

    except Exception as e:
        print(f'File {filename} with error {e}')
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
    date = os.path.getmtime(filename)
    date = datetime.datetime.fromtimestamp(date)
    date = date.strftime('%Y-%m-%d %H:%M:%S')
    graph_data = generate_graph(x, y, index, filename, [], str(date))

    return html.Div(
            children=graph_data,
            style={
                'box-shadow': '0 4px 8px 0 rgba(0,0,0,0.2)',
                'border-radius': '20px',
                'padding': '30px 30px',
                'margin': '10px'})


# Generates the graph and populates the table with tags
def generate_graph(x, y, index, filename, tags_data, uid):
    # color = random.sample(range(0, 0xFFFFFF), len(tags_data))
    # list_colors = ['#'+''.join('%06x' % shade) for shade in color]
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

    graph_data = [
        html.H5(
            id={'type': 'filename', 'index': index},
            children=filename),
        html.H6(
            id={'type': 'uid', 'index': index},
            children='uid: ' + uid),
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
                'width': 'fit-content',
                'padding-bottom': '2rem'}),
        dbc.Row([
            dbc.Col(
                children=[
                    html.H6("Detection method:"),
                    dcc.Dropdown(
                        id={'type': 'params_selection', 'index': index},
                        options=[
                            {'label': 'Window', 'value': 'window'},
                            {'label': 'Blocks', 'value': 'blocks'}
                        ],
                        placeholder='Method',
                        value='blocks',
                        style={'width': '100%', 'marginBottom': '5px'}
                    ),
                    html.Div(id={'type': 'domain', 'index': index}),
                    html.Div(id={'type': 'show_num_peaks', 'index': index},
                             children=[dcc.Input(
                                 id={'type': 'input_peaks', 'index': index},
                                 type='number',
                                 min=0,
                                 placeholder='Number of Peaks',
                                 style={'width': '100%', 'marginBottom': '5px'})]
                             ),
                    dcc.Input(id={'type': 'input_tags', 'index': index},
                              type='text',
                              persistence=True,
                              placeholder='Tag Name',
                              style={'width': '100%', 'marginBottom': '5px'}),
                    dcc.Dropdown(id={'type': 'input_shape', 'index': index},
                                 options=[{'label': 'Gaussian', 'value': 'Gaussian'},
                                          {'label': 'Voigt', 'value': 'Voigt'}],
                                 placeholder='Peak Shape',
                                 style={'width': '100%', 'marginBottom': '5px'}),
                    dbc.Button('TAG',
                               id={'type': 'tag', 'index': index},
                               style={'width': '100%', 'marginBottom': '5px'})
                ], width=3),
            dbc.Col(
                html.Div(
                    children=[
                        html.H5('Current Tags'),
                        dash_table.DataTable(
                            id={'type': 'tag_table', 'index': index},
                            columns=[
                                {'name': 'Tag', 'id': 'Tag'},
                                {'name': 'Peak', 'id': 'Peak'},
                                {'name': 'FWHM', 'id': 'FWHM'},
                                {'name': 'flag', 'id': 'flag'},
                                {'name': 'COLOR', 'id': 'COLOR'},
                                {'name': 'UID', 'id': 'tag_uid'}
                            ],
                            data=tags_data,
                            style_data_conditional=[
                                {'if': {'row_index': i, 'column_id': 'COLOR'},
                                 'background-color': tags_data[i]['COLOR'],
                                 'color': tags_data[i]['COLOR']}
                                for i in range(len(tags_data))],
                            style_cell={'padding': '1rem',
                                        'textOverflow': 'ellipsis',
                                        'overflow': 'hidden',
                                        'overflowX': 'auto',
                                        'textAlign': 'left',
                                        'font-size': '15px'},
                            css=[{'selector': '.show-hide', 'rule': "display: none"}],
                            style_table={'overflowX': 'auto', 'marginBottom': '5px'},
                            hidden_columns=['tag_uid', 'flag'],
                            row_deletable=True),
                        dbc.Button('Download Tags',
                                   id={'type': 'save_labels', 'index': index},
                                   style={'width': '100%', 'marginBottom': '5px'}),
                        dcc.Download(id={
                            'type': 'download_csv_tags',
                            'index': index}),
                        html.Div(id={'type': 'saved_response', 'index': index}),
                        dbc.Button(
                            'Save Tags to Splash-ML',
                            id={'type': 'save_splash', 'index': index},
                            style={'width': '100%', 'marginBottom': '5px'}),
                        html.Div(id={'type': 'splash_response', 'index': index})
                    ]),
                width=9
            )
        ])
    ]
    return graph_data


@app.callback(
    Output({'type': 'show_num_peaks', 'index': MATCH}, 'style'),
    Input({'type': 'params_selection', 'index': MATCH}, 'value')
)
def get_params(selection):
    '''
    This callback displays the corresponding parameters per method
    :param selection: Peak detection method
    :return: style
    '''
    if selection == 'blocks':
        return {'display': 'none'}
    return {}


# Takes the tags and converts them to a .csv format to save locally for the
# user on the webpage
def save_local_file(rows_of_tags, file_name):
    # getting rid of .csv from file name to add _tags.csv to end
    tags_file = file_name[:-4]+'_tags.csv'
    f = open('/app/tmp/'+tags_file, 'w')
    for i in rows_of_tags:
        x1 = i['Peak'].split()[0][:-1]
        x2 = i['Peak'].split()[1]
        f.write(i['Tag']+','+x1+','+x2+','+i['FWHM']+','+str(i['flag'])+'\n')
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


# Combined the upload and splash-ml data graphs into one callback to simplify
# the whole process (could be split into targeted callbacks for organization)
@app.callback(
        Output('output_data_upload', 'children'),
        Output('splash_ml_data', 'n_clicks'),
        Input('upload_data', 'isCompleted'),
        Input('splash_ml_data', 'n_clicks'),
        State('upload_data', 'fileNames'),
        State('GET_uri', 'value'),
        State('GET_tag', 'value'),
        prevent_initial_call=True)
def update_output(iscompleted, n_clicks, upload_filename, uri, tagname):
    children = []
    # If splash-ml needs to populate the page
    if n_clicks:
        offset = 0
        limit = 10
        file_info = splash_GET_call(uri, tagname, offset, limit)
        children = []
        for i in range(len(file_info)):
            f_type = file_info[i]['type']
            if f_type == 'file':
                try:
                    filename = file_info[i]['uri']
                    c = open(filename, 'r')
                except Exception as e:
                    print(f'File {filename} with error {e}')
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

    elif iscompleted:
        list_filenames = []
        supported_formats = ['csv', 'npy']
        if upload_filename is not None:
            path_to_zip_file = pathlib.Path(UPLOAD_FOLDER_ROOT) / upload_filename[0]
            if upload_filename[0].split('.')[-1] == 'zip':  # unzip files and delete zip file
                zip_ref = zipfile.ZipFile(path_to_zip_file)  # create zipfile object
                path_to_folder = pathlib.Path(UPLOAD_FOLDER_ROOT) / upload_filename[0].split('.')[-2]
                if (upload_filename[0].split('.')[-2] + '/') in zip_ref.namelist():
                    zip_ref.extractall(pathlib.Path(UPLOAD_FOLDER_ROOT))  # extract file to dir
                else:
                    zip_ref.extractall(path_to_folder)
                zip_ref.close()  # close file
                os.remove(path_to_zip_file)
                list_filenames = add_paths_from_dir(str(path_to_folder), supported_formats, list_filenames)
            else:
                list_filenames.append(str(path_to_zip_file))

        # If local upload needs to populate the page
        if list_filenames:
            for indx, filename in enumerate(list_filenames):
                children.append(parse_contents(filename, indx))

    if len(children) == 0:
        children.append(html.Div('No data found to graph'))

    return children, 0


# Tag callback for when the graph is from uploaded data.  Updates the tag table
# and saves it in current session. Handles window tagging
def single_tags_table(n_clicks):
    input_states = dash.callback_context.states
    state_iter = iter(input_states.values())
    method = next(state_iter)
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
                    peak_shape,
                    False,
                    method == 'blocks'
            )
        else:
            peak_info, unfit_list, fit_list, residual, base_list = get_peaks(
                    x_data[start:end],
                    y_data[start:end],
                    num_peaks,
                    peak_shape,
                    True,
                    method == 'blocks'
            )

        color = random.sample(range(0, 0xFFFFFF), len(peak_info))
        list_colors = ['#' + ''.join('%06x' % shade) for shade in color]

        for i, peak in enumerate(peak_info):
            index = peak['index']+start
            x, y = x_data[index], y_data[index]
            fwhm = peak['FWHM']
            if peak['flag'] == 1:
                temp = Tag('(F)'+tag, x, y, fwhm, peak['flag'], 'TBD', list_colors[i])
                temp = temp.__dict__
            else:
                temp = Tag(tag, x, y, fwhm, peak['flag'], 'TBD', list_colors[i])
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
        Input({'type': 'tag', 'index': MATCH}, 'n_clicks'),
        Output({'type': 'tag_table', 'index': MATCH}, 'data'),
        State({'type': 'params_selection', 'index': MATCH}, 'value'),
        State({'type': 'baseline', 'index': MATCH}, 'value'),
        State({'type': 'tag_table', 'index': MATCH}, 'data'),
        State({'type': 'input_tags', 'index': MATCH}, 'value'),
        State({'type': 'input_peaks', 'index': MATCH}, 'value'),
        State({'type': 'input_shape', 'index': MATCH}, 'value'),
        State({'type': 'graph', 'index': MATCH}, 'figure'),
        app=app)


# Callback that updates the conditional formatting of the dash table
def update_table_color(rows):
    return [{'if': {'row_index': i, 'column_id': 'COLOR'},
             'background-color': rows[i]['COLOR'],
             'color': rows[i]['COLOR']}
            for i in range(len(rows))]


targeted_callback(
        update_table_color,
        Input({'type': 'tag_table', 'index': MATCH}, 'data'),
        Output({'type': 'tag_table', 'index': MATCH}, 'style_data_conditional'),
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
def save_tags_button(n_clicks, rows, filename):
    if n_clicks and rows:
        # delete the slash (/) if the data comes from splash-ml
        filename = filename.replace('/', '_')
        save = save_local_file(rows, filename)
        return save, html.Div('Downloading Tags')
    else:
        return None, html.Div('No Tags Selected')


# Handles upload of table data to splash-ml.  It first does a GET call, and
# then compares to figure out what data needs to be uploaded.  Currently no
# way to really compare tag to tag, so I just compare size of tag list
def update_splash_data(n_clicks):
    input_states = dash.callback_context.states
    state_iter = iter(input_states.values())
    rows = next(state_iter)
    uri = next(state_iter)
    uid = next(state_iter)
    uid = uid[5:]
    current_tag_uid = [row['tag_uid'] for row in rows]
    row_idx_add = [i for i, e in enumerate(current_tag_uid) if e == 'TBD']
    tags2add = []
    for idx in row_idx_add:
        i = rows[idx]
        peak = i['Peak'].split()
        x = float(peak[0][:-1])
        y = float(peak[1][:-1])
        tags2add.append({'name': i['Tag'],
                         'locator': {'spec': 'Peak location',
                                     'path': {'Xpeak': x,
                                              'Ypeak': y,
                                              'fwhm': i['FWHM'],
                                              'flag': i['flag']}}
                         })
    try:
        datetime.datetime.strptime(uid, '%Y-%m-%d %H:%M:%S')
        response = splash_POST_call(uri, tags2add)
    except ValueError:
        splash_data = splash_GET_call(None, None, 0, 1, uid)
        splash_tags = splash_data[0]['tags']
        splash_tags_uid = [tag['uid'] for tag in splash_tags]
        tags2remove = np.setdiff1d(splash_tags_uid, current_tag_uid)
        response = splash_PATCH_call(uid, tags2add, tags2remove)
    if response != 200:
        return html.Div('Response: '+str(response))
    return html.Div('Response: 200')


targeted_callback(
        update_splash_data,
        Input({'type': 'save_splash', 'index': MATCH}, 'n_clicks'),
        Output({'type': 'splash_response', 'index': MATCH}, 'children'),
        State({'type': 'tag_table', 'index': MATCH}, 'data'),
        State({'type': 'filename', 'index': MATCH}, 'children'),
        State({'type': 'uid', 'index': MATCH}, 'children'),
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


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')
