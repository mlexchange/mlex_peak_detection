import base64
import datetime
import io
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


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Setting up initial webpage layout
app.layout = html.Div(children=[
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
        html.Div(id='output_data_upload')])


def get_fig(x, y):
    if len(y) > 0:
        fig = go.Figure(
                go.Scatter(x=x, y=y))
    else:
        fig = px.line(x)
    fig.update_layout(
             xaxis=dict(
                rangeslider=dict(
                        visible=True),
                type='linear'))

    return fig


def update_annotation_helper(rows, x, y, g_unfit, g_fit):
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
    figure.add_trace(go.Scatter(x=x, y=g_unfit(x), mode='lines', name='unfit'))
    figure.add_trace(go.Scatter(x=x, y=g_fit(x), mode='lines', name='fit'))

    total_img = signal.cwt(y, signal.ricker, [1])
    figure.add_trace(go.Contour(x=total_img))
    return figure


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
            html.Div(id={'type': 'cwt_img', 'index': index}),
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


@app.callback(
        Output('output_data_upload', 'children'),
        Input('upload_data', 'contents'),
        State('upload_data', 'filename'),
        State('upload_data', 'last_modified'),
        prevent_initial_call=True)
def update_output(
        list_of_contents,
        list_of_names,
        list_of_dates):
    children = []
    # If local upload needs to populate the page
    for i in range(len(list_of_contents)):
        c = list_of_contents[i]
        n = list_of_names[i]
        d = list_of_dates[i]
        children.append(parse_contents(c, n, d, i))
    if len(children) == 0:
        children.append(html.Div('No data found to graph'))
    return children


@app.callback(
        Output({'type': 'tag_table', 'index': MATCH}, 'data'),
        Output({'type': 'graph', 'index': MATCH}, 'figure'),
        Output({'type': 'cwt_img', 'index': MATCH}, 'children'),
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
        fig = dcc.Graph(figure=px.imshow(cwt_img))
        # fig = html.Div()

    return rows, figure, fig


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


# Populates the graph with tags and the peak locations for local upload
#   @app.callback(
#           Output({'type': 'graph', 'index': MATCH}, 'figure'),
#           Input({'type': 'tag_table', 'index': MATCH}, 'data'),
#           State({'type': 'graph', 'index': MATCH}, 'figure'))
#   def update_graph_annotation(rows, figure):
#       input_states = dash.callback_context.states
#       figure = next(iter(input_states.values()))
#       x_data = figure['data'][0]['x']
#       y_data = figure['data'][0]['y']
#       return update_annotation_helper(rows, x_data, y_data)


if __name__ == '__main__':
    app.run_server(debug=True)
