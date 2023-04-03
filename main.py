import argparse

from dash import Dash, html, dcc, Input, Output
from dash import dcc
import dash_bootstrap_components as dbc

from dashboard.config import all_config
from tcn import run as run_f

parser = argparse.ArgumentParser(
    description='PyTorch video prediction model - TCTN')
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

input_list = []
ag_list = []
f_list = []


for name, con in all_config.items():
    f_list.append(
        html.H1(name, style={'color': '#ccccccc', 'textAlign': 'center'}),)
    config = con.schema()
    for key, val in config['properties'].items():
        if 'title' in val:
            if val['type'] == 'string':
                value = ''
                if 'default' in val:
                    value = val['default']
                ag_list.append(key)
                input_list.append(Input(component_id=str(
                    key)+'1', component_property="value"))
                f_list.append(dbc.Col(width="3", children=[
                    dbc.Label(val['title']+':'),
                    dbc.Input(value=value, type='text',
                              className="form-control mb-3", id=str(key)+'1')
                ]))
            if val['type'] == 'integer':
                value = 0
                if 'default' in val:
                    value = val['default']
                ag_list.append(key)
                input_list.append(Input(component_id=str(
                    key)+'1', component_property="value"))
                f_list.append(dbc.Col(width="3", children=[
                    dbc.Label(val['title']),
                    dbc.Input(value=value, type='number',
                              className="form-control mb-3", id=str(key)+'1')
                ]))
            if val['type'] == 'boolean':
                deff = 'False'
                if val['default']:
                    deff = 'True'
                ag_list.append(key)
                input_list.append(Input(component_id=str(
                    key)+'1', component_property="value"))
                f_list.append(dbc.Col(width="3", children=[
                    dbc.Label(val['title']),
                    dbc.RadioItems(['True', 'False'], deff,
                                   className="form-control mb-3", id=str(key)+'1')
                ]))

            if val['type'] == 'array':
                ag_list.append(key)
                input_list.append(Input(component_id=str(
                    key)+'1', component_property="value"))
                f_list.append(dbc.Col(width="3", children=[
                    dbc.Label(val['title']),
                    dcc.Dropdown(val['default'], multi=True, id=str(key)+'1')
                ]))

            if val['type'] is None:
                ag_list.append(key)
                input_list.append(Input(component_id=str(
                    key)+'1', component_property="value"))
                f_list.append(dbc.Col(width="3", children=[
                    dbc.Label(val['title']+':'),
                    dbc.Input(value=value, type='text',
                              className="form-control mb-3", id=str(key)+'1')
                ]))

ff = html.Div(dbc.Row([
    dbc.Form(children=[dbc.Row(f_list)], id="main-form"),
    html.A(dbc.Button('finished', className="btn btn-primary gap-2 col-1 mx-auto",
           color='primary', id='button-submit', n_clicks=0), href='/'),
    html.Div(id='body-div')
]), className='p-5')

app.layout = html.Div(ff)

input_list = [Input(component_id='button-submit',
                    component_property="n_clicks")] + input_list


def check_type(whatever):
    if whatever == 'True':
        return bool, True
    elif whatever == 'False':
        return bool, False
    elif isinstance(whatever, int):
        return int, int(whatever) or 0
    return str, whatever


def parse_vals(ag_vals):

    print(len(ag_list))
    print(len(ag_vals))
    for i, j in zip(ag_list, ag_vals):
        typo, val = check_type(j)
        parser.add_argument("--{0}".format(i), type=typo, default=val)
    return parser


@app.callback(Output('body-div', 'children'), input_list)
def submit_message(n, *ag_list):
    try:
        n = int(n)
    except:
        n = None
    if n is not None and n > 0:
        if int(n) > 0:
            run_f.starter1(parse_vals(list(ag_list)))
            # print(parse_vals(list(ag_list)), 'mmmmmmmmmmmmmmmmmmmmmmmm')


if __name__ == '__main__':
    app.run_server(debug=True)
