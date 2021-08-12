# Made by Ronald Pandolfi at
# https://gist.github.com/ronpandolfi/98b9f16d6946178bea01b0f7d786bccb
# Slight changes made Jul 15th to fit 80 char length and not use print f.
# print f was not working with borken down string lines

from typing import Callable
from dash._utils import create_callback_id
from dash.dependencies import State
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import dash
from dataclasses import dataclass
import json
import warnings

app = None

_targeted_callbacks = []


@dataclass
class Callback:
    input: Input
    output: Output
    callable: Callable


def _dispatcher(*_):
    triggered = dash.callback_context.triggered
    if not triggered:
        raise PreventUpdate

    for callback in _targeted_callbacks:
        _id, _property = triggered[0]['prop_id'].split('.')
        if '{' in _id:
            _id = json.loads(_id)
        _input = Input(_id, _property)
        _id, _property = dash.callback_context.outputs_list.values()
        _output = Output(_id, _property)
        if callback.input == _input and callback.output == _output:
            return_value = callback.callable(triggered[0]['value'])
            if return_value is None:
                warnings.warn(
                        'A callback returned None. Perhaps you forgot a return'
                        ' value? Callback: {}'.format(repr(callback.callable)))
            return return_value


def targeted_callback(callback, input: Input, output: Output, *states: State,
                      app=app, prevent_initial_call=None):
    if prevent_initial_call is None:
        prevent_initial_call = app.config.prevent_initial_callbacks

    callback_id = create_callback_id(output)
    if callback_id in app.callback_map:
        if app.callback_map[callback_id]["callback"].__name__ != '_dispatcher':
            raise ValueError(
                    'Attempting to use a targeted callback with an''output '
                    'already assigned to a standard callback. These are not '
                    'compatible.')

        for callback_spec in app._callback_list:
            if callback_spec['output'] == callback_id:
                if (callback_spec['prevent_initial_call'] !=
                        prevent_initial_call):
                    raise ValueError(
                            'A callback has already been registered to this '
                            'output with a conflicting value for '
                            'prevent_initial_callback. You should decide which'
                            ' you want.')
                callback_spec['inputs'].append(input.to_dict())
                callback_spec['state'].extend(
                        [state.to_dict() for state in states])
    else:
        app.callback(output, input, *states,
                     prevent_initial_call=prevent_initial_call)(_dispatcher)
    _targeted_callbacks.append(Callback(input, output, callback))


{"mode": "full", "isActive": False}
