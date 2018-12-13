import subprocess
import os
import errno
import mxnet as mx
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])
import numpy as np


def download_file(url, local_fname=None, force_write=False):
    # requests is not default installed
    import requests
    if local_fname is None:
        local_fname = url.split('/')[-1]
    if not force_write and os.path.exists(local_fname):
        return local_fname

    dir_name = os.path.dirname(local_fname)

    if dir_name != "":
        if not os.path.exists(dir_name):
            try: # try to create the directory if it doesn't exists
                os.makedirs(dir_name)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise

    r = requests.get(url, stream=True)
    assert r.status_code == 200, "failed to open %s" % url
    with open(local_fname, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
    return local_fname

def get_gpus():
    """
    return a list of GPUs
    """
    try:
        re = subprocess.check_output(["nvidia-smi", "-L"], universal_newlines=True)
    except OSError:
        return []
    return range(len([i for i in re.split('\n') if 'GPU' in i]))


def show_model_function(model):
    print '--------------------------------type(model)---------------------------------'
    print type(model)
    print '---------model.get_params()----------'
    print model.get_params()
    print '---------model.get_params()[0]----------'
    print model.get_params()[0]
    print '---------model.get_params()[0].keys()----------'
    print model.get_params()[0].keys()
    print '---------model.get_params()["one layer"].asnumpy()----------'
    print model.get_params()[0][model.get_params()[0].keys()[0]].asnumpy()

    print '--------------------------------model._exec_group.grad_arrays---------------------------------'
    # print model.get_input_grads()
    print model._exec_group.grad_arrays
    print '--------------------------------type(model._exec_group)---------------------------------'
    print type(model._exec_group)
    print '--------------------------------model._exec_group---------------------------------'
    print model._exec_group
    print '--------------------------------model._exec_group.param_arrays---------------------------------'
    print model._exec_group.param_arrays
    print '--------------------------------model._exec_group.param_names---------------------------------'
    print model._exec_group.param_names


def initialize_firsttask_end(model, freeze_layers=[]):
    reg_params = {}
    for index, name in enumerate(model._exec_group.param_names):
        if not name in freeze_layers:
            reg_param = {}
            init_val = model._exec_group.param_arrays[index][0].copy()
            reg_param['init_val'] = init_val
            reg_param['omega'] = mx.nd.zeros(init_val.shape)
            reg_param['prev_omega'] = mx.nd.zeros(init_val.shape)
            reg_params[name] = reg_param
    return reg_params


def initialize_secondtask_end(model, reg_params, freeze_layers=[]):
    for index, name in enumerate(model._exec_group.param_names):
        if not name in freeze_layers:
            reg_param = reg_params[name]
            init_val = model._exec_group.param_arrays[index][0].copy()
            reg_param['init_val'] = init_val
            reg_param['prev_omega'] = reg_param['omega']
            reg_param['omega'] = mx.nd.zeros(init_val.shape)
            reg_params[name] = reg_param
    return reg_params


def clear_last_fc_weight(model, last_layer='relu2', num_classes=2):
    sym = model.symbol()
    arg_params = model._arg_params
    aux_params = model._aux_params
    
    all_layers = sym.get_internals()
    left_layers_names = []
    for index, x in enumerate(all_layers):
        left_layers_names.append(str(x.name))
        if x.name == last_layer:
            net = all_layers[index]
            break
    net  = mx.symbol.FullyConnected(data=net, name='fc3', num_hidden=num_classes)
    net  = mx.symbol.SoftmaxOutput(data=net, name='softmax')
    arg_params = dict({k:arg_params[k] for k in arg_params if k in left_layers_names})
    aux_params = dict({k:aux_params[k] for k in aux_params if k in left_layers_names})
    return net, arg_params, aux_params


def change_last_fc_weight(sym_old, arg_params_old, aux_params_old, sym_new, arg_params_new, aux_params_new, last_layer='relu2'):
    all_layers = sym_new.get_internals()
    left_layers_names = []
    for index, x in enumerate(all_layers):
        left_layers_names.append(str(x.name))
        if x.name == last_layer:
            break
    for key, value in arg_params_old.iteritems():
        if not key in left_layers_names:
            assert arg_params_new[key].shape == arg_params_old[key].shape
            arg_params_new[key] = arg_params_old[key] 

    return sym_new, arg_params_new, aux_params_new


def print_symbol(sym):
    net = sym.get_internals()
    for index, x in enumerate(net):
        print index, type(x), x, type(net[index]), net[index]


def accumelate_reg_params(reg_params, freeze_layers=[]):
    for name, reg_param in reg_params.iteritems():
        if reg_param.has_key('prev_omega') and reg_param.has_key('omega'):
            reg_param['omega'] = reg_param['omega'] + reg_param['prev_omega']
            reg_params[name] = reg_param
    return reg_params


def save_reg_params(reg_params, save_path):
    assert reg_params != []
    for name, reg_param in reg_params.iteritems():
        dst_path = os.path.join(save_path, name)
        if os.path.isfile(dst_path):
            os.remove(dst_path)
        mx.nd.save(dst_path, reg_param)
    return reg_params


def load_reg_params(save_path):
    reg_params = {}
    for f in os.listdir(save_path):
        src_path = os.path.join(save_path, f)
        if os.path.isfile(src_path):
            reg_param = mx.nd.load(src_path)
            reg_params[f] = reg_param
    return reg_params

def show_reg_params(reg_params):
    print 'fc2_weight', reg_params['fc2_weight']
    # for name, reg_param in reg_params.iteritems():
    #     print name, reg_param

