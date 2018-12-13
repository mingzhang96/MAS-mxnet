import logging
import os
import time
import re
import math
import mxnet as mx
import numpy as np
from util import *
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])


# def update_weights_params():


def get_epoch_size(args, kv):
    return math.ceil(int(args.num_examples / kv.num_workers) / args.batch_size)

def _get_lr_scheduler(args, kv):
    if 'lr_factor' not in args or args.lr_factor >= 1:
        return (args.lr, None)
    epoch_size = get_epoch_size(args, kv)
    begin_epoch = args.load_epoch if args.load_epoch else 0
    if 'pow' in args.lr_step_epochs:
        lr = args.lr
        max_up = args.num_epochs * epoch_size
        pwr = float(re.sub('pow[- ]*', '', args.lr_step_epochs))
        poly_sched = mx.lr_scheduler.PolyScheduler(max_up, lr, pwr)
        return (lr, poly_sched)
    step_epochs = [int(l) for l in args.lr_step_epochs.split(',')]
    lr = args.lr
    for s in step_epochs:
        if begin_epoch >= s:
            lr *= args.lr_factor
    if lr != args.lr:
        logging.info('Adjust learning rate to %e for epoch %d',
                     lr, begin_epoch)

    steps = [epoch_size * (x - begin_epoch)
             for x in step_epochs if x - begin_epoch > 0]
    if steps:
        return (lr, mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=args.lr_factor))
    else:
        return (lr, None)

def _load_model(args):
    if 'load_epoch' not in args or args.load_epoch is None:
        return (None, None, None)
    assert args.model_prefix_lasttime is not None
    model_prefix_lasttime = args.model_prefix_lasttime
    model_epoch = args.load_epoch
    if os.path.exists(os.path.join(args.model_save_dir_lasttime, "%s-symbol.json" % (model_prefix_lasttime))):
        model_prefix_lasttime = os.path.join(args.model_save_dir_lasttime, model_prefix_lasttime)
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix_lasttime, args.load_epoch)
    logging.info('Loaded model %s-%04d.params', model_prefix_lasttime, args.load_epoch)
    return (sym, arg_params, aux_params)


def _save_model(args):
    if args.model_prefix_thistime is None:
        return None
    # print args.model_save_dir_thistime, args.model_prefix_thistime
    print "save_model in", os.path.join(args.model_save_dir_thistime, args.model_prefix_thistime)
    return mx.callback.do_checkpoint(os.path.join(args.model_save_dir_thistime, args.model_prefix_thistime), period=args.save_period)


def add_fit_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    train = parser.add_argument_group('Training', 'model training')
    train.add_argument('--network', type=str,
                       help='the neural network to use')
    train.add_argument('--num-layers', type=int,
                       help='number of layers in the neural network, \
                             required by some networks such as resnet')
    train.add_argument('--gpus', type=str, default='2',
                       help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu')
    train.add_argument('--kv-store', type=str, default='device',
                       help='key-value store type')
    train.add_argument('--num-epochs', type=int, default=100,
                       help='max num of epochs')
    train.add_argument('--lr', type=float, default=0.1,
                       help='initial learning rate')
    train.add_argument('--lr-factor', type=float, default=0.1,
                       help='the ratio to reduce lr on each step')
    train.add_argument('--lr-step-epochs', type=str,
                       help='the epochs to reduce the lr, e.g. 30,60')
    train.add_argument('--initializer', type=str, default='default',
                       help='the initializer type')
    train.add_argument('--optimizer', type=str, default='sgd',
                       help='the optimizer type')
    train.add_argument('--mom', type=float, default=0.9,
                       help='momentum for sgd')
    train.add_argument('--wd', type=float, default=0.0001,
                       help='weight decay for sgd')
    train.add_argument('--batch-size', type=int, default=128,
                       help='the batch size')
    train.add_argument('--disp-batches', type=int, default=20,
                       help='show progress for every n batches')
    train.add_argument('--model-prefix-lasttime', type=str, help='model prefix last time')
    train.add_argument('--model-prefix-thistime', type=str, help='model prefix this time')
    train.add_argument('--model-save-dir-lasttime', type=str, help='model save dir last time')
    train.add_argument('--model-save-dir-thistime', type=str, help='model save dir this time')
    train.add_argument('--save-period', type=int, default=1, help='params saving period')
    parser.add_argument('--monitor', dest='monitor', type=int, default=0,
                        help='log network parameters every N iters if larger than 0')
    train.add_argument('--load-epoch', type=int,
                       help='load the model on an epoch using the model-load-prefix')
    train.add_argument('--top-k', type=int, default=0,
                       help='report the top-k accuracy. 0 means no report.')
    train.add_argument('--loss', type=str, default='',
                       help='show the cross-entropy or nll loss. ce strands for cross-entropy, nll-loss stands for likelihood loss')
    train.add_argument('--dtype', type=str, default='float32',
                       help='precision: float32 or float16')
    train.add_argument('--gc-type', type=str, default='none',
                       help='type of gradient compression to use, \
                             takes `2bit` or `none` for now')
    train.add_argument('--gc-threshold', type=float, default=0.5,
                       help='threshold for 2bit gradient compression')
    # additional parameters for large batch sgd
    train.add_argument('--macrobatch-size', type=int, default=0,
                       help='distributed effective batch size')
    train.add_argument('--warmup-epochs', type=int, default=5,
                       help='the epochs to ramp-up lr to scaled large-batch value')
    train.add_argument('--warmup-strategy', type=str, default='linear',
                       help='the ramping-up strategy for large batch sgd')
    train.add_argument('--profile-worker-suffix', type=str, default='',
                       help='profile workers actions into this file. During distributed training\
                             filename saved will be rank1_ followed by this suffix')
    train.add_argument('--profile-server-suffix', type=str, default='',
                       help='profile server actions into a file with name like rank1_ followed by this suffix \
                             during distributed training')
    train.add_argument('--reg-params-path', type=str, default='../reg_params', help='reg_params path')
    train.add_argument('--reg-lambda', type=int, default=1, help='reg_lambda')
    return train


def fit(args, network, train, val, **kwargs):
    """
    train a model
    args : argparse returns
    network : the symbol definition of the nerual network
    data_loader : function that returns the train and val data iterators
    """
    # kvstore
    kv = mx.kvstore.create(args.kv_store)
    if args.gc_type != 'none':
        kv.set_gradient_compression({'type': args.gc_type,
                                     'threshold': args.gc_threshold})
    # if args.profile_server_suffix:
    #     mx.profiler.set_config(filename=args.profile_server_suffix, profile_all=True, profile_process='server')
    #     mx.profiler.set_state(state='run', profile_process='server')

    # if args.profile_worker_suffix:
    #     if kv.num_workers > 1:
    #         filename = 'rank' + str(kv.rank) + '_' + args.profile_worker_suffix
    #     else:
    #         filename = args.profile_worker_suffix
    #     mx.profiler.set_config(filename=filename, profile_all=True, profile_process='worker')
    #     mx.profiler.set_state(state='run', profile_process='worker')

    # logging
    head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    logging.info('start with arguments %s', args)
    
    epoch_size = get_epoch_size(args, kv)

    # data iterators
    # (train, val) = data_loader(args, kv, choose_label)

    # if 'dist' in args.kv_store and not 'async' in args.kv_store:
    #     logging.info('Resizing training data to %d batches per machine', epoch_size)
    #     # resize train iter to ensure each machine has same number of batches per epoch
    #     # if not, dist_sync can hang at the end with one machine waiting for other machines
    #     train = mx.io.ResizeIter(train, epoch_size)

    # load model
    if 'arg_params' in kwargs and 'aux_params' in kwargs:
        arg_params = kwargs['arg_params']
        aux_params = kwargs['aux_params']
    else:
        sym, arg_params, aux_params = _load_model(args)
        if sym is not None:
            # print sym.tojson()
            # print '----------------'
            # print network.tojson()
            # print '-------type---------'
            # print type(network)
            # print '-------list_arguments---------'
            # print network.list_arguments()
            # print '-------list_outputs---------'
            # print network.list_outputs()
            assert sym.tojson() == network.tojson()

    # save model
    checkpoint = _save_model(args)

    # devices for training
    devs = mx.cpu() if args.gpus is None or args.gpus == "" else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]

    # learning rate
    lr, lr_scheduler = _get_lr_scheduler(args, kv)
    print "lr =", lr

    # create model
    model = mx.mod.Module(
        context=devs,
        symbol=network
    )

    lr_scheduler = lr_scheduler


    optimizer_params = {
        'learning_rate': lr,
        'wd': args.wd,
        'lr_scheduler': lr_scheduler,
        'multi_precision': True}

    # Only a limited number of optimizers have 'momentum' property
    has_momentum = {'sgd', 'dcasgd', 'nag', 'signum', 'lbsgd'}
    if args.optimizer in has_momentum:
        optimizer_params['momentum'] = args.mom

    monitor = mx.mon.Monitor(
        args.monitor, pattern=".*") if args.monitor > 0 else None

    # A limited number of optimizers have a warmup period
    has_warmup = {'lbsgd', 'lbnag'}
    if args.optimizer in has_warmup:
        nworkers = kv.num_workers
        if epoch_size < 1:
            epoch_size = 1
        macrobatch_size = args.macrobatch_size
        if macrobatch_size < args.batch_size * nworkers:
            macrobatch_size = args.batch_size * nworkers
        #batch_scale = round(float(macrobatch_size) / args.batch_size / nworkers +0.4999)
        batch_scale = math.ceil(
            float(macrobatch_size) / args.batch_size / nworkers)
        optimizer_params['updates_per_epoch'] = epoch_size
        optimizer_params['begin_epoch'] = args.load_epoch if args.load_epoch else 0
        optimizer_params['batch_scale'] = batch_scale
        optimizer_params['warmup_strategy'] = args.warmup_strategy
        optimizer_params['warmup_epochs'] = args.warmup_epochs
        optimizer_params['num_epochs'] = args.num_epochs

    if args.initializer == 'default':
        if args.network == 'alexnet':
            # AlexNet will not converge using Xavier
            initializer = mx.init.Normal()
            # VGG will not trend to converge using Xavier-Gaussian
        elif args.network and 'vgg' in args.network:
            initializer = mx.init.Xavier()
        else:
            initializer = mx.init.Xavier(
                rnd_type='gaussian', factor_type="in", magnitude=2)
    # initializer   = mx.init.Xavier(factor_type="in", magnitude=2.34),
    elif args.initializer == 'xavier':
        initializer = mx.init.Xavier()
    elif args.initializer == 'msra':
        initializer = mx.init.MSRAPrelu()
    elif args.initializer == 'orthogonal':
        initializer = mx.init.Orthogonal()
    elif args.initializer == 'normal':
        initializer = mx.init.Normal()
    elif args.initializer == 'uniform':
        initializer = mx.init.Uniform()
    elif args.initializer == 'one':
        initializer = mx.init.One()
    elif args.initializer == 'zero':
        initializer = mx.init.Zero()

    # evaluation metrices
    eval_metrics = ['accuracy']
    if args.top_k > 0:
        eval_metrics.append(mx.metric.create(
            'top_k_accuracy', top_k=args.top_k))

    supported_loss = ['ce', 'nll_loss']
    if len(args.loss) > 0:
        # ce or nll loss is only applicable to softmax output
        loss_type_list = args.loss.split(',')
        if 'softmax_output' in network.list_outputs():
            for loss_type in loss_type_list:
                loss_type = loss_type.strip()
                if loss_type == 'nll':
                    loss_type = 'nll_loss'
                if loss_type not in supported_loss:
                    logging.warning(loss_type + ' is not an valid loss type, only cross-entropy or ' \
                                    'negative likelihood loss is supported!')
                else:
                    eval_metrics.append(mx.metric.create(loss_type))
        else:
            logging.warning("The output is not softmax_output, loss argument will be skipped!")

    # callbacks that run after each batch
    batch_end_callbacks = [mx.callback.Speedometer(args.batch_size, args.disp_batches)]

    # run
    model.fit(train,
              # begin_epoch=args.load_epoch if args.load_epoch else 0,
              begin_epoch=0,
              num_epoch=args.num_epochs,
              eval_data=val,
              eval_metric=eval_metrics,
              kvstore=kv,
              optimizer=args.optimizer,
              optimizer_params=optimizer_params,
              initializer=initializer,
              arg_params=arg_params,
              aux_params=aux_params,
              batch_end_callback=batch_end_callbacks,
              epoch_end_callback=checkpoint,
              allow_missing=True,
              monitor=monitor)

    reg_params = initialize_first(model)
    save_reg_params(reg_params, args.reg_params_path)


def fit_normal(args, network, train, val, **kwargs):
    """
    train a model
    args : argparse returns
    network : the symbol definition of the nerual network
    data_loader : function that returns the train and val data iterators
    """
    # kvstore
    kv = mx.kvstore.create(args.kv_store)
    if args.gc_type != 'none':
        kv.set_gradient_compression({'type': args.gc_type,
                                     'threshold': args.gc_threshold})
    # logging
    head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    logging.info('start with arguments %s', args)
    
    epoch_size = get_epoch_size(args, kv)

    
    print network.tojson()
    # load model
    if 'arg_params' in kwargs and 'aux_params' in kwargs:
        arg_params = kwargs['arg_params']
        aux_params = kwargs['aux_params']
    else:
        sym, arg_params, aux_params = _load_model(args)
        if sym is not None:
            print sym.tojson()
            assert sym.tojson() == network.tojson()

    # save model
    checkpoint = _save_model(args)

    # devices for training
    devs = mx.cpu() if args.gpus is None or args.gpus == "" else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]

    # learning rate
    lr, lr_scheduler = _get_lr_scheduler(args, kv)
    print "lr =", lr

    # create model
    model = mx.mod.Module(
        context=devs,
        symbol=network
    )

    lr_scheduler = lr_scheduler


    optimizer_params = {
        'learning_rate': lr,
        'wd': args.wd,
        'lr_scheduler': lr_scheduler,
        'multi_precision': True}

    # Only a limited number of optimizers have 'momentum' property
    has_momentum = {'sgd', 'dcasgd', 'nag', 'signum', 'lbsgd'}
    if args.optimizer in has_momentum:
        optimizer_params['momentum'] = args.mom

    monitor = mx.mon.Monitor(
        args.monitor, pattern=".*") if args.monitor > 0 else None

    # A limited number of optimizers have a warmup period
    has_warmup = {'lbsgd', 'lbnag'}
    if args.optimizer in has_warmup:
        nworkers = kv.num_workers
        if epoch_size < 1:
            epoch_size = 1
        macrobatch_size = args.macrobatch_size
        if macrobatch_size < args.batch_size * nworkers:
            macrobatch_size = args.batch_size * nworkers
        #batch_scale = round(float(macrobatch_size) / args.batch_size / nworkers +0.4999)
        batch_scale = math.ceil(
            float(macrobatch_size) / args.batch_size / nworkers)
        optimizer_params['updates_per_epoch'] = epoch_size
        optimizer_params['begin_epoch'] = args.load_epoch if args.load_epoch else 0
        optimizer_params['batch_scale'] = batch_scale
        optimizer_params['warmup_strategy'] = args.warmup_strategy
        optimizer_params['warmup_epochs'] = args.warmup_epochs
        optimizer_params['num_epochs'] = args.num_epochs

    if args.initializer == 'default':
        if args.network == 'alexnet':
            # AlexNet will not converge using Xavier
            initializer = mx.init.Normal()
            # VGG will not trend to converge using Xavier-Gaussian
        elif args.network and 'vgg' in args.network:
            initializer = mx.init.Xavier()
        else:
            initializer = mx.init.Xavier(
                rnd_type='gaussian', factor_type="in", magnitude=2)
    # initializer   = mx.init.Xavier(factor_type="in", magnitude=2.34),
    elif args.initializer == 'xavier':
        initializer = mx.init.Xavier()
    elif args.initializer == 'msra':
        initializer = mx.init.MSRAPrelu()
    elif args.initializer == 'orthogonal':
        initializer = mx.init.Orthogonal()
    elif args.initializer == 'normal':
        initializer = mx.init.Normal()
    elif args.initializer == 'uniform':
        initializer = mx.init.Uniform()
    elif args.initializer == 'one':
        initializer = mx.init.One()
    elif args.initializer == 'zero':
        initializer = mx.init.Zero()

    # evaluation metrices
    eval_metrics = ['accuracy']
    if args.top_k > 0:
        eval_metrics.append(mx.metric.create(
            'top_k_accuracy', top_k=args.top_k))

    supported_loss = ['ce', 'nll_loss']
    if len(args.loss) > 0:
        # ce or nll loss is only applicable to softmax output
        loss_type_list = args.loss.split(',')
        if 'softmax_output' in network.list_outputs():
            for loss_type in loss_type_list:
                loss_type = loss_type.strip()
                if loss_type == 'nll':
                    loss_type = 'nll_loss'
                if loss_type not in supported_loss:
                    logging.warning(loss_type + ' is not an valid loss type, only cross-entropy or ' \
                                    'negative likelihood loss is supported!')
                else:
                    eval_metrics.append(mx.metric.create(loss_type))
        else:
            logging.warning("The output is not softmax_output, loss argument will be skipped!")

    # callbacks that run after each batch
    batch_end_callbacks = [mx.callback.Speedometer(args.batch_size, args.disp_batches)]

    # run
    model.fit(train,
              # begin_epoch=args.load_epoch if args.load_epoch else 0,
              begin_epoch=0,
              num_epoch=args.num_epochs,
              eval_data=val,
              eval_metric=eval_metrics,
              kvstore=kv,
              optimizer=args.optimizer,
              optimizer_params=optimizer_params,
              initializer=initializer,
              arg_params=arg_params,
              aux_params=aux_params,
              batch_end_callback=batch_end_callbacks,
              epoch_end_callback=checkpoint,
              allow_missing=True,
              monitor=monitor)

    reg_params = initialize_firsttask_end(model)
    save_reg_params(reg_params, args.reg_params_path)


def fit_update_omega(args, network, train, val, **kwargs):

    kv = mx.kvstore.create(args.kv_store)
    if args.gc_type != 'none':
        kv.set_gradient_compression({'type': args.gc_type,
                                     'threshold': args.gc_threshold})

    # logging
    head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    logging.info('start with arguments %s', args)
    
    epoch_size = get_epoch_size(args, kv)

    # load model
    if 'arg_params' in kwargs and 'aux_params' in kwargs:
        arg_params = kwargs['arg_params']
        aux_params = kwargs['aux_params']
    else:
        sym, arg_params, aux_params = _load_model(args)
        # if sym is not None:
        #     print sym.tojson()
        #     print '----------------'
        #     print network.tojson()
        #     print '-------type---------'
        #     print type(network)
        #     print '-------list_arguments---------'
        #     print network.list_arguments()
        #     print '-------list_outputs---------'
        #     print network.list_outputs()
        #     print '-------arg_params---------'
        #     print arg_params
        #     print '-------aux_params---------'
        #     print aux_params
        #     assert sym.tojson() == network.tojson()

    # save model
    # checkpoint = _save_model(args)

    # devices for training
    devs = mx.cpu() if args.gpus is None or args.gpus == "" else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]

    # learning rate
    lr, lr_scheduler = _get_lr_scheduler(args, kv)
    print "lr =", lr

    # create model
    model = mx.mod.Module(
        context=devs,
        symbol=network
    )

    # initialize omega
    reg_params = load_reg_params(args.reg_params_path)
    for name, reg_param in reg_params.iteritems():
        for key, value in reg_param.iteritems():
            reg_param[key] = value.as_in_context(devs[0])
    print '---------------------------load reg_params in', args.model_prefix_thistime, '----------------------------'
    show_reg_params(reg_params)
    print '----------------------------------------finish-----------------------------------------'

    lr_scheduler = lr_scheduler
    optimizer_params = {
        'reg_params': reg_params,
        'batch_size': args.batch_size,
        'learning_rate': lr,
        'wd': args.wd,
        'lr_scheduler': lr_scheduler,
        'multi_precision': True}

    # Only a limited number of optimizers have 'momentum' property
    has_momentum = {'sgd', 'dcasgd', 'nag', 'signum', 'lbsgd'}
    if args.optimizer in has_momentum:
        optimizer_params['momentum'] = args.mom

    monitor = mx.mon.Monitor(
        args.monitor, pattern=".*") if args.monitor > 0 else None

    # A limited number of optimizers have a warmup period
    has_warmup = {'lbsgd', 'lbnag'}
    if args.optimizer in has_warmup:
        nworkers = kv.num_workers
        if epoch_size < 1:
            epoch_size = 1
        macrobatch_size = args.macrobatch_size
        if macrobatch_size < args.batch_size * nworkers:
            macrobatch_size = args.batch_size * nworkers
        #batch_scale = round(float(macrobatch_size) / args.batch_size / nworkers +0.4999)
        batch_scale = math.ceil(
            float(macrobatch_size) / args.batch_size / nworkers)
        optimizer_params['updates_per_epoch'] = epoch_size
        optimizer_params['begin_epoch'] = args.load_epoch if args.load_epoch else 0
        optimizer_params['batch_scale'] = batch_scale
        optimizer_params['warmup_strategy'] = args.warmup_strategy
        optimizer_params['warmup_epochs'] = args.warmup_epochs
        optimizer_params['num_epochs'] = args.num_epochs

    if args.initializer == 'default':
        if args.network == 'alexnet':
            # AlexNet will not converge using Xavier
            initializer = mx.init.Normal()
            # VGG will not trend to converge using Xavier-Gaussian
        elif args.network and 'vgg' in args.network:
            initializer = mx.init.Xavier()
        else:
            initializer = mx.init.Xavier(
                rnd_type='gaussian', factor_type="in", magnitude=2)
    # initializer   = mx.init.Xavier(factor_type="in", magnitude=2.34),
    elif args.initializer == 'xavier':
        initializer = mx.init.Xavier()
    elif args.initializer == 'msra':
        initializer = mx.init.MSRAPrelu()
    elif args.initializer == 'orthogonal':
        initializer = mx.init.Orthogonal()
    elif args.initializer == 'normal':
        initializer = mx.init.Normal()
    elif args.initializer == 'uniform':
        initializer = mx.init.Uniform()
    elif args.initializer == 'one':
        initializer = mx.init.One()
    elif args.initializer == 'zero':
        initializer = mx.init.Zero()

    # evaluation metrices
    eval_metrics = ['accuracy']
    if args.top_k > 0:
        eval_metrics.append(mx.metric.create(
            'top_k_accuracy', top_k=args.top_k))

    supported_loss = ['ce', 'nll_loss']
    if len(args.loss) > 0:
        # ce or nll loss is only applicable to softmax output
        loss_type_list = args.loss.split(',')
        if 'softmax_output' in network.list_outputs():
            for loss_type in loss_type_list:
                loss_type = loss_type.strip()
                if loss_type == 'nll':
                    loss_type = 'nll_loss'
                if loss_type not in supported_loss:
                    logging.warning(loss_type + ' is not an valid loss type, only cross-entropy or ' \
                                    'negative likelihood loss is supported!')
                else:
                    eval_metrics.append(mx.metric.create(loss_type))
        else:
            logging.warning("The output is not softmax_output, loss argument will be skipped!")

    # callbacks that run after each batch
    batch_end_callbacks = [mx.callback.Speedometer(args.batch_size, args.disp_batches)]

    # run
    model.fit(train,
              # begin_epoch=args.load_epoch if args.load_epoch else 0,
              begin_epoch=0,
              num_epoch=args.num_epochs,
              eval_data=val,
              eval_metric=eval_metrics,
              kvstore=kv,
              optimizer=args.optimizer,
              optimizer_params=optimizer_params,
              initializer=initializer,
              arg_params=arg_params,
              aux_params=aux_params,
              batch_end_callback=batch_end_callbacks,
              # epoch_end_callback=checkpoint,
              allow_missing=True,
              monitor=monitor)

    # print '---------------------------after update_omega in', args.model_prefix_thistime, '-----------------------------'
    # show_reg_params(reg_params)
    # print '----------------------------------------finish-----------------------------------------'
    # reg_params = accumelate_reg_params(reg_params)
    # print '---------------------------after accumelate_reg_params in', args.model_prefix_thistime, '-----------------------------'
    # show_reg_params(reg_params)
    # print '----------------------------------------finish-----------------------------------------'
    # reg_params = initialize_secondtask_end(model, reg_params)
    # print '---------------------------after initialize_secondtask_end in', args.model_prefix_thistime, '-----------------------------'
    # show_reg_params(reg_params)
    # print '----------------------------------------finish-----------------------------------------'
    # save_reg_params(reg_params, args.reg_params_path)

    return model, reg_params


def fit_use_omega(args, model, train, val, reg_params, **kwargs):

    network, arg_params, aux_params = clear_last_fc_weight(model)
    print_symbol(network)

    kv = mx.kvstore.create(args.kv_store)
    if args.gc_type != 'none':
        kv.set_gradient_compression({'type': args.gc_type,
                                     'threshold': args.gc_threshold})

    # logging
    head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    logging.info('start with arguments %s', args)
    
    epoch_size = get_epoch_size(args, kv)

    # save model
    checkpoint = _save_model(args)

    # devices for training
    devs = mx.cpu() if args.gpus is None or args.gpus == "" else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]

    # learning rate
    lr, lr_scheduler = _get_lr_scheduler(args, kv)
    print "lr =", lr

    # create model
    model = mx.mod.Module(
        context=devs,
        symbol=network
    )

    # initialize omega
    # reg_params = load_reg_params(args.reg_params_path)
    # for name, reg_param in reg_params.iteritems():
    #     for key, value in reg_param.iteritems():
    #         reg_param[key] = value.as_in_context(devs[0])
    print '---------------------------load reg_params in', args.model_prefix_thistime, '----------------------------'
    show_reg_params(reg_params)
    print '----------------------------------------finish-----------------------------------------'

    lr_scheduler = lr_scheduler
    optimizer_params = {
        'reg_params': reg_params,
        'reg_lambda': args.reg_lambda,
        'learning_rate': lr,
        'wd': args.wd,
        'lr_scheduler': lr_scheduler,
        'multi_precision': True}

    # Only a limited number of optimizers have 'momentum' property
    has_momentum = {'sgd', 'dcasgd', 'nag', 'signum', 'lbsgd'}
    if args.optimizer in has_momentum:
        optimizer_params['momentum'] = args.mom

    monitor = mx.mon.Monitor(
        args.monitor, pattern=".*") if args.monitor > 0 else None

    # A limited number of optimizers have a warmup period
    has_warmup = {'lbsgd', 'lbnag'}
    if args.optimizer in has_warmup:
        nworkers = kv.num_workers
        if epoch_size < 1:
            epoch_size = 1
        macrobatch_size = args.macrobatch_size
        if macrobatch_size < args.batch_size * nworkers:
            macrobatch_size = args.batch_size * nworkers
        #batch_scale = round(float(macrobatch_size) / args.batch_size / nworkers +0.4999)
        batch_scale = math.ceil(
            float(macrobatch_size) / args.batch_size / nworkers)
        optimizer_params['updates_per_epoch'] = epoch_size
        optimizer_params['begin_epoch'] = args.load_epoch if args.load_epoch else 0
        optimizer_params['batch_scale'] = batch_scale
        optimizer_params['warmup_strategy'] = args.warmup_strategy
        optimizer_params['warmup_epochs'] = args.warmup_epochs
        optimizer_params['num_epochs'] = args.num_epochs

    if args.initializer == 'default':
        if args.network == 'alexnet':
            # AlexNet will not converge using Xavier
            initializer = mx.init.Normal()
            # VGG will not trend to converge using Xavier-Gaussian
        elif args.network and 'vgg' in args.network:
            initializer = mx.init.Xavier()
        else:
            initializer = mx.init.Xavier(
                rnd_type='gaussian', factor_type="in", magnitude=2)
    # initializer   = mx.init.Xavier(factor_type="in", magnitude=2.34),
    elif args.initializer == 'xavier':
        initializer = mx.init.Xavier()
    elif args.initializer == 'msra':
        initializer = mx.init.MSRAPrelu()
    elif args.initializer == 'orthogonal':
        initializer = mx.init.Orthogonal()
    elif args.initializer == 'normal':
        initializer = mx.init.Normal()
    elif args.initializer == 'uniform':
        initializer = mx.init.Uniform()
    elif args.initializer == 'one':
        initializer = mx.init.One()
    elif args.initializer == 'zero':
        initializer = mx.init.Zero()

    # evaluation metrices
    eval_metrics = ['accuracy']
    if args.top_k > 0:
        eval_metrics.append(mx.metric.create(
            'top_k_accuracy', top_k=args.top_k))

    supported_loss = ['ce', 'nll_loss']
    if len(args.loss) > 0:
        # ce or nll loss is only applicable to softmax output
        loss_type_list = args.loss.split(',')
        if 'softmax_output' in network.list_outputs():
            for loss_type in loss_type_list:
                loss_type = loss_type.strip()
                if loss_type == 'nll':
                    loss_type = 'nll_loss'
                if loss_type not in supported_loss:
                    logging.warning(loss_type + ' is not an valid loss type, only cross-entropy or ' \
                                    'negative likelihood loss is supported!')
                else:
                    eval_metrics.append(mx.metric.create(loss_type))
        else:
            logging.warning("The output is not softmax_output, loss argument will be skipped!")

    # callbacks that run after each batch
    batch_end_callbacks = [mx.callback.Speedometer(args.batch_size, args.disp_batches)]

    # run
    model.fit(train,
              # begin_epoch=args.load_epoch if args.load_epoch else 0,
              begin_epoch=0,
              num_epoch=args.num_epochs,
              eval_data=val,
              eval_metric=eval_metrics,
              kvstore=kv,
              optimizer=args.optimizer,
              optimizer_params=optimizer_params,
              initializer=initializer,
              arg_params=arg_params,
              aux_params=aux_params,
              batch_end_callback=batch_end_callbacks,
              epoch_end_callback=checkpoint,
              allow_missing=True,
              monitor=monitor)

    print '---------------------------after update_omega in', args.model_prefix_thistime, '-----------------------------'
    show_reg_params(reg_params)
    print '----------------------------------------finish-----------------------------------------'
    reg_params = accumelate_reg_params(reg_params)
    print '---------------------------after accumelate_reg_params in', args.model_prefix_thistime, '-----------------------------'
    show_reg_params(reg_params)
    print '----------------------------------------finish-----------------------------------------'
    reg_params = initialize_secondtask_end(model, reg_params)
    print '---------------------------after initialize_secondtask_end in', args.model_prefix_thistime, '-----------------------------'
    show_reg_params(reg_params)
    print '----------------------------------------finish-----------------------------------------'
    save_reg_params(reg_params, args.reg_params_path)

    return model


def test_model_accuracy(args, model_path_old, model_path_new, epoch, val_img, val_lbl):
    sym_old, arg_params_old, aux_params_old = mx.model.load_checkpoint(model_path_old, epoch)
    sym_new, arg_params_new, aux_params_new = mx.model.load_checkpoint(model_path_new, epoch)
    sym, arg_params, aux_params = change_last_fc_weight(sym_old, arg_params_old, aux_params_old, sym_new, arg_params_new, aux_params_new)
    model = mx.mod.Module(symbol=sym, context=mx.gpu(int(args.gpus)), label_names=None)
    model.bind(for_training=False, data_shapes=[('data', val_img.shape)], label_shapes=model._label_shapes)
    model.set_params(arg_params, aux_params, allow_missing=True)

    model.forward(Batch([mx.nd.array(val_img)]))
    prob = model.get_outputs()[0].asnumpy()
    # todo : get accuracy
    prob = np.argsort(prob)
    prob = np.where(prob==1)[1]
    assert prob.shape == val_lbl.shape
    accuracy = np.sum((prob==val_lbl)==1) / float(prob.shape[0])
    return accuracy



def test_model(args, model_path, epoch, val_img, val_lbl):
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_path, epoch)
    model = mx.mod.Module(symbol=sym, context=mx.gpu(int(args.gpus)), label_names=None)
    print model._label_shapes
    model.bind(for_training=False, data_shapes=[('data', val_img.shape)], label_shapes=model._label_shapes)
    model.set_params(arg_params, aux_params, allow_missing=True)

    model.forward(Batch([mx.nd.array(val_img)]))
    prob = model.get_outputs()[0].asnumpy()
    # todo : get accuracy
    prob = np.argsort(prob)
    prob = np.where(prob==1)[1]
    assert prob.shape == val_lbl.shape
    accuracy = np.sum((prob==val_lbl)==1) / float(prob.shape[0])
    return accuracy



