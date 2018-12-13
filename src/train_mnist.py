import os
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
import fit
from util import download_file
import mxnet as mx
import numpy as np
import gzip, struct
import mlp
import weight_regularized_sgd
import MAS_Omega_Update
from util import *


def read_data(label, image):
    """
    download and read data into numpy
    """
    base_url = 'http://yann.lecun.com/exdb/mnist/'
    with gzip.open(os.path.join('../data', label)) as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        label = np.fromstring(flbl.read(), dtype=np.int8)
    with gzip.open(os.path.join('../data', image), 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
    return (label, image)


def to4d(img):
    """
    reshape to 4D arrays
    """
    return img.reshape(img.shape[0], 1, 28, 28).astype(np.float32)/255


def split_mnist_by_label(img, label, choose_label, ifupdate=False):
    assert len(choose_label) == 2
    pick_0 = np.where((label == choose_label[0]))
    img_0 = img[pick_0]
    label_0 = label[pick_0] - choose_label[0]

    pick_1 = np.where((label == choose_label[1]))
    img_1 = img[pick_1]
    if ifupdate:
        label_1 = label[pick_1] - choose_label[1]
    else:
        label_1 = label[pick_1] - choose_label[0]


    img = np.concatenate((img_0, img_1))
    label = np.concatenate((label_0, label_1))
    if ifupdate:
        label = mx.nd.zeros((label.shape[0], 2))
    return img, label


def get_mnist_iter(args, choose_label, ifupdate=False):
    """
    create data iterator with NDArrayIter
    """
    (train_lbl, train_img) = read_data(
            'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz')
    (val_lbl, val_img) = read_data(
            't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz')

    train_img, train_lbl = split_mnist_by_label(train_img, train_lbl, choose_label, ifupdate)
    val_img, val_lbl = split_mnist_by_label(val_img, val_lbl, choose_label, ifupdate)
    num_examples = train_img.shape[0]

    # print train_lbl
    # print val_lbl

    train = mx.io.NDArrayIter(
        to4d(train_img), train_lbl, args.batch_size, shuffle=True)
    val = mx.io.NDArrayIter(
        to4d(val_img), val_lbl, args.batch_size)
    return (train, val, num_examples)


def list_to_str(choose_label):
    return ''.join(str(x) for x in choose_label)


def get_val_iter(choose_label):
    (val_lbl, val_img) = read_data(
            't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz')
    val_img, val_lbl = split_mnist_by_label(val_img, val_lbl, choose_label, False)
    return to4d(val_img), val_lbl



def train_mission_first(parser, choose_label, save_dir_lasttime='', save_dir_thistime=None, load_epoch=None):
    if not os.path.isdir(os.path.join('..','ckpt', save_dir_thistime)):
        os.mkdir(os.path.join('..', 'ckpt', save_dir_thistime))

    model_prefix_lasttime = save_dir_lasttime
    model_save_dir_lasttime = os.path.join('..', 'ckpt', save_dir_lasttime)
    model_prefix_thistime = save_dir_thistime
    model_save_dir_thistime = os.path.join('..', 'ckpt', save_dir_thistime)

    parser.set_defaults(
        # network
        network        = 'mlp',
        # train
        gpus           = '2',
        batch_size     = 512,
        disp_batches   = 100,
        num_epochs     = 200,
        lr             = .005,
        lr_step_epochs = '100',
        optimizer      = "SGD",
        model_prefix_lasttime = model_prefix_lasttime,
        model_save_dir_lasttime = model_save_dir_lasttime,
        model_prefix_thistime = model_prefix_thistime,
        model_save_dir_thistime = model_save_dir_thistime,
        load_epoch     = load_epoch
    )
    args = parser.parse_args()

    (train, val, num_examples) = get_mnist_iter(args, choose_label, False)
    args.num_examples = num_examples

    # load network
    sym = mlp.get_symbol(**vars(args))

    # train
    fit.fit_normal(args, sym, train, val)


def train_mission_second_update_omega(parser, choose_label_lasttime, choose_label, save_dir_lasttime='', save_dir_thistime=None, load_epoch=None):
    if not os.path.isdir(os.path.join('..','ckpt', save_dir_thistime)):
        os.mkdir(os.path.join('..', 'ckpt', save_dir_thistime))

    model_prefix_lasttime = save_dir_lasttime
    model_save_dir_lasttime = os.path.join('..', 'ckpt', save_dir_lasttime)
    model_prefix_thistime = save_dir_thistime
    model_save_dir_thistime = os.path.join('..', 'ckpt', save_dir_thistime)

    parser.set_defaults(
        # network
        network        = 'mlp',
        # train
        gpus           = '2',
        batch_size     = 512,
        disp_batches   = 100,
        num_epochs     = 200,
        lr             = .0001,
        lr_step_epochs = '100',
        optimizer      = "MAS_Omega_Update",
        model_prefix_lasttime = model_prefix_lasttime,
        model_save_dir_lasttime = model_save_dir_lasttime,
        model_prefix_thistime = model_prefix_thistime,
        model_save_dir_thistime = model_save_dir_thistime,
        load_epoch     = load_epoch
    )
    args = parser.parse_args()

    (train, val, num_examples) = get_mnist_iter(args, choose_label_lasttime, True)
    args.num_examples = num_examples

    # load network
    sym = mlp.get_symbol_update_omega(**vars(args))

    # train
    model, reg_params = fit.fit_update_omega(args, sym, train, val)

    return model, reg_params


def train_mission_second_use_omega(parser, choose_label, save_dir_lasttime='', save_dir_thistime=None, load_epoch=None, model=None, reg_params={}):
    if not os.path.isdir(os.path.join('..','ckpt', save_dir_thistime)):
        os.mkdir(os.path.join('..', 'ckpt', save_dir_thistime))

    model_prefix_lasttime = save_dir_lasttime
    model_save_dir_lasttime = os.path.join('..', 'ckpt', save_dir_lasttime)
    model_prefix_thistime = save_dir_thistime
    model_save_dir_thistime = os.path.join('..', 'ckpt', save_dir_thistime)

    parser.set_defaults(
        # network
        network        = 'mlp',
        # train
        gpus           = '2',
        batch_size     = 512,
        disp_batches   = 100,
        num_epochs     = 200,
        lr             = .005,
        lr_step_epochs = '100',
        optimizer      = "weight_regularized_sgd",
        model_prefix_lasttime = model_prefix_lasttime,
        model_save_dir_lasttime = model_save_dir_lasttime,
        model_prefix_thistime = model_prefix_thistime,
        model_save_dir_thistime = model_save_dir_thistime,
        load_epoch     = load_epoch
    )
    args = parser.parse_args()

    (train, val, num_examples) = get_mnist_iter(args, choose_label, False)
    args.num_examples = num_examples

    # train
    model = fit.fit_use_omega(args, model, train, val, reg_params)
    if choose_label == [8,9]:
        for mission in [[0,1],[2,3],[4,5],[6,7],[8,9]]:
            val_img, val_lbl = get_val_iter(mission)
            model_path_old = os.path.join('../ckpt', list_to_str(mission), list_to_str(mission))
            model_path_new = os.path.join(model_save_dir_thistime, model_prefix_thistime)
            mission_accuracy = fit.test_model_accuracy(args, model_path_old, model_path_new, load_epoch, val_img, val_lbl)
            # mission_accuracy = fit.test_model(args, model_path_new, 9, val_img, val_lbl)
            print 'using model-'+list_to_str(choose_label)+' with fc by '+list_to_str(mission)+' get accuracy '+str(mission_accuracy)


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description="train mnist",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num-classes', type=int, default=2,
                        help='the number of classes')
    parser.add_argument('--num-examples', type=int, default=60000,
                        help='the number of training examples')

    parser.add_argument('--add_stn',  action="store_true", default=False, help='Add Spatial Transformer Network Layer (lenet only)')
    parser.add_argument('--image_shape', default='1, 28, 28', help='shape of training images')

    fit.add_fit_args(parser)

    choose_labels = [[0,1],[2,3],[4,5],[6,7],[8,9]]

    for index, choose_label in enumerate(choose_labels):
        if index == 0:
            train_mission_first(parser=parser, choose_label=choose_label, save_dir_thistime=list_to_str(choose_label))
        else:
            model, reg_params = train_mission_second_update_omega(
                                        parser=parser, 
                                        choose_label_lasttime=choose_labels[index-1], 
                                        choose_label=choose_label, 
                                        save_dir_lasttime=list_to_str(choose_labels[index-1]), 
                                        save_dir_thistime=list_to_str(choose_label), 
                                        load_epoch=200)
            train_mission_second_use_omega(
                                        parser=parser, 
                                        choose_label=choose_label, 
                                        save_dir_lasttime=list_to_str(choose_labels[index-1]), 
                                        save_dir_thistime=list_to_str(choose_label), 
                                        load_epoch=200, 
                                        model=model, 
                                        reg_params=reg_params)

