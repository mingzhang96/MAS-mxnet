## Implement of MAS on MXNet

This is an implement of MAS on MXNet. 

[Origin MAS on pytorch](https://github.com/rahafaljundi/MAS-Memory-Aware-Synapses)

## what does this project finish

* standard setup and training on several task.
* finally calculate accuracy on each task.

## how to use

1. clone the project
```shell
$ git clone https://github.com/Canappeco/MAS-mxnet.git
$ cd MAS-mxnet
$ mkdir ckpt && mkdir data && mkdir reg_params
```

2. We assume that you are in the $MAS-mxnet directory, and in $MAS-mxnet/data the mnist (.gz) data stays there.
```shell
python train_mnist.py
```

## result

*we use mlp instead of AlexNet as our base model.*

**notice: we use model trained on last task to test other tasks.**

#### 100 epoch, update_lr = 0.05, train_lr = 0.05

task | accuracy
---|---
01 | 0.6274231678486998
23 | 0.9417238001958864
45 | 0.9797225186766275
67 | 0.972306143001007
89 | 0.9389813414019162

#### 200 epoch, update_lr = 0.0001, train_lr = 0.0008

task | accuracy
---|---
01 | 0.9952718676122931
23 | 0.8805093046033301
45 | 0.955709711846318
67 | 0.9823766364551864
89 | 0.9536056480080686

#### 200 epoch, update_lr = 0.0001, train_lr = 0.005, fc2.output = 256

task | accuracy
---|---
01 | 0.9933806146572104
23 | 0.9299706170421156
45 | 0.9802561366061899
67 | 0.9914400805639476
89 | 0.9646999495713565

## tips

* the more tasks are, the more epoch need to train.
* use small train_lr to finetune.
* the last fc performs well if it has large output.
