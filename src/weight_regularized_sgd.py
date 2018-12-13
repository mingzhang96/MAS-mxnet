import mxnet as mx
import logging
import math
import pickle
import warnings
import numpy
from mxnet.ndarray import sgd_update, sgd_mom_update, NDArray

@mx.optimizer.Optimizer.register
class Weight_Regularized_SGD(mx.optimizer.Optimizer):
    """The SGD optimizer with momentum and weight decay.

    If the storage types of weight and grad are both ``row_sparse``, and ``lazy_update`` is True, \
    **lazy updates** are applied by::

        for row in grad.indices:
            rescaled_grad[row] = lr * rescale_grad * clip(grad[row], clip_gradient) + wd * weight[row]
            state[row] = momentum[row] * state[row] + rescaled_grad[row]
            weight[row] = weight[row] - state[row]

    The sparse update only updates the momentum for the weights whose row_sparse
    gradient indices appear in the current batch, rather than updating it for all
    indices. Compared with the original update, it can provide large
    improvements in model training throughput for some applications. However, it
    provides slightly different semantics than the original update, and
    may lead to different empirical results.

    Otherwise, **standard updates** are applied by::

        rescaled_grad = lr * rescale_grad * clip(grad, clip_gradient) + wd * weight
        state = momentum * state + rescaled_grad
        weight = weight - state

    For details of the update algorithm see
    :class:`~mxnet.ndarray.sgd_update` and :class:`~mxnet.ndarray.sgd_mom_update`.

    This optimizer accepts the following parameters in addition to those accepted
    by :class:`.Optimizer`.

    Parameters
    ----------
    momentum : float, optional
       The momentum value.
    lazy_update : bool, optional
       Default is True. If True, lazy updates are applied \
       if the storage types of weight and grad are both ``row_sparse``.
    multi_precision: bool, optional
       Flag to control the internal precision of the optimizer.
       ``False`` results in using the same precision as the weights (default),
       ``True`` makes internal 32-bit copy of the weights and applies gradients \
                in 32-bit precision even if actual weights used in the model have lower precision.\
                Turning this on can improve convergence and accuracy when training with float16.
    """
    def __init__(self, momentum=0.0, lazy_update=True, reg_params={}, reg_lambda=1, **kwargs):
        super(Weight_Regularized_SGD, self).__init__(**kwargs)
        self.momentum = momentum
        self.lazy_update = lazy_update
        self.reg_params = reg_params
        self.reg_lambda = reg_lambda

    def create_state_multi_precision(self, index, weight):
        weight_master_copy = None
        if self.multi_precision and weight.dtype == numpy.float16:
            weight_master_copy = weight.astype(numpy.float32)
            return (self.create_state(index, weight_master_copy), weight_master_copy)
        if weight.dtype == numpy.float16 and not self.multi_precision:
            warnings.warn("Accumulating with float16 in optimizer can lead to "
                          "poor accuracy or slow convergence. "
                          "Consider using multi_precision=True option of the "
                          "SGD optimizer")
        return self.create_state(index, weight)

    def create_state(self, index, weight):
        momentum = None
        stype = weight.stype if self.lazy_update else 'default'
        if self.momentum != 0.0:
            momentum = zeros(weight.shape, weight.context, dtype=weight.dtype, stype=stype)
        return momentum

    def _update_impl(self, index, weight, grad, state, multi_precision=False):
        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))
        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)

        # ---------- MAS begin ----------
        # index : layer's name 
        if index in self.reg_params:
            reg_param = self.reg_params[index]
            weight_dif = weight - reg_param['init_val']
            regulizer = weight_dif * (2 * self.reg_lambda * reg_param['omega'])
            grad = grad + regulizer
        # ----------- MAS end -----------

        kwargs = {'rescale_grad': self.rescale_grad}
        if self.momentum > 0:
            kwargs['momentum'] = self.momentum
        if self.clip_gradient:
            kwargs['clip_gradient'] = self.clip_gradient

        if not multi_precision:
            if state is not None:
                sgd_mom_update(weight, grad, state, out=weight,
                               lr=lr, wd=wd, **kwargs)
            else:
                sgd_update(weight, grad, out=weight,
                           lr=lr, wd=wd, **kwargs)
        else:
            if state[0] is not None:
                mp_sgd_mom_update(weight, grad, state[0], state[1], out=weight,
                                  lr=lr, wd=wd, **kwargs)
            else:
                mp_sgd_update(weight, grad, state[1], out=weight,
                              lr=lr, wd=wd, **kwargs)

    def update(self, index, weight, grad, state):
        self._update_impl(index, weight, grad, state, multi_precision=False)

    def update_multi_precision(self, index, weight, grad, state):
        use_multi_precision = self.multi_precision and weight.dtype == numpy.float16
        self._update_impl(index, weight, grad, state,
                          multi_precision=use_multi_precision)