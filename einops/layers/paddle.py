from typing import Optional, Dict, cast

import paddle

from . import RearrangeMixin, ReduceMixin
from ._einmix import _EinmixMixin
from .._paddle_specific import apply_for_scriptable_paddle

__author__ = 'HydrogenSulfate'


class Rearrange(RearrangeMixin, paddle.nn.Layer):
    def forward(self, input):
        return apply_for_scriptable_paddle(self._recipe, input, reduction_type='rearrange')

    def _apply_recipe(self, x):
        # overriding parent method to prevent it's scripting
        pass


class Reduce(ReduceMixin, paddle.nn.Layer):
    def forward(self, input):
        return apply_for_scriptable_paddle(self._recipe, input, reduction_type=self.reduction)

    def _apply_recipe(self, x):
        # overriding parent method to prevent it's scripting
        pass


class EinMix(_EinmixMixin, paddle.nn.Layer):
    def _create_parameters(self, weight_shape, weight_bound, bias_shape, bias_bound):
        self.weight = paddle.create_parameter(weight_shape,
            "float32",
            default_initializer=paddle.nn.initializer.Uniform(-weight_bound, weight_bound))
        if bias_shape is not None:
            self.bias = paddle.create_parameter(bias_shape,
                "float32",
                default_initializer=paddle.nn.initializer.Uniform(-bias_bound, bias_bound))
        else:
            self.bias = None

    def _create_rearrange_layers(self,
                                 pre_reshape_pattern: Optional[str],
                                 pre_reshape_lengths: Optional[Dict],
                                 post_reshape_pattern: Optional[str],
                                 post_reshape_lengths: Optional[Dict],
                                 ):
        self.pre_rearrange = None
        if pre_reshape_pattern is not None:
            self.pre_rearrange = Rearrange(pre_reshape_pattern, **cast(dict, pre_reshape_lengths))

        self.post_rearrange = None
        if post_reshape_pattern is not None:
            self.post_rearrange = Rearrange(post_reshape_pattern, **cast(dict, post_reshape_lengths))

    def forward(self, input):
        if self.pre_rearrange is not None:
            input = self.pre_rearrange(input)
        result = paddle.einsum(self.einsum_pattern, input, self.weight)
        if self.bias is not None:
            result += self.bias
        if self.post_rearrange is not None:
            result = self.post_rearrange(result)
        return result
