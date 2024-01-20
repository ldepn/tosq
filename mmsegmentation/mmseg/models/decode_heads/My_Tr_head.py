# -*- coding: UTF-8 -*-  
# @Time : 2023/4/11 22:23
import math
import warnings

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule, Conv2d, build_norm_layer, build_activation_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import MultiheadAttention, FFN
from mmengine import deprecated_api_warning

from mmengine.model import BaseModule, ModuleList, Sequential, trunc_normal_init, constant_init, normal_init

from mmseg.models.backbones.mit import MixFFN
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from ..utils import resize, nlc_to_nchw, nchw_to_nlc


@MODELS.register_module()
class MyMultiheadAttention(BaseModule):
    """A wrapper for ``torch.nn.MultiheadAttention``.

    This module implements MultiheadAttention with identity connection,
    and positional encoding  is also passed as input.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): When it is True,  Key, Query and Value are shape of
            (batch, n, embed_dim), otherwise (n, batch, embed_dim).
             Default to False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 init_cfg=None,
                 batch_first=False,
                 **kwargs):
        super().__init__(init_cfg)
        if 'dropout' in kwargs:
            warnings.warn(
                'The arguments `dropout` in MultiheadAttention '
                'has been deprecated, now you can separately '
                'set `attn_drop`(float), proj_drop(float), '
                'and `dropout_layer`(dict) ', DeprecationWarning)
            attn_drop = kwargs['dropout']
            dropout_layer['drop_prob'] = kwargs.pop('dropout')

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first

        self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop,
                                          **kwargs)

        self.dropout = nn.Dropout(proj_drop)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()

    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MultiheadAttention')
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `MultiheadAttention`.

        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.

        Args:
            query (Tensor): The input query with shape [num_queries, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
                If None, the ``query`` will be used. Defaults to None.
            value (Tensor): The value tensor with same shape as `key`.
                Same in `nn.MultiheadAttention.forward`. Defaults to None.
                If None, the `key` will be used.
            identity (Tensor): This tensor, with the same shape as x,
                will be used for the identity link.
                If None, `x` will be used. Defaults to None.
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `x`. If not None, it will
                be added to `x` before forward function. Defaults to None.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`. Defaults to None. If not None, it will
                be added to `key` before forward function. If None, and
                `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`. Defaults to None.
            attn_mask (Tensor): ByteTensor mask with shape [num_queries,
                num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_keys].
                Defaults to None.

        Returns:
            Tensor: forwarded results with shape
            [num_queries, bs, embed_dims]
            if self.batch_first is False, else
            [bs, num_queries embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f'position encoding of key is'
                                  f'missing in {self.__class__.__name__}.')
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # Because the dataflow('key', 'query', 'value') of
        # ``torch.nn.MultiheadAttention`` is (num_query, batch,
        # embed_dims), We should adjust the shape of dataflow from
        # batch_first (batch, num_query, embed_dims) to num_query_first
        # (num_query ,batch, embed_dims), and recover ``attn_output``
        # from num_query_first to batch_first.
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        out, attn_map = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False)

        if self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.dropout_layer(self.dropout(out)), attn_map


class QueryParsingModule(BaseModule):
    def __init__(self,
                 encoder_in_channels,
                 embed_dims,
                 num_heads,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 mlp_ratio=4,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 batch_first=True,
                 qkv_bias=True):
        super().__init__()

        # The ret[0] of build_norm_layer is norm name.
        self.norm0 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.LinearProjection = Conv2d(
            in_channels=encoder_in_channels,
            out_channels=embed_dims,
            kernel_size=1,
            stride=1)

        # The ret[0] of build_norm_layer is norm name.
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.attn = MyMultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            init_cfg=init_cfg,
            batch_first=batch_first,
            bias=qkv_bias)

        # The ret[0] of build_norm_layer is norm name.
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=embed_dims * mlp_ratio,
            act_cfg=act_cfg,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate))

    def forward(self, x_q, x_kv):
        x_kv = self.norm0(nchw_to_nlc(self.LinearProjection(x_kv)))
        x_q, attn_map = self.attn(query=x_q, key=x_kv, value=x_kv, identity=x_q)
        x_q = self.norm1(x_q)
        x_q = self.norm2(self.ffn(x_q, identity=x_q))  # bs, h*w, c
        return x_q, attn_map


@MODELS.register_module()
class SegTrueHead(BaseDecodeHead):
    def __init__(self,
                 query_embed_dims=256,
                 num_stages=4,
                 num_heads=[8, 8, 8, 8],
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 mlp_ratio=4,
                 interpolate_mode='bilinear',
                 **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        self.query_embed_dims = query_embed_dims
        self.num_stages = num_stages
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.mlp_ratio = mlp_ratio

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)
        assert num_inputs == len(self.in_index)

        self.num_query = self.num_classes
        self.cls_embed = nn.Parameter(torch.randn((1, self.num_query, query_embed_dims)))
        nn.init.trunc_normal_(self.cls_embed, std=.02)

        self.decoder = ModuleList()
        for i in range(num_stages):
            self.decoder.append(
                QueryParsingModule(
                    encoder_in_channels=self.in_channels[i],
                    embed_dims=query_embed_dims,
                    num_heads=num_heads[i],
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=drop_path_rate,
                    mlp_ratio=mlp_ratio,
                    act_cfg=self.act_cfg,
                    norm_cfg=self.norm_cfg,
                    qkv_bias=qkv_bias))

        self.feat_enc_conv = ConvModule(
            in_channels=self.in_channels[0],
            out_channels=self.channels,
            # kernel_size=1,
            kernel_size=3,
            padding=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU', inplace=True))

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.num_heads[i],
                    out_channels=self.channels,
                    # kernel_size=1,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    norm_cfg=dict(type='BN', requires_grad=True),
                    act_cfg=dict(type='ReLU', inplace=True)))

        self.fusion_attn = DepthwiseSeparableConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            # kernel_size=1,
            kernel_size=3,
            padding=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU', inplace=True))

        self.fusion_attn_feat = ConvModule(
            in_channels=2 * self.channels,
            out_channels=self.channels,
            # kernel_size=1,
            kernel_size=3,
            padding=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU', inplace=True))

        self.pred = Sequential(
            self.dropout,
            Conv2d(
                in_channels=self.channels,
                out_channels=1,
                kernel_size=1,
                stride=1))

    def init_weights(self):
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super().init_weights()

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        query = self.cls_embed.expand(inputs[0].shape[0], -1, -1)
        feat_enc = inputs[0]  # bs, c, h, w
        bs, _, *size = feat_enc.shape
        outs = []
        for idx in range(len(inputs)):
            x_kv = inputs[self.num_stages - 1 - idx]
            qpm_i = self.decoder[self.num_stages - 1 - idx]
            query, attn_map = qpm_i(query, x_kv)
            bs, nhead, nclass, hw = attn_map.shape
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(attn_map.transpose(1, 2).contiguous().reshape(bs * nclass, nhead,
                                                                             inputs[self.num_stages - 1 - idx].shape[2],
                                                                             inputs[self.num_stages - 1 - idx].shape[
                                                                                 3])),
                    size=size,
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_attn(torch.cat(outs, dim=1))  # bs*nclass, c, h, w
        feat_enc = self.feat_enc_conv(feat_enc)  # bs, c, h, w

        out = self.fusion_attn_feat(torch.cat([_expand(feat_enc, self.num_classes), out], dim=1))
        out = self.pred(out).reshape(bs, self.num_classes, size[0], size[1])  # bs, nclass, h, w

        return out


def _expand(x, nclass):  # (bs, c, h, w) to (bs * nclass, c, h, w)
    return x.unsqueeze(1).repeat(1, nclass, 1, 1, 1).flatten(0, 1)
