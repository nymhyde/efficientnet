#!/usr/local/bin/python3

"""
model.py :: Model and module class for EfficientNet.
            They are built to mirror those in the official TF implementation.
"""


# imports
import torch
from torch import nn
from torch.nn import functional as F
from .util import ( 
        round_filters, 
        round_repeats, 
        drop_connect, 
        get_same_padding_conv2d, 
        get_model_params, 
        efficientnet_params, 
        Swish, 
        MemoryEfficientSwish, 
        calculate_output_image_size 
                )





VALID_MODELS = (
        'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4',
        'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7', 'efficientnet-b8' 
               )


class MBConvBlock(nn.Module):
    '''
    Mobile Inverted Residual Bottleneck Block.

    Args :: 
        block_args (namedtuple) : BlockArgs, defined in util.py
        global_params (namedtuple) : GlobalParam, defined in util.py
        image_size (tuple or list) : [image_height, image_width]

    '''

    def __init__(self, block_args, global_params, image_size=None):
        super().__init__()
        self._block_args = block_args
        self._bn_nom = 1 - global_params.batch_norm_momentum         # pytorch tf difference
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0< self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip           # whether to use skip connection and drop connect


        # Expansion phase (Inverted Bottleneck)
        inp = self._block_args.input_filters        # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio    # number of output channels
        if self._block_args.expand_ratio != 1:
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_nom, eps=self._bn_eps)
            # image_size = calculate_output_image_size(image_size, 1) <-- this wouldn't modify image_size


        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride

        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._depthwise_conv = Conv2d(
                in_channels=oup, out_channels=oup, groups=oup,
                kernel_size=k, stride=s, bias=False)

        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_nom, eps=self._bn_eps)
        image_size = calculate_output_image_size(image_size, s)


        # Squeeze and Excitation layer, if desired
        if self.has_se:
            Conv2d = get_same_padding_conv2d(image_size=(1,1))
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)


        # Pointwise convolution phase
        final_oup = self._block_args.output_filters
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_nom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()



    def forward(self, inputs, drop_connect_rate=None):
        '''
        MBConvBlock's forward function ::

        Args ::
                inputs (tensor) : Input tensor
                drop_connect_rate (bool) : Drop connect rate (float, between 0 and 1).

        Returns ::
                Output of this block after processing.
        '''

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x,1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x

        # pointwise convolution
        x = self._project_conv(x)
        x = self._bn2(x)

        # skip connection and drop connection
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            # the combination of skip connection and drop connect brings about stochastic depth
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection

        return x

    def set_swish(self, memory_efficient=True):
        '''
        Sets swish function as memory efficient (for training) or standard (for export).

        Args ::
                memory_efficient (bool) : whether to use memory-efficient version of swish.
        '''

        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()




class EfficientNet(nn.Module):
    '''
    EfficientNet model.

    Most easily loaded with the .from_name or .from_pretrained methods.

    Args ::
            blocks_args (list[namedtuple]) : A list of BlockArgs to construct blocks.
            global_params (namedtuple) : A set of GlobalParms shared between blocks.

    '''


    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        # Batch norm parameters
        bn_nom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # get stem static or dynamic convolution depending on image size
        image_size = global_params.image_size
        Conv2d = get_same_padding_conv2d(image_size=image_size)

        # stem
        in_channels = 1 # rgb
        out_channels = round_filters(32, self._global_params)   # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_nom, eps=bn_eps)
        image_size = calculate_output_image_size(image_size, 2)

        # Build Blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters = round_filters(block_args.input_filters, self._global_params),
                output_filters = round_filters(block_args.output_filters, self._global_params),
                num_repeat = round_repeats(block_args.num_repeat, self._global_params)
                                            )

            # the first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params, image_size=image_size))
            image_size = calculate_output_image_size(image_size, block_args.stride)

            if block_args.num_repeat > 1:       # modify block_args to keep same output size
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params, image_size=image_size))
                # image_size = calculate_output_image_size(image_size, block_args.stride)   
                # print(image_size)

        # Head
        in_channels = block_args.output_filters         # output of final block
        out_channels = round_filters(1280, self._global_params)
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_nom, eps=bn_eps)

        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        if self._global_params.include_top:
            self._dropout = nn.Dropout(self._global_params.dropout_rate)
            self._fc = nn.Linear(out_channels, self._global_params.num_classes)

        # set activation to memory efficient swish by default
        self._swish = MemoryEfficientSwish()



    def set_swish(self, memory_efficient=True):
        '''
        Sets swish function as memory efficient (for training) or standard (for export).

        Args ::
                memory_efficient (bool) : whether to use memory-efficient version of swish.
        '''

        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()

        for block in self._blocks:
            block.set_swish(memory_efficient)



    def extract_endpoints(self, inputs):
        '''
        use convolution layer to extract features
        from reduction levels i in [1, 2, 3, 4, 5].

        Args ::
                inputs (tensor) : input tensor

        Returns ::
                dictionary of last intermediate features
                with reduction levels i in [1, 2, 3, 4, 5, 6].
        '''


        endpoints = dict()

        # stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        prev_x = x

        # blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)     # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            elif idx == len(self._blocks) - 1:
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = x
            prev_x = x

        # head
        x = self._swish(self._bn1(self._conv_head(x)))
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x


        return endpoints


    def extract_features(self, inputs:torch.Tensor)->torch.Tensor:
        '''
        Use convolution layer to extract featuers

        Args ::
                inputs (tensor) : input tensor

        Returns ::
                output of the final convolution layer in the efficientnet model.
        '''

        # stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)     # scale drop connect rate
            x = block(x, drop_connect_rate=drop_connect_rate)

        # head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x


    def forward(self, inputs):
        '''
        EfficientNet's model forward function.
        Calls extract_features to extract features, applies final linear layer and returns logits.

        Args ::
                inputs (tensor) : Input Tensor.

        Returns ::
                output of this model after processing.
        '''

        # Convolution layers
        x = self.extract_features(inputs)
        # pooling and final linear layer
        x = self._avg_pooling(x)
        if self._global_params.include_top:
            x = x.flatten(start_dim=1)
            x = self._dropout(x)
            x = self._fc(x)

        return x



    @classmethod
    def from_name(cls, model_name, in_channels=3, **kwargs):
        '''
        Create an efficientnet model according to name.

        Args ::
                model_name (str) : valid model name for efficientnet.
                in_channels (int) : Input data's channel number (RGB)
                kwargs (other optiona params) : 
                    'width_coefficient', 'detph_coefficient',
                    'image_size', 'dropout_rate',
                    'num_classes', 'batch_norm_momentum',
                    'batch_norm_epsilon', 'drop_connect_rate',
                    'depth_divisor', 'min_depth'

        Returns ::
                efficientnet model.
        '''

        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, kwargs)
        model = cls(blocks_args, global_params)

        model._change_in_channels(in_channels)

        return model


    @classmethod
    def get_image_size(cls, model_name):
        '''
        Get the input image size for a given efficientnet model.

        Args ::
                model_name (str) : Name for efficientnet.

        Returns ::
                input image size (resolution).
        '''

        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)

        return res


    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        '''
        Validates model name

        Args ::
                model_name (str) : Name for efficientnet.

        Returns ::
                bool : whether the name is valid or not.
        '''

        if model_name not in VALID_MODELS:
            raise ValueError('model_name should be one of :: ' + ', '.join(VALID_MODELS))



    def _change_in_channels(self, in_channels):
        '''
        Adjust model's first convolution layer to in_channels, if in_channels are not 3.

        Args ::
                in_channels (int) : input data's channel number.
        '''

        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(image_size=self._global_params.image_size)
            out_channels = round_filters(32, self._global_params)
            self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)

