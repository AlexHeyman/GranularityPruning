import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.resnet import conv3x3, conv1x1
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from torch.nn.common_types import _size_2_t

from torch import Tensor
from typing import Type, Any, Callable, Union, Iterable, List, Optional


def prod(iterable):
  product = 1
  for number in iterable:
    product *= number
  return product


def get_total_masks_size(mask_shapes):
  total = 0

  for element in mask_shapes:
    if isinstance(element, list):
      total += get_total_masks_size(element)
    else: # element is assumed to be a tuple defining a mask tensor shape
      total += prod(element)
  
  return total


def make_mask_tensors(mask_shapes, masks_array, masks_array_index, masks_flat):
  masks = []

  for element in mask_shapes:
    if isinstance(element, list):
      masks_element, masks_array_index = make_mask_tensors(
        element, masks_array, masks_array_index, masks_flat)
      masks.append(masks_element)
    else: # element is assumed to be a tuple defining a mask tensor shape
      new_index = masks_array_index + prod(element)
      new_mask_tensor = masks_array[masks_array_index:new_index].view(element)
      masks_array_index = new_index
      new_mask = nn.Parameter(new_mask_tensor, requires_grad=False)
      masks.append(new_mask)
      masks_flat.append(new_mask)
  
  return masks, masks_array_index


class MaskedModule(nn.Module):

  def __init__(self, mask_shapes: List = None, **kwargs) -> None:
    super().__init__(**kwargs)
    total_masks_size = get_total_masks_size(mask_shapes)
    self.masks_array = torch.ones(total_masks_size)
    self.masks_flat = []
    self.masks, _ = make_mask_tensors(mask_shapes, self.masks_array,
                                      0, self.masks_flat)
    for i in range(len(self.masks_flat)):
      self.register_parameter('mask%d' % i, self.masks_flat[i])

  def sync_masks(self, device) -> None:
    self.masks_array = self.masks_array.to(device)
    masks_array_index = 0
    for mask in self.masks_flat:
      new_index = masks_array_index + torch.numel(mask)
      mask.data = self.masks_array[masks_array_index:new_index].view(mask.size())
      masks_array_index = new_index

  def non_mask_parameters(self) -> Iterable[nn.Parameter]:
    mask_params = set(self.masks_flat)
    return [param for param in self.parameters() if param not in mask_params]


class BlockMaskedModule(MaskedModule):
  
  def __init__(self, mask_shapes: List = None, **kwargs) -> None:
    super().__init__(mask_shapes=mask_shapes, **kwargs)
    self.blocks = []


class ExposedMaskedModule(MaskedModule):

  def __init__(self, mask_shapes: List = None, **kwargs) -> None:
    super().__init__(mask_shapes=mask_shapes, **kwargs)
    self.exposed_tensors = []

  def reset_exposed_tensors(self) -> None:
    self.exposed_tensors.clear()


class WeightMaskedModule(MaskedModule):

  def __init__(self, mask_shapes: List = None, **kwargs) -> None:
    super().__init__(mask_shapes=mask_shapes, **kwargs)
    # List of this model's masked weight tensors with the same structure as
    # self.masks_flat, for ease of finding the mask element corresponding to
    # a weight and vice versa. Subclasses of WeightMaskedModule should fill out
    # this list in their initializers.
    self.masked_weight_tensors = []


class SimpleMNISTNetNodeMasked(BlockMaskedModule, ExposedMaskedModule):
  
  def __init__(self, layer_widths, nonlinearities):
    mask_shapes = [(width,) for width in layer_widths]
    super().__init__(mask_shapes)
    complete_widths = [784] + layer_widths + [10]
    
    self.fcs = nn.ModuleList(
      [nn.Linear(complete_widths[i], complete_widths[i + 1])
       for i in range(len(complete_widths) - 1)])
    
    for i in range(len(self.fcs) - 1):
      self.blocks.append(self.fcs[i].weight)
    
    self.nonlinearities = nn.ModuleList(nonlinearities)

  def forward(self, x):
    self.reset_exposed_tensors()
    
    x = x.view(-1, 784)
    
    for i in range(len(self.fcs) - 1):
      x = self.nonlinearities[i](self.fcs[i](x) * self.masks[i])
      self.exposed_tensors.append(x)
    
    return self.fcs[-1](x)


class LinearWeightMasked(nn.Module):

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor
    bias: Union[Tensor, None]
    mask: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, mask=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.mask = mask
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight * self.mask, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class SimpleMNISTNetWeightMasked(WeightMaskedModule, ExposedMaskedModule):
  
  def __init__(self, layer_widths, nonlinearities):
    complete_widths = [784] + layer_widths + [10]
    mask_shapes = [(complete_widths[i + 1], complete_widths[i])
                   for i in range(len(complete_widths) - 1)]
    super().__init__(mask_shapes)
    
    self.fcs = nn.ModuleList(
      [LinearWeightMasked(complete_widths[i], complete_widths[i + 1],
                          mask=self.masks[i])
       for i in range(len(complete_widths) - 1)])
    for layer in self.fcs:
      self.masked_weight_tensors.append(layer.weight)
    
    self.nonlinearities = nn.ModuleList(nonlinearities)

  def forward(self, x):
    self.reset_exposed_tensors()
    
    x = x.view(-1, 784)
    
    for i in range(len(self.fcs) - 1):
      x = self.nonlinearities[i](self.fcs[i](x))
      self.exposed_tensors.append(x)
    
    return self.fcs[-1](x)


# Now supports groups != 1, base_width != 64, and dilation != 1
class BasicBlockChannelMasked(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        masks: List = None,
        blocks: List = None,
        exposed_tensors: List = None
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.masks = masks
        self.exposed_tensors = exposed_tensors
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, width, stride, groups, dilation)
        blocks.append(self.conv1.weight)
        self.bn1 = norm_layer(width)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(width, planes)
        blocks.append(self.conv2.weight)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = out * self.masks[0]
        self.exposed_tensors.append(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        out = out * self.masks[1]
        self.exposed_tensors.append(out)

        return out


# Currently doesn't have masking stuff implemented
class BottleneckChannelMasked(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetChannelMasked(BlockMaskedModule, ExposedMaskedModule):

    def __init__(
        self,
        block: Type[Union[BasicBlockChannelMasked, BottleneckChannelMasked]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        inplanes: int = 64,
        initial_kernel_size: int = 7,
        initial_stride: int = 2,
        do_maxpool: bool = True,
        mask_shapes: List = None
    ) -> None:
        super().__init__(mask_shapes)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = inplanes
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False for _ in range(len(layers) - 1)]
        if len(replace_stride_with_dilation) != len(layers) - 1:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a tuple with length one less than layers, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=initial_kernel_size,
          stride=initial_stride, padding=int((initial_kernel_size-1)/2), bias=False)
        self.blocks.append(self.conv1.weight)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        if do_maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.maxpool = None
        
        self.layers = nn.ModuleList()
        planes = self.inplanes
        self.layers.append(self._make_layer(block, planes, layers[0],
          masks=self.masks[1]))
        for i in range(1, len(layers)):
          planes *= 2
          self.layers.append(self._make_layer(block, planes, layers[i],
            stride=2, dilate=replace_stride_with_dilation[i-1],
            masks=self.masks[i+1]))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(planes * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleneckChannelMasked):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlockChannelMasked):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlockChannelMasked, BottleneckChannelMasked]],
                    planes: int, blocks: int, stride: int = 1, dilate: bool = False,
                    masks: List = None) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        new_block = block(self.inplanes, planes, stride, downsample,
                          self.groups, self.base_width, previous_dilation,
                          norm_layer, masks=masks[0], blocks=self.blocks,
                          exposed_tensors=self.exposed_tensors)
        layers.append(new_block)
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            new_block = block(self.inplanes, planes, groups=self.groups,
                              base_width=self.base_width, dilation=self.dilation,
                              norm_layer=norm_layer, masks=masks[i], blocks=self.blocks,
                              exposed_tensors=self.exposed_tensors)
            layers.append(new_block)

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        self.reset_exposed_tensors()
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = x * self.masks[0]
        self.exposed_tensors.append(x)
        
        if self.maxpool is not None:
          x = self.maxpool(x)
        
        for layer in self.layers:
          x = layer(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def resnet18_channel_masked(inplanes=64, **kwargs):
  ip = inplanes
  # Shaping the masks to have size 1 in the non-channel dimensions allows us to
  # exploit PyTorch's broadcasting semantics, applied to elementwise
  # multiplication, to mask out entire channels at once.
  # https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics
  mask_shapes = [(ip, 1, 1),
                 [[(ip, 1, 1), (ip, 1, 1)], [(ip, 1, 1), (ip, 1, 1)]],
                 [[(2*ip, 1, 1), (2*ip, 1, 1)], [(2*ip, 1, 1), (2*ip, 1, 1)]],
                 [[(4*ip, 1, 1), (4*ip, 1, 1)], [(4*ip, 1, 1), (4*ip, 1, 1)]],
                 [[(8*ip, 1, 1), (8*ip, 1, 1)], [(8*ip, 1, 1), (8*ip, 1, 1)]]
                 ]
  return ResNetChannelMasked(BasicBlockChannelMasked, [2, 2, 2, 2], inplanes=ip,
    initial_kernel_size=7, initial_stride=2, do_maxpool=True, mask_shapes=mask_shapes, **kwargs)


def resnet20_channel_masked(inplanes=16, **kwargs):
  ip = inplanes
  mask_shapes = [(ip, 1, 1),
                 [[(ip, 1, 1), (ip, 1, 1)],
                  [(ip, 1, 1), (ip, 1, 1)],
                  [(ip, 1, 1), (ip, 1, 1)]],
                 [[(2*ip, 1, 1), (2*ip, 1, 1)],
                  [(2*ip, 1, 1), (2*ip, 1, 1)],
                  [(2*ip, 1, 1), (2*ip, 1, 1)]],
                 [[(4*ip, 1, 1), (4*ip, 1, 1)],
                  [(4*ip, 1, 1), (4*ip, 1, 1)],
                  [(4*ip, 1, 1), (4*ip, 1, 1)]]
                 ]
  return ResNetChannelMasked(BasicBlockChannelMasked, [3, 3, 3], inplanes=ip,
    initial_kernel_size=3, initial_stride=1, do_maxpool=False,
    mask_shapes=mask_shapes, **kwargs)


class Conv2dWeightTensorMasked(_ConvNd):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None,
        mask=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super(Conv2dWeightTensorMasked, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode, **factory_kwargs)
        self.mask = mask
    
    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight * self.mask, self.bias)


def conv3x3_weight_tensor_masked(in_planes: int, out_planes: int,
    stride: int = 1, groups: int = 1, dilation: int = 1,
    mask = None) -> Conv2dWeightTensorMasked:
  return Conv2dWeightTensorMasked(in_planes, out_planes, kernel_size=3,
    stride=stride, padding=dilation, groups=groups, bias=False,
    dilation=dilation, mask=mask)


# Now supports groups != 1, base_width != 64, and dilation != 1
class BasicBlockWeightTensorMasked(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        masks: List = None,
        masked_weight_tensors: List = None,
        blocks: List = None,
        exposed_tensors: List = None
    ) -> None:
        # masked_weight_tensors is a list maintained by the overall model that
        # this module appends its masked weight tensors to
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.exposed_tensors = exposed_tensors
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3_weight_tensor_masked(
          inplanes, width, stride, groups, dilation, mask=masks[0])
        masked_weight_tensors.append(self.conv1.weight)
        blocks.append(self.conv1.weight.view(-1, 3, 3))
        self.bn1 = norm_layer(width)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_weight_tensor_masked(width, planes, mask=masks[1])
        masked_weight_tensors.append(self.conv2.weight)
        blocks.append(self.conv2.weight.view(-1, 3, 3))
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        self.exposed_tensors.append(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        self.exposed_tensors.append(out)

        return out


# Currently doesn't have masking stuff implemented
class BottleneckWeightTensorMasked(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetWeightTensorMasked(ExposedMaskedModule):

    def __init__(
        self,
        block: Type[Union[BasicBlockWeightTensorMasked,
                          BottleneckWeightTensorMasked]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        inplanes: int = 64,
        initial_kernel_size: int = 7,
        initial_stride: int = 2,
        do_maxpool: bool = True,
        mask_shapes: List = None,
        **kwargs
    ) -> None:
        super().__init__(mask_shapes=mask_shapes, **kwargs)

        self.masked_weight_tensors = []
        self.blocks = []
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = inplanes
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False for _ in range(len(layers) - 1)]
        if len(replace_stride_with_dilation) != len(layers) - 1:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a tuple with length one less than layers, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = Conv2dWeightTensorMasked(3, self.inplanes,
          kernel_size=initial_kernel_size, stride=initial_stride,
          padding=int((initial_kernel_size-1)/2), bias=False, mask=self.masks[0])
        self.masked_weight_tensors.append(self.conv1.weight)
        self.blocks.append(self.conv1.weight.view(
          -1, initial_kernel_size, initial_kernel_size))
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        if do_maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.maxpool = None
        
        self.layers = nn.ModuleList()
        planes = self.inplanes
        self.layers.append(self._make_layer(block, planes, layers[0], masks=self.masks[1]))
        for i in range(1, len(layers)):
          planes *= 2
          self.layers.append(self._make_layer(block, planes, layers[i], stride=2,
            dilate=replace_stride_with_dilation[i-1], masks=self.masks[i+1]))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(planes * block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, Conv2dWeightTensorMasked)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleneckWeightTensorMasked):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlockWeightTensorMasked):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlockWeightTensorMasked,
                                            BottleneckWeightTensorMasked]],
                    planes: int, blocks: int, stride: int = 1,
                    dilate: bool = False, masks: List = None) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        new_block = block(self.inplanes, planes, stride, downsample,
          self.groups, self.base_width, previous_dilation, norm_layer,
          masks=masks[0], masked_weight_tensors=self.masked_weight_tensors,
          blocks=self.blocks, exposed_tensors=self.exposed_tensors)
        layers.append(new_block)
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            new_block = block(self.inplanes, planes, groups=self.groups,
              base_width=self.base_width, dilation=self.dilation,
              norm_layer=norm_layer, masks=masks[i],
              masked_weight_tensors=self.masked_weight_tensors,
              blocks=self.blocks, exposed_tensors=self.exposed_tensors)
            layers.append(new_block)

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        self.reset_exposed_tensors()
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        self.exposed_tensors.append(x)
        
        if self.maxpool is not None:
          x = self.maxpool(x)

        for layer in self.layers:
          x = layer(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


# WeightMaskedModule needs to be listed second here so its initializer is
# called first
class ResNetWeightMasked(ResNetWeightTensorMasked, WeightMaskedModule):

  def __init__(
    self,
    block: Type[Union[BasicBlockWeightTensorMasked,
                      BottleneckWeightTensorMasked]],
    layers: List[int],
    num_classes: int = 1000,
    zero_init_residual: bool = False,
    groups: int = 1,
    width_per_group: int = 64,
    replace_stride_with_dilation: Optional[List[bool]] = None,
    norm_layer: Optional[Callable[..., nn.Module]] = None,
    inplanes: int = 64,
    initial_kernel_size: int = 7,
    initial_stride: int = 2,
    do_maxpool: bool = True,
    mask_shapes: List = None
    ) -> None:
    super().__init__(block=block, layers=layers, num_classes=num_classes,
      zero_init_residual=zero_init_residual, groups=groups, width_per_group=\
      width_per_group, replace_stride_with_dilation=replace_stride_with_dilation,
      norm_layer=norm_layer, inplanes=inplanes, initial_kernel_size=\
      initial_kernel_size, initial_stride=initial_stride, do_maxpool=do_maxpool,
      mask_shapes=mask_shapes)


# BlockMaskedModule needs to be listed second here so its initializer is called
# first
class ResNetKernelMasked(ResNetWeightTensorMasked, BlockMaskedModule):

  def __init__(
    self,
    block: Type[Union[BasicBlockWeightTensorMasked,
                      BottleneckWeightTensorMasked]],
    layers: List[int],
    num_classes: int = 1000,
    zero_init_residual: bool = False,
    groups: int = 1,
    width_per_group: int = 64,
    replace_stride_with_dilation: Optional[List[bool]] = None,
    norm_layer: Optional[Callable[..., nn.Module]] = None,
    inplanes: int = 64,
    initial_kernel_size: int = 7,
    initial_stride: int = 2,
    do_maxpool: bool = True,
    mask_shapes: List = None
    ) -> None:
    super().__init__(block=block, layers=layers, num_classes=num_classes,
      zero_init_residual=zero_init_residual, groups=groups, width_per_group=\
      width_per_group, replace_stride_with_dilation=replace_stride_with_dilation,
      norm_layer=norm_layer, inplanes=inplanes, initial_kernel_size=\
      initial_kernel_size, initial_stride=initial_stride, do_maxpool=do_maxpool,
      mask_shapes=mask_shapes)


def resnet18_weight_masked(inplanes=64, **kwargs):
  ip = inplanes
  mask_shapes = [(ip, 3, 7, 7),
                 [[(ip, ip, 3, 3), (ip, ip, 3, 3)],
                  [(ip, ip, 3, 3), (ip, ip, 3, 3)]],
                 [[(2*ip, ip, 3, 3), (2*ip, 2*ip, 3, 3)],
                  [(2*ip, 2*ip, 3, 3), (2*ip, 2*ip, 3, 3)]],
                 [[(4*ip, 2*ip, 3, 3), (4*ip, 4*ip, 3, 3)],
                  [(4*ip, 4*ip, 3, 3), (4*ip, 4*ip, 3, 3)]],
                 [[(8*ip, 4*ip, 3, 3), (8*ip, 8*ip, 3, 3)],
                  [(8*ip, 8*ip, 3, 3), (8*ip, 8*ip, 3, 3)]]
                 ]
  return ResNetWeightMasked(BasicBlockWeightTensorMasked, [2, 2, 2, 2],
    initial_kernel_size=7, initial_stride=2, do_maxpool=True, inplanes=ip,
    mask_shapes=mask_shapes, **kwargs)


def resnet20_weight_masked(inplanes=16, **kwargs):
  ip = inplanes
  mask_shapes = [(ip, 3, 3, 3),
                 [[(ip, ip, 3, 3), (ip, ip, 3, 3)],
                  [(ip, ip, 3, 3), (ip, ip, 3, 3)],
                  [(ip, ip, 3, 3), (ip, ip, 3, 3)]],
                 [[(2*ip, ip, 3, 3), (2*ip, 2*ip, 3, 3)],
                  [(2*ip, 2*ip, 3, 3), (2*ip, 2*ip, 3, 3)],
                  [(2*ip, 2*ip, 3, 3), (2*ip, 2*ip, 3, 3)]],
                 [[(4*ip, 2*ip, 3, 3), (4*ip, 4*ip, 3, 3)],
                  [(4*ip, 4*ip, 3, 3), (4*ip, 4*ip, 3, 3)],
                  [(4*ip, 4*ip, 3, 3), (4*ip, 4*ip, 3, 3)]]
                 ]
  return ResNetWeightMasked(BasicBlockWeightTensorMasked, [3, 3, 3],
    initial_kernel_size=3, initial_stride=1, do_maxpool=False, inplanes=ip,
    mask_shapes=mask_shapes, **kwargs)


def resnet20_kernel_masked(inplanes=16, **kwargs):
  ip = inplanes
  mask_shapes = [(ip, 3, 1, 1),
                 [[(ip, ip, 1, 1), (ip, ip, 1, 1)],
                  [(ip, ip, 1, 1), (ip, ip, 1, 1)],
                  [(ip, ip, 1, 1), (ip, ip, 1, 1)]],
                 [[(2*ip, ip, 1, 1), (2*ip, 2*ip, 1, 1)],
                  [(2*ip, 2*ip, 1, 1), (2*ip, 2*ip, 1, 1)],
                  [(2*ip, 2*ip, 1, 1), (2*ip, 2*ip, 1, 1)]],
                 [[(4*ip, 2*ip, 1, 1), (4*ip, 4*ip, 1, 1)],
                  [(4*ip, 4*ip, 1, 1), (4*ip, 4*ip, 1, 1)],
                  [(4*ip, 4*ip, 1, 1), (4*ip, 4*ip, 1, 1)]]
                 ]
  return ResNetKernelMasked(BasicBlockWeightTensorMasked, [3, 3, 3],
    initial_kernel_size=3, initial_stride=1, do_maxpool=False, inplanes=ip,
    mask_shapes=mask_shapes, **kwargs)
