import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
from itertools import repeat
import collections.abc
import math
from functools import partial
from timm.models.layers import DropPath
from timm.models.registry import register_model
from typing import List # Import List for type hinting

def parse(x, n):
    if isinstance(x, collections.abc.Iterable):
        if len(x) == 1:
            return list(repeat(x[0], n))
        elif len(x) == n:
            return x
        else:
            raise ValueError('length of x should be 1 or n')
    else:
        return list(repeat(x, n))


class Attention(nn.Module):
    def __init__(self, in_planes, reduction, num_static_cell, num_local_mixture, norm_layer=nn.BatchNorm1d, # norm_layer default will be overridden
                 cell_num_ratio=1.0, nonlocal_basis_ratio=1.0, start_cell_idx=None):
        super(Attention, self).__init__()
        hidden_planes = max(int(in_planes * reduction), 16)
        self.kw_planes_per_mixture = num_static_cell + 1
        self.num_local_mixture = num_local_mixture
        self.kw_planes = self.kw_planes_per_mixture * num_local_mixture

        self.num_local_cell = int(cell_num_ratio * num_local_mixture)
        self.num_nonlocal_cell = num_static_cell - self.num_local_cell
        self.start_cell_idx = start_cell_idx

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # Bias in fc1 is True if norm_layer is not BatchNorm1d (e.g., if it's LayerNorm)
        self.fc1 = nn.Linear(in_planes, hidden_planes, bias=(norm_layer is not nn.BatchNorm1d))
        self.norm1 = norm_layer(hidden_planes) # Instantiates BatchNorm1d(hidden_planes) or LayerNorm(hidden_planes)
        self.act1 = nn.ReLU(inplace=True)

        if nonlocal_basis_ratio >= 1.0:
            self.map_to_cell = nn.Identity()
            self.fc2 = nn.Linear(hidden_planes, self.kw_planes, bias=True)
        else:
            self.map_to_cell = self.map_to_cell_basis
            self.num_basis = max(int(self.num_nonlocal_cell * nonlocal_basis_ratio), 16)
            self.fc2 = nn.Linear(hidden_planes, (self.num_local_cell + self.num_basis + 1) * num_local_mixture, bias=False)
            self.fc3 = nn.Linear(self.num_basis, self.num_nonlocal_cell, bias=False)
            self.basis_bias = nn.Parameter(torch.zeros([self.kw_planes]), requires_grad=True).float()

        self.temp_bias = torch.zeros([self.kw_planes], requires_grad=False).float()
        self.temp_value = 0
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm): # Added initialization for LayerNorm
                nn.init.constant_(m.weight, 1) # gamma
                nn.init.constant_(m.bias, 0)   # beta


    def update_temperature(self, temp_value):
        self.temp_value = temp_value

    def init_temperature(self, start_cell_idx, num_cell_per_mixture):
        if num_cell_per_mixture >= 1.0:
            num_cell_per_mixture = int(num_cell_per_mixture)
            for idx in range(self.num_local_mixture):
                assigned_kernel_idx = int(idx * self.kw_planes_per_mixture + start_cell_idx)
                self.temp_bias[assigned_kernel_idx] = 1
                start_cell_idx += num_cell_per_mixture
            return start_cell_idx
        else:
            num_mixture_per_cell = int(1.0 / num_cell_per_mixture)
            for idx in range(self.num_local_mixture):
                if idx % num_mixture_per_cell == (idx // num_mixture_per_cell) % num_mixture_per_cell:
                    assigned_kernel_idx = int(idx * self.kw_planes_per_mixture + start_cell_idx)
                    self.temp_bias[assigned_kernel_idx] = 1
                    start_cell_idx += 1
                else:
                    assigned_kernel_idx = int(idx * self.kw_planes_per_mixture + self.kw_planes_per_mixture - 1)
                    self.temp_bias[assigned_kernel_idx] = 1
            return start_cell_idx

    def map_to_cell_basis(self, x):
        x = x.reshape([-1, self.num_local_cell + self.num_basis + 1])
        x_local, x_nonlocal, x_zero = x[:, :self.num_local_cell], x[:, self.num_local_cell:-1], x[:, -1:]
        x_nonlocal = self.fc3(x_nonlocal)
        x = torch.cat([x_nonlocal[:, :self.start_cell_idx], x_local, x_nonlocal[:, self.start_cell_idx:], x_zero], dim=1)
        x = x.reshape(-1, self.kw_planes) + self.basis_bias.reshape(1, -1)
        return x

    def forward(self, x): # x is the input feature map to the KWConv layer, e.g. (B, C, H, W)
        x_pooled = self.avgpool(x.reshape(*x.shape[:2], -1)).squeeze(dim=-1) # Shape: (B, in_planes)
        
        fc1_out = self.fc1(x_pooled) # Shape: (B, hidden_planes)
        norm1_out = self.norm1(fc1_out) # norm1 is now batch-size independent if LayerNorm
        act1_out = self.act1(norm1_out)
        
        final_x = self.map_to_cell(self.fc2(act1_out)).reshape(-1, self.kw_planes_per_mixture)
        final_x = final_x / (torch.sum(torch.abs(final_x), dim=1).view(-1, 1) + 1e-3)
        final_x = (1.0 - self.temp_value) * final_x.reshape(-1, self.kw_planes) \
            + self.temp_value * self.temp_bias.to(final_x.device).view(1, -1)
        return final_x.reshape(-1, self.kw_planes_per_mixture)[:, :-1]


class KWconvNd(nn.Module):
    dimension = None
    permute = None
    func_conv = None

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=False, warehouse_id=None, warehouse_manager=None):
        super(KWconvNd, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = parse(kernel_size, self.dimension)
        self.stride = parse(stride, self.dimension)
        self.padding = parse(padding, self.dimension)
        self.dilation = parse(dilation, self.dimension)
        self.groups = groups
        self.bias = nn.Parameter(torch.zeros([self.out_planes]), requires_grad=True).float() if bias else None
        self.warehouse_id = warehouse_id
        self.warehouse_manager = [warehouse_manager]
        self.attention = None

    def init_attention(self, cell, start_cell_idx, reduction, cell_num_ratio, norm_layer, nonlocal_basis_ratio=1.0):
        self.cell_shape = cell.shape
        self.groups_out_channel = self.out_planes // self.cell_shape[1]
        self.groups_in_channel = self.in_planes // self.cell_shape[2] // self.groups
        self.groups_spatial = 1
        for idx in range(len(self.kernel_size)):
            self.groups_spatial = self.groups_spatial * self.kernel_size[idx] // self.cell_shape[3 + idx]
        num_local_mixture = self.groups_out_channel * self.groups_in_channel * self.groups_spatial
        
        # norm_layer is passed from Warehouse_Manager, which now defaults to nn.LayerNorm via KW_ResNet
        self.attention = Attention(self.in_planes, reduction, self.cell_shape[0], num_local_mixture,
                                   norm_layer=norm_layer, nonlocal_basis_ratio=nonlocal_basis_ratio,
                                   cell_num_ratio=cell_num_ratio, start_cell_idx=start_cell_idx)
        return self.attention.init_temperature(start_cell_idx, cell_num_ratio)

    def forward(self, x):
        if self.attention is None:
            # This case should ideally be handled if a dummy forward is needed before full init.
            # For now, assuming 'allocate' has run and 'attention' is initialized.
            # If not, this will raise an AttributeError.
            # For YOLO's stride calculation, it expects a valid forward.
             out_shape = [x.shape[0], self.out_planes]
             for i in range(self.dimension):
                 out_shape.append(math.floor((x.shape[2+i] + 2*self.padding[i] - self.dilation[i]*(self.kernel_size[i]-1) - 1) / self.stride[i] + 1))
             return torch.zeros(*out_shape, device=x.device, dtype=x.dtype)


        kw_attention = self.attention(x) # x is (B, C_in, H, W)
        batch_size = x.shape[0]
        
        x_reshaped = x.reshape(1, batch_size * self.in_planes, *x.shape[2:])

        weight_warehouse = self.warehouse_manager[0].take_cell(self.warehouse_id)
        
        _cell_shape_0 = self.cell_shape[0]
        _cell_shape_1_plus = self.cell_shape[1:]

        weight = weight_warehouse.reshape(_cell_shape_0, -1)
        aggregate_weight = torch.mm(kw_attention, weight)
        
        _groups_out_channel = self.groups_out_channel
        _groups_in_channel = self.groups_in_channel
        _groups_spatial = self.groups_spatial

        aggregate_weight = aggregate_weight.reshape(batch_size, _groups_spatial, _groups_out_channel,
                                                     _groups_in_channel, *_cell_shape_1_plus)
        aggregate_weight = aggregate_weight.permute(*self.permute)
        aggregate_weight = aggregate_weight.reshape(batch_size * self.out_planes, self.in_planes // self.groups, *self.kernel_size)
        
        output = self.func_conv(x_reshaped, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                                dilation=self.dilation, groups=self.groups * batch_size)
        
        output = output.view(batch_size, self.out_planes, *output.shape[2:])
        if self.bias is not None:
            output = output + self.bias.reshape(1, -1, *([1]*self.dimension))
        return output


class KWConv1d(KWconvNd):
    dimension = 1
    permute = (0, 2, 4, 3, 5, 1, 6)
    func_conv = F.conv1d


class KWConv2d(KWconvNd):
    dimension = 2
    permute = (0, 2, 4, 3, 5, 1, 6, 7)
    func_conv = F.conv2d


class KWConv3d(KWconvNd):
    dimension = 3
    permute = (0, 2, 4, 3, 5, 1, 6, 7, 8)
    func_conv = F.conv3d


class KWLinear(nn.Module):
    dimension = 1

    def __init__(self, *args, **kwargs):
        super(KWLinear, self).__init__()
        in_features, out_features = args[0], args[1]
        conv_kwargs = {k: v for k, v in kwargs.items() if k not in ['in_planes', 'out_planes']}
        conv_kwargs.setdefault('kernel_size', 1)
        self.conv = KWConv1d(in_features, out_features, **conv_kwargs)

    def forward(self, x):
        shape = x.shape
        x_reshaped = x.reshape(-1, shape[-1], 1)
        out_conv = self.conv(x_reshaped)
        out_reshaped = out_conv.squeeze(-1).reshape(*shape[:-1], -1)
        return out_reshaped


class Warehouse_Manager(nn.Module):
    def __init__(self, reduction=0.0625, cell_num_ratio=1, cell_inplane_ratio=1,
                 cell_outplane_ratio=1, sharing_range=(), nonlocal_basis_ratio=1,
                 norm_layer=nn.LayerNorm, spatial_partition=True): # Default norm_layer changed here for clarity, but it's set by KW_ResNet
        super(Warehouse_Manager, self).__init__()
        self.sharing_range = sharing_range
        self.warehouse_list = {}
        self.reduction = reduction
        self.spatial_partition = spatial_partition
        self.cell_num_ratio = cell_num_ratio
        self.cell_outplane_ratio = cell_outplane_ratio
        self.cell_inplane_ratio = cell_inplane_ratio
        self.norm_layer = norm_layer # This will be nn.LayerNorm if KW_ResNet passes it
        self.nonlocal_basis_ratio = nonlocal_basis_ratio
        self.weights = nn.ParameterList()

    def fuse_warehouse_name(self, warehouse_name):
        fused_names = []
        for sub_name in warehouse_name.split('_'):
            match_name = sub_name
            for sharing_name in self.sharing_range:
                if str.startswith(match_name, sharing_name):
                    match_name = sharing_name
            fused_names.append(match_name)
        fused_names = '_'.join(fused_names)
        return fused_names

    def reserve(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                bias=True, warehouse_name='default', enabled=True, layer_type='conv2d'):
        kw_mapping = {'conv1d': KWConv1d, 'conv2d': KWConv2d, 'conv3d': KWConv3d, 'linear': KWLinear}
        org_mapping = {'conv1d': nn.Conv1d, 'conv2d': nn.Conv2d, 'conv3d': nn.Conv3d, 'linear': nn.Linear}

        if not enabled:
            layer_cls = org_mapping[layer_type]
            if layer_cls is nn.Linear:
                return layer_cls(in_planes, out_planes, bias=bias)
            else:
                return layer_cls(in_planes, out_planes, kernel_size, stride=stride, padding=padding, dilation=dilation,
                                  groups=groups, bias=bias)
        else:
            layer_cls = kw_mapping[layer_type]
            warehouse_name = self.fuse_warehouse_name(warehouse_name)
            
            if layer_type == 'linear':
                _kernel_size = parse(kernel_size if isinstance(kernel_size, (int, tuple, list)) else 1, layer_cls.dimension)
            else:
                 _kernel_size = parse(kernel_size, layer_cls.dimension)
            weight_shape = [out_planes, in_planes // groups, *_kernel_size]

            if warehouse_name not in self.warehouse_list.keys():
                self.warehouse_list[warehouse_name] = []
            self.warehouse_list[warehouse_name].append(weight_shape)
            
            warehouse_idx = int(list(self.warehouse_list.keys()).index(warehouse_name))

            if layer_type == 'linear':
                return layer_cls(in_planes, out_planes, 
                                 stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
                                 warehouse_id=warehouse_idx,
                                 warehouse_manager=self)
            else:
                return layer_cls(in_planes, out_planes, kernel_size, stride=stride, padding=padding,
                                dilation=dilation, groups=groups, bias=bias,
                                warehouse_id=warehouse_idx,
                                warehouse_manager=self)

    def store(self):
        if not self.warehouse_list:
            return

        warehouse_names = list(self.warehouse_list.keys())
        self.reduction = parse(self.reduction, len(warehouse_names))
        self.spatial_partition = parse(self.spatial_partition, len(warehouse_names))
        self.cell_num_ratio = parse(self.cell_num_ratio, len(warehouse_names))
        self.cell_outplane_ratio = parse(self.cell_outplane_ratio, len(warehouse_names))
        self.cell_inplane_ratio = parse(self.cell_inplane_ratio, len(warehouse_names))
        
        # nonlocal_basis_ratio can be a single value or a list/tuple
        if not isinstance(self.nonlocal_basis_ratio, (list, tuple)):
            self.nonlocal_basis_ratio = parse(self.nonlocal_basis_ratio, len(warehouse_names))


        for idx, warehouse_name in enumerate(self.warehouse_list.keys()):
            warehouse = self.warehouse_list[warehouse_name]
            dimension = len(warehouse[0]) - 2

            out_plane_gcd, in_plane_gcd, kernel_size_list = warehouse[0][0], warehouse[0][1], warehouse[0][2:]
            for layer_shape in warehouse:
                out_plane_gcd = math.gcd(out_plane_gcd, layer_shape[0])
                in_plane_gcd = math.gcd(in_plane_gcd, layer_shape[1])
                if not self.spatial_partition[idx]:
                    current_kernel_size = layer_shape[2:]
                    if len(kernel_size_list) != len(current_kernel_size) or \
                       any(k1 != k2 for k1, k2 in zip(kernel_size_list, current_kernel_size)):
                        raise ValueError(
                            f"Warehouse '{warehouse_name}': All layers must have the same kernel size "
                            f"({kernel_size_list}) when spatial_partition is False, but found {current_kernel_size}."
                        )
            
            cell_kernel_size = parse(1, dimension) if self.spatial_partition[idx] else kernel_size_list
            cell_in_plane = max(int(in_plane_gcd * self.cell_inplane_ratio[idx]), 1)
            cell_out_plane = max(int(out_plane_gcd * self.cell_outplane_ratio[idx]), 1)

            num_total_mixtures = 0
            for layer_shape_in_warehouse in warehouse:
                groups_channel = int(layer_shape_in_warehouse[0] // cell_out_plane * layer_shape_in_warehouse[1] // cell_in_plane)
                groups_spatial = 1
                for d_idx in range(dimension):
                    groups_spatial = int(groups_spatial * layer_shape_in_warehouse[2 + d_idx] // cell_kernel_size[d_idx])
                num_layer_mixtures = groups_spatial * groups_channel
                num_total_mixtures += num_layer_mixtures
            
            num_cells_in_warehouse = max(int(num_total_mixtures * self.cell_num_ratio[idx]), 1)
            self.weights.append(nn.Parameter(torch.randn(
                num_cells_in_warehouse,
                cell_out_plane, cell_in_plane, *cell_kernel_size), requires_grad=True))

    def allocate(self, network, _init_weights=partial(nn.init.kaiming_normal_, mode='fan_out', nonlinearity='relu')):
        if not self.weights:
            return

        num_warehouse = len(self.weights)
        end_idxs = [0] * num_warehouse

        for m_module in network.modules():
            if isinstance(m_module, KWconvNd):
                if m_module.warehouse_id is None:
                    print(f"Warning: KW layer {m_module} has no warehouse_id. Skipping allocation.")
                    continue
                warehouse_idx = m_module.warehouse_id
                
                if not (0 <= warehouse_idx < len(self.weights) and \
                        0 <= warehouse_idx < len(self.reduction) and \
                        0 <= warehouse_idx < len(self.cell_num_ratio) and \
                        0 <= warehouse_idx < len(self.nonlocal_basis_ratio)):
                    raise RuntimeError(f"Error allocating for KW layer {m_module}: "
                                       f"warehouse_id {warehouse_idx} is out of bounds for manager's parsed lists. "
                                       f"Weights len: {len(self.weights)}, Reductions len: {len(self.reduction)}, etc. "
                                       "Ensure warehouse_manager.store() was called and ran correctly.")
                
                start_cell_idx = end_idxs[warehouse_idx]
                current_nonlocal_basis_ratio = self.nonlocal_basis_ratio[warehouse_idx]

                end_cell_idx = m_module.init_attention(self.weights[warehouse_idx],
                                                    start_cell_idx,
                                                    self.reduction[warehouse_idx],
                                                    self.cell_num_ratio[warehouse_idx],
                                                    norm_layer=self.norm_layer, # This is crucial, uses the manager's norm_layer
                                                    nonlocal_basis_ratio=current_nonlocal_basis_ratio)
                
                if start_cell_idx < end_cell_idx :
                     _init_weights(self.weights[warehouse_idx][start_cell_idx:end_cell_idx].view(
                        -1, *self.weights[warehouse_idx].shape[2:]))
                
                end_idxs[warehouse_idx] = end_cell_idx

        for warehouse_idx in range(len(end_idxs)):
            if self.weights[warehouse_idx].numel() > 0:
                if end_idxs[warehouse_idx] != self.weights[warehouse_idx].shape[0]:
                    print(f"Warning: Warehouse {warehouse_idx}: end_cell_idx ({end_idxs[warehouse_idx]}) "
                          f"does not match total cells ({self.weights[warehouse_idx].shape[0]}).")

    def take_cell(self, warehouse_idx):
        return self.weights[warehouse_idx]


def kwconv3x3(in_planes, out_planes, stride=1, warehouse_name=None, warehouse_manager=None, enabled=True):
    return warehouse_manager.reserve(in_planes, out_planes, kernel_size=3, stride=stride, padding=1,
                                     warehouse_name=warehouse_name, enabled=enabled, bias=False, layer_type='conv2d')


def kwconv1x1(in_planes, out_planes, stride=1, warehouse_name=None, warehouse_manager=None, enabled=True):
    return warehouse_manager.reserve(in_planes, out_planes, kernel_size=1, stride=stride, padding=0,
                                     warehouse_name=warehouse_name, enabled=enabled, bias=False, layer_type='conv2d')


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 stage_idx=None, layer_idx=None, warehouse_manager=None, warehouse_handover=False, drop_path=0.):
        super(BasicBlock, self).__init__()
        conv1_stage_idx = max(stage_idx - 1 if warehouse_handover else stage_idx, 0)
        self.conv1 = kwconv3x3(inplanes, planes, stride,
                               warehouse_name='stage{}_layer{}_conv{}'.format(conv1_stage_idx, layer_idx, 0),
                               warehouse_manager=warehouse_manager)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        layer_idx_conv2 = 0 if warehouse_handover and stage_idx > 0 else layer_idx 
        self.conv2 = kwconv3x3(planes, planes,
                               warehouse_name='stage{}_layer{}_conv{}'.format(stage_idx, layer_idx_conv2, 1),
                               warehouse_manager=warehouse_manager)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = identity + self.drop_path(out)
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 stage_idx=None, layer_idx=None, warehouse_manager=None, warehouse_handover=False, drop_path=0.):
        super(Bottleneck, self).__init__()
        conv1_stage_idx = stage_idx - 1 if warehouse_handover else stage_idx
        self.conv1 = kwconv1x1(inplanes, planes,
                               warehouse_name='stage{}_layer{}_conv{}'.format(conv1_stage_idx, layer_idx, 0),
                               warehouse_manager=warehouse_manager, enabled=(conv1_stage_idx >= 0))
        self.bn1 = nn.BatchNorm2d(planes)
        layer_idx_conv23 = 0 if warehouse_handover and stage_idx > 0 else layer_idx
        self.conv2 = kwconv3x3(planes, planes, stride,
                               warehouse_name='stage{}_layer{}_conv{}'.format(stage_idx, layer_idx_conv23, 1),
                               warehouse_manager=warehouse_manager)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = kwconv1x1(planes, planes * self.expansion,
                               warehouse_name='stage{}_layer{}_conv{}'.format(stage_idx, layer_idx_conv23, 2),
                               warehouse_manager=warehouse_manager)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
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
        out = identity + self.drop_path(out)
        out = self.relu(out)
        return out


class KW_ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, dropout=0.0,
                 reduction=0.0625,
                 cell_num_ratio=1, cell_inplane_ratio=1, cell_outplane_ratio=1,
                 sharing_range=('layer', 'conv'), nonlocal_basis_ratio=1, drop_path_rate=0., 
                 input_channels=3, **kwargs):
        super(KW_ResNet, self).__init__()
        # CRITICAL CHANGE: Use nn.LayerNorm for the Attention module's internal normalization
        # This makes it robust to batch_size=1 issues during training mode (e.g. in YOLO's stride calculation)
        self.warehouse_manager = Warehouse_Manager(
            reduction, cell_num_ratio, cell_inplane_ratio, cell_outplane_ratio,
            sharing_range, nonlocal_basis_ratio,
            norm_layer=nn.LayerNorm # <<<< CHANGED from nn.BatchNorm1d
        )
        self.inplanes = 64
        self.layer_idx_counter = 0 

        self.conv1 = nn.Conv2d(input_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(layers))] 
        
        current_dpr_idx = 0
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, stage_idx=0, 
                                       drop_paths=dpr[current_dpr_idx : current_dpr_idx + layers[0]])
        current_dpr_idx += layers[0]
        
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, stage_idx=1,
                                       drop_paths=dpr[current_dpr_idx : current_dpr_idx + layers[1]])
        current_dpr_idx += layers[1]

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, stage_idx=2,
                                       drop_paths=dpr[current_dpr_idx : current_dpr_idx + layers[2]])
        current_dpr_idx += layers[2]
        
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, stage_idx=3,
                                       drop_paths=dpr[current_dpr_idx : current_dpr_idx + layers[3]])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) and not isinstance(m, KWConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)): # Exclude LayerNorm as it's handled in Attention
                if m.weight is not None: nn.init.constant_(m.weight, 1)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
        
        self.width_list = []
        try:
            original_mode = self.training
            self.eval()
            with torch.no_grad():
                dummy_input_size = kwargs.get('dummy_input_size', 224)
                dummy_input = torch.randn(1, input_channels, dummy_input_size, dummy_input_size)
                # The dummy forward for width_list should ideally run after allocate
                # However, for pure shape inference, if KWConvNd.forward can handle uninitialized attention
                # (e.g., by returning zero tensor of correct shape), this might be okay.
                # For safety, let's ensure allocate is called if possible, or that KWConv is robust.
                # The current KWConvNd.forward has a fallback if self.attention is None.

                # To avoid issues with unallocated warehouses during this dummy pass for width_list,
                # we ensure store and allocate are called BEFORE this dummy pass.
                # This implies that KW_ResNet layers must be fully defined before this.
                # The current order is: define layers -> width_list calc -> store -> allocate.
                # Let's try to call store/allocate *before* width_list calc for the dummy pass.
                # This is complex because store/allocate depend on all KW layers being registered.
                # The simplest is to ensure KWConvNd's forward is robust for shape inference.
                # The current KWConvNd.forward already has a small fallback if self.attention is None.
                features = self._forward_extract_features(dummy_input)
                self.width_list = [f.size(1) for f in features]
            self.train(original_mode)
        except Exception as e:
            print(f"Warning: Could not compute width_list during KW_ResNet init: {e}")
            # import traceback; traceback.print_exc() # Uncomment for detailed debugging
            self.width_list = []

        self.warehouse_manager.store()
        self.warehouse_manager.allocate(self)

    def _make_layer(self, block, planes, num_blocks, stride=1, stage_idx=-1, drop_paths=None):
        downsample = None
        downsample_enabled = (stride != 1 or self.inplanes != planes * block.expansion)
        
        # warehouse_handover means the conv1 of the first block of this stage
        # might use weights from the *previous* stage's warehouse index.
        # This primarily affects the warehouse_name.
        # The downsample layer, if strided, conceptually belongs to the transition *from* the previous feature map size.
        warehouse_handover_for_downsample_naming = (stage_idx > 0 and stride !=1) 

        if downsample_enabled:
            # Name for downsample conv, potentially associated with previous stage if handover naming
            downsample_conv_name = f'stage{stage_idx-1 if warehouse_handover_for_downsample_naming else stage_idx}_ds_layer{self.layer_idx_counter}'
            
            use_kw_downsample = True # Default to using KW for downsample if it's a conv operation

            if stride != 1: # Actual strided convolution, suitable for KW
                downsample_conv = self.warehouse_manager.reserve(
                    self.inplanes, planes * block.expansion, kernel_size=1, stride=stride,
                    warehouse_name=downsample_conv_name,
                    enabled=use_kw_downsample, 
                    layer_type='conv2d', bias=False
                )
            else: # Projection only (stride=1, inplanes != planes*exp), use standard Conv2d
                downsample_conv = nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1, stride=1, bias=False)
                nn.init.kaiming_normal_(downsample_conv.weight, mode='fan_out', nonlinearity='relu')

            downsample = nn.Sequential(
                downsample_conv,
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # warehouse_handover for the block itself implies its conv1 might use prev stage name
        block_warehouse_handover = (stage_idx > 0) 

        layers.append(block(self.inplanes, planes, stride, downsample,
                            stage_idx=stage_idx, layer_idx=self.layer_idx_counter, 
                            warehouse_manager=self.warehouse_manager,
                            warehouse_handover=block_warehouse_handover, 
                            drop_path=drop_paths[0] if drop_paths else 0.))
        
        self.inplanes = planes * block.expansion
        # self.layer_idx_counter += 1 # Increment counter *after* using it for the first block

        for idx_block in range(1, num_blocks): # Corrected: use new var for block index
            # layer_idx for subsequent blocks in the same stage should increment or be managed
            # If layer_idx is per-stage, it would reset. If global, it increments.
            # Current `layer_idx` in block is from `self.layer_idx_counter` which is global.
            current_block_layer_idx = self.layer_idx_counter + idx_block # Make layer_idx unique per block
            layers.append(block(self.inplanes, planes, stage_idx=stage_idx, layer_idx=current_block_layer_idx,
                                warehouse_manager=self.warehouse_manager,
                                warehouse_handover=False, 
                                drop_path=drop_paths[idx_block] if drop_paths else 0.))
        
        self.layer_idx_counter += num_blocks # Increment by number of blocks in this layer
            
        return nn.Sequential(*layers)

    def net_update_temperature(self, temp):
        for m in self.modules():
            if hasattr(m, "update_temperature"):
                m.update_temperature(temp)

    def _forward_extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x_pool = self.maxpool(x)

        f1 = self.layer1(x_pool)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return [f1, f2, f3, f4] 

    def forward_classification(self, x: torch.Tensor) -> torch.Tensor:
        features = self._forward_extract_features(x)
        out = self.avgpool(features[-1]) 
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self._forward_extract_features(x)


@register_model
def kw_resnet18(pretrained=False, **kwargs) -> KW_ResNet:
    model_kwargs = {k: v for k, v in kwargs.items() if k in [
        'num_classes', 'dropout', 'reduction', 'cell_num_ratio', 
        'cell_inplane_ratio', 'cell_outplane_ratio', 'sharing_range', 
        'nonlocal_basis_ratio', 'drop_path_rate', 'input_channels', 'dummy_input_size'
    ]}
    model = KW_ResNet(BasicBlock, [2, 2, 2, 2], **model_kwargs)
    return model

@register_model
def kw_resnet50(pretrained=False, **kwargs) -> KW_ResNet:
    model_kwargs = {k: v for k, v in kwargs.items() if k in [
        'num_classes', 'dropout', 'reduction', 'cell_num_ratio', 
        'cell_inplane_ratio', 'cell_outplane_ratio', 'sharing_range', 
        'nonlocal_basis_ratio', 'drop_path_rate', 'input_channels', 'dummy_input_size'
    ]}
    model = KW_ResNet(Bottleneck, [3, 4, 6, 3], **model_kwargs)
    return model

# Example Usage:
if __name__ == '__main__':
    print("--- Testing KW-ResNet18 ---")
    image_size = (2, 3, 640, 640) 

    model18_default = kw_resnet18(num_classes=100, drop_path_rate=0.1, input_channels=image_size[1], dummy_input_size=image_size[2])
    print(f"KW-ResNet18 (default) with input {image_size}")
    dummy_input = torch.randn(*image_size)
    
    print("Running forward pass for feature extraction...")
    try:
        # Test in eval mode first, then training mode to check BN/LN behavior
        model18_default.eval()
        output_features_eval = model18_default(dummy_input)
        print("Output features (eval mode):")
        for i, feat in enumerate(output_features_eval):
            print(f"Feature {i+1} shape: {feat.shape}")
        
        model18_default.train()
        output_features_train = model18_default(dummy_input)
        print("\nOutput features (train mode):")
        for i, feat in enumerate(output_features_train):
            print(f"Feature {i+1} shape: {feat.shape}")

        print(f"Model width_list: {model18_default.width_list}")
        if model18_default.width_list : # check if width_list was populated
            assert len(model18_default.width_list) == 4
            for i in range(4):
                assert model18_default.width_list[i] == output_features_eval[i].size(1)

        print("\nRunning forward pass for classification...")
        model18_default.eval() # Classification typically done in eval mode
        classification_output = model18_default.forward_classification(dummy_input)
        print(f"Classification output shape: {classification_output.shape}")
        assert classification_output.shape == (image_size[0], 100)
        print("KW-ResNet18 (default) test successful.")

    except Exception as e:
        print(f"Error during KW-ResNet18 (default) test: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Testing KW-ResNet50 with custom warehouse params ---")
    model50_custom = kw_resnet50(
        num_classes=50,
        reduction=0.125,                   
        cell_num_ratio=1.2,                
        cell_inplane_ratio=0.5,            
        cell_outplane_ratio=0.5,           
        sharing_range=('stage',),          
        nonlocal_basis_ratio=0.5,          
        drop_path_rate=0.2,
        input_channels=image_size[1],
        dummy_input_size=image_size[2]     
    )
    print(f"KW-ResNet50 (custom) with input {image_size}")
    dummy_input_50 = torch.randn(*image_size)
    
    print("Running forward pass for feature extraction...")
    try:
        model50_custom.eval()
        output_features_50 = model50_custom(dummy_input_50)
        print("Output features (list of tensors):")
        for i, feat in enumerate(output_features_50):
            print(f"Feature {i+1} shape: {feat.shape}")
        print(f"Model width_list: {model50_custom.width_list}")

        print("\nRunning forward pass for classification...")
        classification_output_50 = model50_custom.forward_classification(dummy_input_50)
        print(f"Classification output shape: {classification_output_50.shape}")
        print("KW-ResNet50 (custom) test successful.")
    except Exception as e:
        print(f"Error during KW-ResNet50 (custom) test: {e}")
        import traceback
        traceback.print_exc()