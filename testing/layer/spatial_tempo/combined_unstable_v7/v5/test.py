# test1.py: CustomConvLayer with reenabled memory update and strict graph management

import torch
import torch.nn as nn
import torch.nn.functional as F
import sympy
from sympy import sympify, symbols
import numpy as np
import re
import itertools
import warnings

def trim(s):
    return s.strip()

def load_kof(path):
    """Load .kof file with support for multiple kernels and channel-specific taps."""
    with open(path, 'r') as f:
        lines = f.readlines()

    ndim = None
    kernels = []
    current_kernel = None
    global_op = 'x * w'
    global_in = 'x'
    global_k = 'w'
    reading_taps = False
    current_tap_index = -1
    channel_taps = {}

    for line in lines:
        s = trim(line)
        if not s or s.startswith('#'): continue
        if s.startswith('ndim:'):
            ndim = int(trim(s[5:]))
        elif s.startswith('kernel:'):
            if current_kernel is not None:
                kernels.append(current_kernel)
            current_kernel = {'taps': [], 'global_op': global_op, 'global_in': global_in, 'global_k': global_k}
            current_tap_index = -1
            reading_taps = False
        elif s.startswith('channel:'):
            current_channel = int(trim(s[8:]))
            channel_taps[current_channel] = []
        elif s.startswith('taps:'):
            reading_taps = True
        elif reading_taps and (s[0].isdigit() or (s[0] == '-' and len(s) > 1 and s[1].isdigit()) or '.' in s):
            parts = re.split(r'\s+', s)
            if len(parts) != ndim + 1:
                raise ValueError(f"Invalid tap format: expected {ndim + 1} parts, got {len(parts)}")
            offs = list(map(float, parts[:ndim]))
            w = float(parts[ndim])
            tap_dict = {'offsets': offs, 'weight': w, 'op': None, 'input_encode': None, 'kernel_encode': None}
            if 'current_channel' in locals() and current_channel in channel_taps:
                channel_taps[current_channel].append(tap_dict)
            elif current_kernel is not None:
                current_kernel['taps'].append(tap_dict)
            else:
                raise ValueError("Taps defined without active kernel")
            current_tap_index += 1
        elif s.startswith('operation:'):
            formula = trim(s[10:])
            if current_tap_index >= 0 and current_kernel is not None:
                current_kernel['taps'][current_tap_index]['op'] = formula
            elif current_kernel is not None:
                current_kernel['global_op'] = formula
            else:
                global_op = formula
        elif s.startswith('input_encode:'):
            formula = trim(s[13:])
            if current_tap_index >= 0 and current_kernel is not None:
                current_kernel['taps'][current_tap_index]['input_encode'] = formula
            elif current_kernel is not None:
                current_kernel['global_in'] = formula
            else:
                global_in = formula
        elif s.startswith('kernel_encode:'):
            formula = trim(s[14:])
            if current_tap_index >= 0 and current_kernel is not None:
                current_kernel['taps'][current_tap_index]['kernel_encode'] = formula
            elif current_kernel is not None:
                current_kernel['global_k'] = formula
            else:
                global_k = formula

    if current_kernel is not None:
        kernels.append(current_kernel)
    if ndim is None or (not kernels and not channel_taps):
        raise ValueError("Invalid .kof file: missing ndim or taps")
    return {'ndim': ndim, 'kernels': kernels, 'channel_taps': channel_taps, 'global_op': global_op, 'global_in': global_in, 'global_k': global_k}

def load_tprl(path):
    """Load .tprl file with support for multiple formulas, conditions, and adaptive thresholds."""
    if not path:
        return None

    with open(path, 'r') as f:
        lines = [trim(line.strip()) for line in f.readlines() if line.strip() and not line.startswith('#')]

    section = None
    vars_list = []
    formulas = []
    conditions = []
    testing_lhs = None
    testing_rhs = None
    rel_op = None
    memory_formula = None
    param_names = []
    param_inits = []
    history_length = 5
    grad_threshold = 1e-4
    adaptive_history = False

    for line in lines:
        if line.startswith('[') and line.endswith(']'):
            section = line[1:-1].lower()
            continue
        if section == 'vars':
            vars_list.extend([v.strip() for v in line.split() if v.strip()])
        elif section == 'formula':
            formulas.append(line)
        elif section == 'conditions':
            parts = [p.strip() for p in line.split(':')]
            if len(parts) == 2:
                conditions.append({'cond': parts[0], 'formula_idx': int(parts[1])})
        elif section == 'testing':
            for op in ['>=', '<=', '!=', '==', '>', '<']:
                if op in line:
                    parts = line.split(op)
                    if len(parts) == 2:
                        testing_lhs = trim(parts[0])
                        testing_rhs = trim(parts[1])
                        rel_op = op
                    break
        elif section == 'memory':
            memory_formula = line
        elif section == 'params':
            parts = re.split(r'\s+', line)
            if len(parts) >= 2 and parts[0].endswith('_init'):
                param_names.append(parts[0][:-5])
                param_inits.append(float(parts[1]))
            elif 'history_length' in line:
                hl_parts = re.split(r'\s+', line)
                history_length = int(hl_parts[1])
            elif 'grad_threshold' in line:
                gt_parts = re.split(r'\s+', line)
                grad_threshold = float(gt_parts[1])
            elif 'adaptive_history' in line and 'true' in line.lower():
                adaptive_history = True

    hist_vars = [v for v in vars_list if v.startswith('wh') or v.startswith('gh')]
    for v in hist_vars:
        if v.startswith('wh') or v.startswith('gh'):
            idx = int(v[2:]) if v[2:].isdigit() else -1
            if idx >= history_length:
                warnings.warn(f"Variable {v} in .tprl exceeds history_length {history_length}")

    if not formulas or not vars_list:
        raise ValueError("Invalid .tprl file: missing formula or vars")
    return {
        'vars': vars_list,
        'formulas': formulas,
        'conditions': conditions,
        'testing_lhs': testing_lhs,
        'testing_rhs': testing_rhs,
        'rel_op': rel_op,
        'memory_formula': memory_formula,
        'param_names': param_names,
        'param_inits': param_inits,
        'history_length': history_length,
        'grad_threshold': grad_threshold,
        'adaptive_history': adaptive_history
    }

class SymPyFunction(torch.autograd.Function):
    """Custom SymPy function evaluation with strict graph management."""
    @staticmethod
    def forward(ctx, input_tensor, sympy_func, *args):
        input_np = input_tensor.detach().cpu().numpy()
        tensor_args = [arg for arg in args if isinstance(arg, torch.Tensor)]
        non_tensor_args = [arg for arg in args if not isinstance(arg, torch.Tensor)]
        tensor_args_np = [arg.detach().cpu().numpy() for arg in tensor_args]
        output_np = sympy_func(input_np, *(tensor_args_np + non_tensor_args))
        output = torch.as_tensor(output_np, dtype=torch.float32, device=input_tensor.device)
        ctx.save_for_backward(input_tensor, *tensor_args)
        ctx.sympy_func = sympy_func
        ctx.non_tensor_args = non_tensor_args
        ctx.num_tensor_args = len(tensor_args)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, *tensor_args = ctx.saved_tensors
        sympy_func = ctx.sympy_func
        non_tensor_args = ctx.non_tensor_args
        num_tensor_args = ctx.num_tensor_args

        h = 1e-5
        input_np = input_tensor.detach().cpu().numpy()
        tensor_args_np = [arg.detach().cpu().numpy() for arg in tensor_args]
        grad_input = None
        grad_args = [None] * num_tensor_args

        with torch.no_grad():
            output_base = sympy_func(input_np, *(tensor_args_np + non_tensor_args))
            perturbed_input = input_np + h
            output_perturbed = sympy_func(perturbed_input, *(tensor_args_np + non_tensor_args))
            grad_input_np = (output_perturbed - output_base) / h
            grad_input = torch.as_tensor(grad_input_np, dtype=torch.float32, device=input_tensor.device) * grad_output

            for i in range(num_tensor_args):
                perturbed_arg = tensor_args_np[i] + h
                output_perturbed = sympy_func(input_np, *([perturbed_arg if j == i else tensor_args_np[j] for j in range(num_tensor_args)] + non_tensor_args))
                grad_args[i] = torch.as_tensor((output_perturbed - output_base) / h, dtype=torch.float32, device=tensor_args[i].device) * grad_output

        return (grad_input, None) + tuple(grad_args) + (None,) * len(non_tensor_args)

class CustomConvLayer(nn.Module):
    """Custom convolutional layer with reenabled memory update."""
    def __init__(self, kof_path, tprl_path=None, in_channels=1, out_channels=1, padding=0, stride=1, kernel_idx=0, verbose=True):
        super().__init__()
        kof = load_kof(kof_path)
        self.ndim = kof['ndim']
        self.kernel_idx = kernel_idx
        self.kernels = kof['kernels']
        self.verbose = verbose
        if not self.kernels:
            raise ValueError("No kernels defined in .kof file")
        if kernel_idx >= len(self.kernels):
            raise ValueError(f"Invalid kernel_idx {kernel_idx}, only {len(self.kernels)} kernels available")
        self.tap_offsets = torch.tensor([t['offsets'] for t in self.kernels[kernel_idx]['taps']])
        self.need_interp = not all(float(o).is_integer() for tap in self.kernels[kernel_idx]['taps'] for o in tap['offsets'])
        self.k_elems = len(self.kernels[kernel_idx]['taps'])
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weights_list = nn.ParameterList()
        for k in self.kernels:
            weights_init = torch.tensor([t['weight'] for t in k['taps']])
            self.weights_list.append(nn.Parameter(weights_init.view(1, 1, -1).repeat(out_channels, in_channels, 1)))

        if isinstance(padding, int):
            padding = [padding] * self.ndim
        min_off = self.tap_offsets.min(dim=0)[0]
        max_off = self.tap_offsets.max(dim=0)[0]
        if not torch.isfinite(min_off).all() or not torch.isfinite(max_off).all():
            raise ValueError("Tap offsets contain non-finite values")
        computed_padding = [max(p, int(torch.ceil(abs(min_off[d])).item()), int(torch.ceil(max_off[d]).item())) for d, p in enumerate(padding)]
        self.padding = torch.tensor(computed_padding, dtype=torch.long)
        if computed_padding != padding and self.verbose:
            warnings.warn(f"Adjusted padding from {padding} to {computed_padding} to accommodate tap offsets")

        if isinstance(stride, int):
            stride = [stride] * self.ndim
        self.stride = torch.tensor(stride, dtype=torch.long)

        if any(self.padding < 0) or any(self.stride <= 0):
            raise ValueError("Padding must be non-negative and stride positive")

        if self.ndim == 3 and not self.need_interp:
            kernel_size = [int(torch.ceil(max_off[d] - min_off[d] + 1).item()) for d in range(self.ndim)]
            if any(ks <= 0 for ks in kernel_size):
                raise ValueError(f"Invalid kernel size {kernel_size} derived from tap offsets")
            self.kernel_size = kernel_size
            self.kernel_offset = -min_off.long()
            self.kernel = torch.zeros(self.out_channels, self.in_channels, *kernel_size)
            for t, offs in enumerate(self.tap_offsets):
                idx = tuple(self.kernel_offset + offs.long())
                self.kernel[(slice(None), slice(None), *idx)] = self.weights_list[kernel_idx][:, :, t]

        x_sym = symbols('x')
        w_sym = symbols('w')
        d_syms = symbols([f'd{i}' for i in range(self.ndim)])
        self.op_funcs_list = []
        self.input_encode_funcs_list = []
        self.kernel_encode_funcs_list = []
        for k in self.kernels:
            op_funcs = []
            input_encode_funcs = []
            kernel_encode_funcs = []
            for tap in k['taps']:
                op_expr = sympify(tap['op'] or k['global_op'])
                in_expr = sympify(tap['input_encode'] or k['global_in'])
                k_expr = sympify(tap['kernel_encode'] or k['global_k'])
                op_funcs.append(sympy.lambdify([x_sym, w_sym], op_expr, 'numpy'))
                input_encode_funcs.append(sympy.lambdify([x_sym], in_expr, 'numpy'))
                kernel_encode_funcs.append(sympy.lambdify([w_sym] + d_syms, k_expr, 'numpy'))
            self.op_funcs_list.append(op_funcs)
            self.input_encode_funcs_list.append(input_encode_funcs)
            self.kernel_encode_funcs_list.append(kernel_encode_funcs)

        self.temporal = False
        self.adaptive_history = False
        if tprl_path:
            self.temporal = True
            tprl = load_tprl(tprl_path)
            existing_hist_vars = {v for v in tprl['vars'] if v.startswith('wh') or v.startswith('gh')}
            hist_wh = [f'wh{i}' for i in range(tprl['history_length']) if f'wh{i}' not in existing_hist_vars]
            hist_gh = [f'gh{i}' for i in range(tprl['history_length']) if f'gh{i}' not in existing_hist_vars]
            self.t_vars = tprl['vars'] + hist_wh + hist_gh
            self.t_syms = symbols(self.t_vars)
            self.formula_funcs = [sympy.lambdify(self.t_syms, sympify(f), 'numpy') for f in tprl['formulas']]
            self.conditions = tprl['conditions']
            self.condition_funcs = [sympy.lambdify([symbols('epoch'), symbols('grad_mean')], sympify(c['cond']), 'numpy') if c['cond'] else None for c in self.conditions]
            self.history_length = tprl['history_length']
            self.adaptive_history = tprl['adaptive_history']
            self.grad_threshold = tprl['grad_threshold']
            self.weight_history = torch.zeros((self.history_length, self.out_channels, self.in_channels, self.k_elems), dtype=torch.float32)
            self.gradient_history = torch.zeros((self.history_length, self.out_channels, self.in_channels, self.k_elems), dtype=torch.float32)
            self.history_count = 0
            self.grad_magnitudes = torch.zeros((self.out_channels, self.in_channels, self.k_elems), dtype=torch.float32)
            self.memory_state = torch.zeros((self.out_channels, self.in_channels, self.k_elems), dtype=torch.float32)
            self.k = nn.Parameter(torch.tensor(tprl['param_inits'], dtype=torch.float32))
            self.param_names = tprl['param_names']

            self.testing_lhs_func = None
            self.testing_rhs_func = None
            self.rel_op = tprl['rel_op']
            if tprl['testing_lhs']:
                lhs_expr = sympify(tprl['testing_lhs'])
                rhs_expr = sympify(tprl['testing_rhs'])
                self.testing_lhs_func = sympy.lambdify(self.t_syms, lhs_expr, 'numpy')
                self.testing_rhs_func = sympy.lambdify(self.t_syms, rhs_expr, 'numpy')

            self.memory_func = None
            if tprl['memory_formula']:
                mem_expr = sympify(tprl['memory_formula'])
                self.memory_func = sympy.lambdify(self.t_syms, mem_expr, 'numpy')

    def _get_temporal_args(self, w_tensor, g_tensor, m_tensor):
        args_dict = {'w': w_tensor.detach(), 'g': g_tensor.detach(), 'm': m_tensor.detach()}
        for i in range(self.history_length):
            args_dict[f'wh{i}'] = self.weight_history[i].detach()
            args_dict[f'gh{i}'] = self.gradient_history[i].detach()
        args = []
        k_idx = 0
        for v in self.t_vars:
            if v in args_dict:
                args.append(args_dict[v])
            else:
                args.append(self.k[k_idx].detach().expand_as(w_tensor))
                k_idx += 1
        return args

    def forward(self, input, epoch=0):
        """Forward pass with strict graph management."""
        if input.dim() != self.ndim + 2:
            raise ValueError(f"Input must have shape [batch, in_channels, *spatial({self.ndim})]")
        batch_size, in_channels = input.shape[:2]
        device = input.device
        in_shape = torch.tensor(input.shape[2:], device=device)

        pad_list = []
        for d in range(self.ndim - 1, -1, -1):
            pad_list.extend([self.padding[d].item(), self.padding[d].item()])
        input_padded = F.pad(input, pad_list)
        padded_shape = torch.tensor(input_padded.shape[2:], device=device)

        min_off = self.tap_offsets.min(dim=0)[0]
        max_off = self.tap_offsets.max(dim=0)[0]
        out_min = -min_off
        out_max = padded_shape - 1 - max_off
        out_shape_no_stride = out_max - out_min + 1
        out_shape_no_stride = torch.clamp(out_shape_no_stride, min=0)
        if torch.any(out_shape_no_stride <= 0):
            warnings.warn("Output shape is empty due to insufficient padding or large offsets")
            return torch.empty((batch_size, self.out_channels, *( [0] * self.ndim )), device=device, dtype=input.dtype)

        ranges = [torch.arange(0, out_shape_no_stride[d].item(), self.stride[d].item(), device=device) for d in range(self.ndim)]
        coords = torch.cartesian_prod(*ranges)
        out_spatial_total = coords.shape[0]
        out_shape = torch.tensor([len(r) for r in ranges], device=device)

        input_coords = coords[:, None, :] + out_min[None, None, :] + self.tap_offsets[None, :, :].to(device)

        op_funcs = self.op_funcs_list[self.kernel_idx]
        input_encode_funcs = self.input_encode_funcs_list[self.kernel_idx]
        kernel_encode_funcs = self.kernel_encode_funcs_list[self.kernel_idx]
        weights = self.weights_list[self.kernel_idx]

        if self.ndim == 3 and not self.need_interp:
            self.kernel = torch.zeros(self.out_channels, self.in_channels, *self.kernel_size, device=device)
            for t, offs in enumerate(self.tap_offsets):
                idx = tuple(self.kernel_offset + offs.long())
                self.kernel[(slice(None), slice(None), *idx)] = weights[:, :, t]
            output = F.conv3d(input_padded, self.kernel, stride=self.stride.tolist(), padding=0)
            slices = [slice(0, s) for s in in_shape.tolist()]
            output = output[(slice(None), slice(None)) + tuple(slices)]
            if self.verbose:
                print(f"F.conv3d path - Input shape: {input.shape}, Output shape before crop: {F.conv3d(input_padded, self.kernel, stride=self.stride.tolist(), padding=0).shape}, Output shape after crop: {output.shape}")
            return output

        if not self.need_interp:
            input_coords = input_coords.long()
            spatial_strides = torch.ones(self.ndim, dtype=torch.long, device=device)
            if self.ndim > 0:
                spatial_strides[-1] = 1
                for i in range(self.ndim - 2, -1, -1):
                    spatial_strides[i] = spatial_strides[i + 1] * padded_shape[i + 1]
            prod_spatial = padded_shape.prod().item()
            batch_offsets = torch.arange(batch_size, device=device)[:, None, None, None] * (self.in_channels * prod_spatial)
            channel_offsets = torch.arange(self.in_channels, device=device)[None, None, None, :] * prod_spatial
            spatial_offsets = torch.sum(input_coords * spatial_strides[None, None, :], dim=-1).view(out_spatial_total, self.k_elems, 1)
            idx = batch_offsets + channel_offsets + spatial_offsets
            idx = idx.view(-1)
            full_flat = input_padded.view(-1)
            x_flat = full_flat[idx]
            x = x_flat.view(batch_size, out_spatial_total, self.k_elems, self.in_channels)
        else:
            spatial_strides = torch.ones(self.ndim, dtype=torch.long, device=device)
            if self.ndim > 0:
                spatial_strides[-1] = 1
                for i in range(self.ndim - 2, -1, -1):
                    spatial_strides[i] = spatial_strides[i + 1] * padded_shape[i + 1]
            prod_spatial = padded_shape.prod().item()
            num_points = out_spatial_total * self.k_elems
            coords_flat = input_coords.view(num_points, self.ndim)
            floor = torch.floor(coords_flat).long()
            frac = coords_flat - floor.float()
            floor = torch.clamp(floor, min=torch.zeros_like(floor), max=padded_shape - 1)
            corner_offs_list = list(itertools.product([0, 1], repeat=self.ndim))
            corner_offs = torch.tensor(corner_offs_list, dtype=torch.long, device=device)
            num_corners = corner_offs.shape[0]
            corner_idx = floor[None, :, :] + corner_offs[:, None, :]
            corner_idx = torch.clamp(corner_idx, min=torch.zeros_like(corner_idx), max=padded_shape[None, None, :] - 1)
            weights_x = torch.prod(torch.where(corner_offs[:, None, :] == 0, 1 - frac[None, :, :], frac[None, :, :]), dim=-1)
            full_idx = (torch.arange(batch_size * self.in_channels, device=device)[:, None, None] * prod_spatial + torch.sum(corner_idx * spatial_strides[None, None, :], dim=-1)[None, :, :]).view(-1)
            full_idx = torch.clamp(full_idx, max=batch_size * self.in_channels * prod_spatial - 1)
            gathered_flat = input_padded.view(-1)[full_idx]
            full_gathered = gathered_flat.view(batch_size, self.in_channels, num_corners, num_points)
            weighted = full_gathered * weights_x[None, None, :, :]
            contrib = weighted.sum(dim=2)
            x = contrib.permute(0, 2, 1).view(batch_size, out_spatial_total, self.k_elems, self.in_channels)

        if self.temporal:
            with torch.no_grad():
                w_tensor = weights.detach()
                g_tensor = self.grad_magnitudes.detach()
                m_tensor = self.memory_state.detach()
                args = self._get_temporal_args(w_tensor, g_tensor, m_tensor)
                formula_idx = 0
                grad_mean = self.grad_magnitudes.mean().item() if self.grad_magnitudes.numel() > 0 else 0.0
                for i, cond in enumerate(self.conditions):
                    if self.condition_funcs[i]:
                        if self.condition_funcs[i](epoch, grad_mean):
                            formula_idx = cond['formula_idx']
                            break
                if formula_idx >= len(self.formula_funcs):
                    warnings.warn(f"Invalid formula_idx {formula_idx}, falling back to 0")
                    formula_idx = 0
                if self.verbose:
                    print(f"Selected formula index: {formula_idx}")
                adjusted_np = self.formula_funcs[formula_idx](*[arg.cpu().numpy() if isinstance(arg, torch.Tensor) else arg for arg in args])
                adjusted = torch.tensor(adjusted_np, dtype=torch.float32, device=device)
                if torch.isnan(adjusted).any() or torch.isinf(adjusted).any():
                    warnings.warn("Temporal formula produced NaN or Inf, falling back to original weights")
                    adjusted = w_tensor
                adjusted = torch.clamp(adjusted, min=-10.0, max=10.0)
                if (adjusted != w_tensor).any() and self.verbose:
                    warnings.warn("Adjusted weights clamped to [-10, 10] to prevent numerical instability")

                mask = torch.ones_like(w_tensor, dtype=torch.bool)
                if self.testing_lhs_func:
                    lhs_np = self.testing_lhs_func(*[arg.cpu().numpy() if isinstance(arg, torch.Tensor) else arg for arg in args])
                    rhs_np = self.testing_rhs_func(*[arg.cpu().numpy() if isinstance(arg, torch.Tensor) else arg for arg in args])
                    lhs = torch.tensor(lhs_np, dtype=torch.float32, device=device)
                    rhs = torch.tensor(rhs_np, dtype=torch.float32, device=device)
                    if self.rel_op == '>':
                        mask = lhs > rhs
                    elif self.rel_op == '<':
                        mask = lhs < rhs
                    elif self.rel_op == '>=':
                        mask = lhs >= rhs
                    elif self.rel_op == '<=':
                        mask = lhs <= rhs
                    elif self.rel_op == '==':
                        mask = lhs == rhs
                    elif self.rel_op == '!=':
                        mask = lhs != rhs
                adjusted_weights = torch.where(mask, adjusted, w_tensor)

            weights = adjusted_weights.clone().requires_grad_(True)

        w_enc = torch.zeros(self.out_channels, self.in_channels, self.k_elems, device=device, dtype=torch.float32)
        for t in range(self.k_elems):
            d_args = self.tap_offsets[t].tolist()
            w_enc[:, :, t] = SymPyFunction.apply(weights[:, :, t], kernel_encode_funcs[t], *d_args)

        x_enc = torch.zeros_like(x)
        for t in range(self.k_elems):
            x_clamped = torch.clamp(x[:, :, t, :], min=-0.999, max=float('inf'))
            if (x_clamped != x[:, :, t, :]).any() and self.verbose:
                warnings.warn("Input to input_encode clamped to prevent log(1 + x) NaN")
            x_enc[:, :, t, :] = SymPyFunction.apply(x_clamped, input_encode_funcs[t])

        output_flat = torch.zeros(batch_size, out_spatial_total, self.out_channels, device=device, dtype=torch.float32)
        for t in range(self.k_elems):
            x_enc_t = x_enc[:, :, t, :]
            w_enc_t = w_enc[:, :, t]
            x_exp = x_enc_t[:, :, None, :]
            w_exp = w_enc_t[None, None, :, :]
            combined_t = SymPyFunction.apply(x_exp, op_funcs[t], w_exp)
            output_flat += combined_t.sum(dim=-1)

        output = output_flat.view(batch_size, self.out_channels, *out_shape.tolist())
        slices = [slice(0, s) for s in in_shape.tolist()]
        output = output[(slice(None), slice(None)) + tuple(slices)]
        if self.verbose:
            print(f"Custom path - Input shape: {input.shape}, Output shape before crop: {output_flat.view(batch_size, self.out_channels, *out_shape.tolist()).shape}, Output shape after crop: {output.shape}")
        return output

    def update_histories(self):
        if not self.temporal or self.weights_list[self.kernel_idx].grad is None:
            return

        with torch.no_grad():
            grads = self.weights_list[self.kernel_idx].grad.abs().detach()
            weights = self.weights_list[self.kernel_idx].detach()

            self.grad_magnitudes = (self.grad_magnitudes * (self.history_length - 1) + grads) / self.history_length
            self.weight_history = torch.roll(self.weight_history, shifts=1, dims=0)
            self.gradient_history = torch.roll(self.gradient_history, shifts=1, dims=0)
            self.weight_history[0] = weights
            self.gradient_history[0] = grads

            if self.history_count < self.history_length:
                self.history_count += 1

            if self.adaptive_history:
                avg_grad = self.grad_magnitudes.mean()
                if avg_grad < self.grad_threshold:
                    self.history_length = max(1, self.history_length // 2)
                    self.weight_history = self.weight_history[:self.history_length]
                    self.gradient_history = self.gradient_history[:self.history_length]

            if self.memory_func:
                w_tensor = weights
                g_tensor = self.grad_magnitudes
                m_tensor = self.memory_state
                args = self._get_temporal_args(w_tensor, g_tensor, m_tensor)
                memory_output = self.memory_func(*[arg.cpu().numpy() if isinstance(arg, torch.Tensor) else arg for arg in args])
                memory_output = torch.tensor(memory_output, dtype=torch.float32, device=w_tensor.device)
                memory_output = torch.clamp(memory_output, min=-10.0, max=10.0)  # Clamp for stability
                self.memory_state = memory_output
                if self.verbose:
                    print(f"Memory state updated: min={memory_output.min().item():.6f}, max={memory_output.max().item():.6f}, mean={memory_output.mean().item():.6f}")

# Example usage for training
if __name__ == "__main__":
    import torch.optim as optim

    class ExampleNet(nn.Module):
        def __init__(self, kof_path, tprl_path):
            super().__init__()
            self.custom1 = CustomConvLayer(kof_path, tprl_path, in_channels=1, out_channels=4, padding=1, stride=1, kernel_idx=0, verbose=True)
            self.conv3d = nn.Conv3d(in_channels=4, out_channels=8, kernel_size=3, padding=1, stride=1)
            self.custom2 = CustomConvLayer(kof_path, tprl_path=None, in_channels=8, out_channels=1, padding=1, stride=1, kernel_idx=1, verbose=True)
            self.relu = nn.ReLU()

        def forward(self, x, epoch=0):
            x = self.custom1(x, epoch=epoch)
            x = self.relu(x)
            x = self.conv3d(x)
            x = self.relu(x)
            x = self.custom2(x, epoch=epoch)
            return x

    net = ExampleNet("test.kof", "full_featured.tprl")
    optimizer = optim.SGD(net.parameters(), lr=0.005, weight_decay=0.001)

    input_data = torch.rand(1, 1, 30, 20, 10)
    target = torch.rand(1, 1, 30, 20, 10)

    epochs = 10
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        output = net(input_data, epoch=epoch)
        loss = nn.MSELoss()(output, target)
        if torch.isnan(loss):
            print(f"NaN loss detected at epoch {epoch}, stopping training")
            break
        loss.backward()
        net.custom1.update_histories()
        net.custom2.update_histories()  # No-op if tprl_path=None
        optimizer.step()
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
