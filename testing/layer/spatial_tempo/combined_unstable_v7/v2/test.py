import torch
import torch.nn as nn
import torch.nn.functional as F
import sympy
from sympy import sympify, symbols
import numpy as np
import re
import itertools

def trim(s):
    return s.strip()

def load_kof(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    ndim = None
    taps = []
    global_op = 'x * w'
    global_in = 'x'
    global_k = 'w'
    reading_taps = False
    current_tap_index = -1

    for line in lines:
        s = trim(line)
        if not s or s.startswith('#'): continue
        if s.startswith('ndim:'):
            ndim = int(trim(s[5:]))
        elif s.startswith('taps:'):
            reading_taps = True
        elif reading_taps and (s[0].isdigit() or (s[0] == '-' and len(s) > 1 and s[1].isdigit()) or '.' in s):
            parts = re.split(r'\s+', s)
            offs = list(map(float, parts[:ndim]))
            w = float(parts[ndim])
            taps.append({'offsets': offs, 'weight': w, 'op': None, 'input_encode': None, 'kernel_encode': None})
            current_tap_index += 1
        elif s.startswith('operation:'):
            formula = trim(s[10:])
            if current_tap_index >= 0:
                taps[current_tap_index]['op'] = formula
            else:
                global_op = formula
        elif s.startswith('input_encode:'):
            formula = trim(s[13:])
            if current_tap_index >= 0:
                taps[current_tap_index]['input_encode'] = formula
            else:
                global_in = formula
        elif s.startswith('kernel_encode:'):
            formula = trim(s[14:])
            if current_tap_index >= 0:
                taps[current_tap_index]['kernel_encode'] = formula
            else:
                global_k = formula

    if ndim is None or not taps:
        raise ValueError("Invalid .kof file")

    return {'ndim': ndim, 'taps': taps, 'global_op': global_op, 'global_in': global_in, 'global_k': global_k}

def load_tprl(path):
    if not path:
        return None

    with open(path, 'r') as f:
        lines = [trim(line.strip()) for line in f.readlines() if line.strip() and not line.startswith('#')]

    section = None
    vars_list = []
    formula = None
    testing_lhs = None
    testing_rhs = None
    rel_op = None
    memory_formula = None
    param_names = []
    param_inits = []
    history_length = 5  # default

    for line in lines:
        if line.startswith('[') and line.endswith(']'):
            section = line[1:-1].lower()
            continue
        if section == 'vars':
            vars_list.extend([v.strip() for v in line.split() if v.strip()])
        elif section == 'formula':
            formula = line
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

    if not formula or not vars_list:
        raise ValueError("Invalid .tprl file")

    return {
        'vars': vars_list,
        'formula': formula,
        'testing_lhs': testing_lhs,
        'testing_rhs': testing_rhs,
        'rel_op': rel_op,
        'memory_formula': memory_formula,
        'param_names': param_names,
        'param_inits': param_inits,
        'history_length': history_length
    }

class SymPyFunction(torch.autograd.Function):
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
    def __init__(self, kof_path, tprl_path=None, in_channels=1, out_channels=1, padding=0, stride=1):
        super().__init__()
        kof = load_kof(kof_path)
        self.ndim = kof['ndim']
        self.tap_offsets = torch.tensor([t['offsets'] for t in kof['taps']])
        self.need_interp = not all(float(o).is_integer() for tap in kof['taps'] for o in tap['offsets'])
        self.k_elems = len(kof['taps'])
        self.in_channels = in_channels
        self.out_channels = out_channels
        weights_init = torch.tensor([t['weight'] for t in kof['taps']])
        self.weights = nn.Parameter(weights_init.view(1, 1, -1).repeat(out_channels, in_channels, 1))

        if isinstance(padding, int):
            padding = [padding] * self.ndim
        self.padding = torch.tensor(padding, dtype=torch.long)

        if isinstance(stride, int):
            stride = [stride] * self.ndim
        self.stride = torch.tensor(stride, dtype=torch.long)

        x_sym = symbols('x')
        w_sym = symbols('w')
        d_syms = symbols([f'd{i}' for i in range(self.ndim)])

        self.op_funcs = []
        self.input_encode_funcs = []
        self.kernel_encode_funcs = []
        for tap in kof['taps']:
            op_expr = sympify(tap['op'] or kof['global_op'])
            in_expr = sympify(tap['input_encode'] or kof['global_in'])
            k_expr = sympify(tap['kernel_encode'] or kof['global_k'])
            self.op_funcs.append(sympy.lambdify([x_sym, w_sym], op_expr, 'numpy'))
            self.input_encode_funcs.append(sympy.lambdify([x_sym], in_expr, 'numpy'))
            self.kernel_encode_funcs.append(sympy.lambdify([w_sym] + d_syms, k_expr, 'numpy'))

        self.temporal = False
        if tprl_path:
            self.temporal = True
            tprl = load_tprl(tprl_path)
            self.t_vars = tprl['vars']
            self.t_syms = symbols(self.t_vars)
            formula_expr = sympify(tprl['formula'])
            self.formula_func = sympy.lambdify(self.t_syms, formula_expr, 'numpy')
            self.history_length = tprl['history_length']
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
        args_dict = {'w': w_tensor, 'g': g_tensor, 'm': m_tensor}
        args = []
        k_idx = 0
        for v in self.t_vars:
            if v in args_dict:
                args.append(args_dict[v])
            else:
                args.append(self.k[k_idx].expand_as(w_tensor))
                k_idx += 1
        return args

    def forward(self, input):
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
        kernel_spans = max_off - min_off + 1
        out_min = -min_off
        out_max = padded_shape - 1 - max_off
        out_shape_no_stride = out_max - out_min + 1
        out_shape_no_stride = torch.clamp(out_shape_no_stride, min=0)
        if torch.any(out_shape_no_stride <= 0):
            return torch.empty((batch_size, self.out_channels, *( [0] * self.ndim )), device=device, dtype=input.dtype)

        ranges = [torch.arange(0, out_shape_no_stride[d].item(), self.stride[d].item(), device=device) for d in range(self.ndim)]
        coords = torch.cartesian_prod(*ranges)
        out_spatial_total = coords.shape[0]
        out_shape = torch.tensor([len(r) for r in ranges], device=device)

        input_coords = coords[:, None, :] + out_min[None, None, :] + self.tap_offsets[None, :, :].to(device)

        # Get x
        if not self.need_interp:
            input_coords = input_coords.long()
            spatial_strides = torch.ones(self.ndim, dtype=torch.long, device=device)
            if self.ndim > 0:
                spatial_strides[-1] = 1
                for i in range(self.ndim - 2, -1, -1):
                    spatial_strides[i] = spatial_strides[i + 1] * padded_shape[i + 1]
            in_flat_per_channel = torch.sum(input_coords * spatial_strides[None, None, :], dim=-1)  # [out_s, k]
            prod_spatial = padded_shape.prod().item()
            batch_offsets = torch.arange(batch_size, device=device)[:, None, None, None] * (self.in_channels * prod_spatial)
            channel_offsets = torch.arange(self.in_channels, device=device)[None, None, None, :] * prod_spatial
            spatial_offsets = in_flat_per_channel.view(out_spatial_total, self.k_elems, 1)
            idx = batch_offsets + channel_offsets + spatial_offsets
            idx = idx.view(-1)
            full_flat = input_padded.view(-1)
            x_flat = full_flat[idx]
            x = x_flat.view(batch_size, out_spatial_total, self.k_elems, self.in_channels)
        else:
            # Multilinear interpolation
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
            x = torch.zeros(batch_size, out_spatial_total, self.k_elems, self.in_channels, device=device, dtype=input.dtype)
            for corner in itertools.product([0, 1], repeat=self.ndim):
                corner_off = torch.tensor(corner, dtype=torch.long, device=device)
                corner_idx = floor + corner_off[None, :]
                corner_idx = torch.clamp(corner_idx, min=torch.zeros_like(corner_idx), max=padded_shape - 1)
                weights = torch.prod(torch.where(corner_off[None, :] == 0, 1 - frac, frac), dim=-1)  # [num_points]
                flat_idx = torch.sum(corner_idx * spatial_strides[None, :], dim=-1)  # [num_points]
                full_idx = (torch.arange(batch_size * self.in_channels, device=device)[:, None] * prod_spatial + flat_idx[None, :]).view(-1)
                full_idx = torch.clamp(full_idx, max=batch_size * self.in_channels * prod_spatial - 1)
                gathered_flat = input_padded.view(-1)[full_idx]
                full_gathered = gathered_flat.view(batch_size, self.in_channels, num_points)
                weighted = full_gathered * weights[None, None, :]
                x += weighted.permute(0, 2, 1).view(batch_size, out_spatial_total, self.k_elems, self.in_channels)

        # Temporal adjustment
        weights = self.weights
        if self.temporal:
            w_tensor = self.weights
            g_tensor = self.grad_magnitudes
            m_tensor = self.memory_state
            args = self._get_temporal_args(w_tensor, g_tensor, m_tensor)
            adjusted = SymPyFunction.apply(w_tensor, self.formula_func, *args[1:])

            mask = torch.ones_like(w_tensor, dtype=torch.bool)
            if self.testing_lhs_func:
                lhs = SymPyFunction.apply(w_tensor, self.testing_lhs_func, *args[1:])
                rhs = SymPyFunction.apply(w_tensor, self.testing_rhs_func, *args[1:])
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

            weights = torch.where(mask, adjusted, self.weights)

        # Compute w_enc
        w_enc = torch.zeros(self.out_channels, self.in_channels, self.k_elems, device=device, dtype=torch.float32)
        for t in range(self.k_elems):
            d_args = self.tap_offsets[t].tolist()
            w_enc[:, :, t] = SymPyFunction.apply(weights[:, :, t], self.kernel_encode_funcs[t], *d_args)

        # Compute x_enc
        x_enc = torch.zeros_like(x)
        for t in range(self.k_elems):
            x_enc[:, :, t, :] = SymPyFunction.apply(x[:, :, t, :], self.input_encode_funcs[t])

        # Compute output
        output_flat = torch.zeros(batch_size, out_spatial_total, self.out_channels, device=device, dtype=torch.float32)
        for t in range(self.k_elems):
            x_enc_t = x_enc[:, :, t, :]  # [b, os, in]
            w_enc_t = w_enc[:, :, t]  # [out, in]
            x_exp = x_enc_t[:, :, None, :]  # [b, os, out, in]
            w_exp = w_enc_t[None, None, :, :]  # [b, os, out, in]
            combined_t = SymPyFunction.apply(x_exp, self.op_funcs[t], w_exp)
            output_flat += combined_t.sum(dim=-1)  # [b, os, out]

        output = output_flat.view(batch_size, self.out_channels, *out_shape.tolist())
        return output

    def update_histories(self):
        if not self.temporal or self.weights.grad is None:
            return

        grads = self.weights.grad.abs().detach()
        self.grad_magnitudes = (self.grad_magnitudes * (self.history_length - 1) + grads) / self.history_length

        self.weight_history = torch.roll(self.weight_history, shifts=1, dims=0)
        self.gradient_history = torch.roll(self.gradient_history, shifts=1, dims=0)
        self.weight_history[0] = self.weights.detach()
        self.gradient_history[0] = grads

        if self.history_count < self.history_length:
            self.history_count += 1

        if self.memory_func:
            w_tensor = self.weights.detach()
            g_tensor = self.grad_magnitudes
            m_tensor = self.memory_state
            args = self._get_temporal_args(w_tensor, g_tensor, m_tensor)
            self.memory_state = SymPyFunction.apply(w_tensor, self.memory_func, *args[1:])

# Example usage for training
if __name__ == "__main__":
    import torch.optim as optim

    # Assume test.kof and temporal.tprl files exist
    layer1 = CustomConvLayer("test.kof", "temporal.tprl", in_channels=1, out_channels=1, padding=0, stride=1)
    layer2 = CustomConvLayer("test.kof", "temporal.tprl", in_channels=1, out_channels=1, padding=0, stride=1)

    in_shape = [30, 20, 10]
    input_data = torch.arange(torch.prod(torch.tensor(in_shape)).item(), dtype=torch.float32).view(1, 1, *in_shape) % 17 / 16.0
    target = input_data + 0.5

    optimizer = optim.SGD([
        {'params': layer1.parameters(), 'lr': 0.01, 'weight_decay': 0.001},
        {'params': layer2.parameters(), 'lr': 0.01, 'weight_decay': 0.001}
    ], lr=0.01)

    epochs = 30
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        output1 = layer1(input_data)
        output2 = layer2(output1)
        loss = nn.MSELoss()(output2, target)
        loss.backward()
        layer1.update_histories()
        layer2.update_histories()
        optimizer.step()

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"epoch {epoch}: loss={loss.item():.6f}")
