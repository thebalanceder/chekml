# Custom Convolutional Layer with Kernel Offset Files (.kof) and Temporal Processing Rules (.tprl)

This repository implements a **custom convolutional layer** (`CustomConvLayer`) in PyTorch, designed for flexible, multi-kernel convolutions with support for **fractional offsets**, **symbolic encoding functions**, and **temporal adaptation** during training.  

The layer configuration is driven by two file formats:  

- **`.kof` (Kernel Offset File):** Defines convolution kernels, taps (offsets and weights), and optional encoding/operation formulas.  
- **`.tprl` (Temporal Processing Rule Language):** Defines adaptive weight update rules during training, based on epochs or gradient statistics.  

Features:
- Supports **3D convolutions** (extendable to other dimensions).  
- Symbolic mathematics via **SymPy** for custom input/kernel encodings.  
- Optimized execution for discrete offsets (`torch.nn.functional.conv3d`).  
- **Interpolation support** for fractional offsets.  

---

## .kof File Format

The `.kof` file specifies dimensionality, kernels, taps (offsets and weights), and optional formulas.  
- Lines starting with `#` are comments.  
- Empty lines are ignored.  

### Required Parameter
- **ndim:** Integer, number of spatial dimensions (e.g., `ndim: 3` for 3D convolution).  

### Kernel Sections
- **kernel:** Marks the start of a new kernel definition (e.g., `kernel: 0`).  
- Each kernel can define its own **taps** and optional overrides for formulas.  

### Taps
- Defined after the `taps:` keyword.  
- Each tap is a list of `ndim` offsets (floats, fractional allowed) followed by a weight.  
- Example (for `ndim=3`):
  `0.5 0.3 0.2 1.0`
→ Offsets `[0.5, 0.3, 0.2]`, weight = `1.0`.  

- Offsets can be positive or negative. Padding is auto-adjusted if needed.  
- Optionally, taps can be **channel-specific** using `channel: <id>` before `taps:`.  

### Formulas
Formulas use **SymPy** syntax and can appear at file, kernel, or tap level:  

- **operation:** Combination function (default: `x * w`).  
- Variables: `x` (input), `w` (weight).  
- **input_encode:** Input encoding (default: `x`).  
- Variable: `x`.  
- **kernel_encode:** Kernel encoding (default: `w`).  
- Variables: `w`, `d0 ... d<ndim-1>` (tap offsets).  

**Precedence:** per-tap > kernel-global > file-global.  

#### Example `.kof`
```text
ndim: 3

kernel: 0
taps:
0.5 0.3 0.2 1.0
operation: x * w
input_encode: log(1 + x)
kernel_encode: w * (1 + d0*d0 + d2*d0 + d1*d1 + d2*d2)

kernel: 1
taps:
0 0 0 1.0
1 0 0 0.5
operation: x * w
input_encode: x
kernel_encode: w
```

## .tprl File Format

The `.tprl` file enables **temporal adaptation of weights** during training.  
Sections are denoted by `[section_name]`.

### Sections

- **[vars]**: Variables used in formulas.  
  Example:  
  `w k g m`

Built-in historical variables: `wh0`, `gh0`, etc.

- **[formula]**: One or more **SymPy formulas** for weight update.  
Example:  
`w * (1 + k * g)`

- **[conditions]**: Maps **conditions → formula index**.  
Example:  
`epoch > 10 : 1`

- **[params]**: Initialization and hyperparameters.  
Example:  
`k_init 0.1
history_length 5
grad_threshold 1e-4
adaptive_history true`

- **[testing]** *(optional)*: Weight masking rule.  
Example:  
`w + k > g`

- **[memory]** *(optional)*: Memory state update rule.  
Example:  
`m + 0.1 * g`

#### Example `.tprl`
```ini
[vars]
w k g m

[formula]
w * (1 + k * g)
w * (1 + k * m + 0.5 * (wh0 + gh0))

[conditions]
epoch > 10 : 1
grad_mean < 0.01 : 1

[params]
k_init 0.1
history_length 2
grad_threshold 1e-4
adaptive_history true
```

# Methods
- `forward(input, epoch=0)`
  Performs convolution.
  - Input shape: `[batch, in_channels, *spatial(ndim)]`
  - `epoch` is used for temporal rules.

- `update_histories()`
  Updates weight/gradient history (if temporal mode enabled).

  ## CustomConvLayer Parameters

The `CustomConvLayer` extends `torch.nn.Module`.

| Parameter     | Type         | Default | Description |
|---------------|-------------|---------|-------------|
| `kof_path`    | `str`       | –       | Path to `.kof` file (**required**). |
| `tprl_path`   | `str`       | `None`  | Path to `.tprl` file (optional). |
| `in_channels` | `int`       | `1`     | Number of input channels. |
| `out_channels`| `int`       | `1`     | Number of output channels. |
| `padding`     | `int or list` | `0`   | Padding (auto-adjusted if insufficient). |
| `stride`      | `int or list` | `1`   | Stride. |
| `kernel_idx`  | `int`       | `0`     | Index of kernel from `.kof`. |
| `verbose`     | `bool`      | `True`  | Print debug/warnings. |

- Example Usage:

```python
# example_module.py: Example module integrating CustomConvLayer with standard PyTorch layers

import torch
import torch.nn as nn
import torch.optim as optim

from chekml.layer.spatial_tempo.test import CustomConvLayer

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

if __name__ == "__main__":
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
        net.custom2.update_histories()
        optimizer.step()
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
```
