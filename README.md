# APTx Activation Function

This repository offers a `Python package` for the `PyTorch` implementation of the APTx activation function, as introduced in the paper "APTx: Better Activation Function than MISH, SWISH, and ReLU's Variants used in Deep Learning."

APTx (Alpha Plus Tanh Times) is a novel activation function designed for computational efficiency in deep learning. It enhances training performance and inference speed, making it particularly suitable for low-end hardware such as IoT devices. Notably, APTx provides flexibility by allowing users to either use its default parameter values or optimize them as trainable parameters during training.

**Paper Title**: APTx: Better Activation Function than MISH, SWISH, and ReLU's Variants used in Deep Learning

**Author**: [Ravin Kumar](https://mr-ravin.github.io)

**Publication**: 5th July, 2022

**Published Paper**: [click here](https://www.svedbergopen.com/files/1666089614_(5)_IJAIML20221791212945BU5_(p_56-61).pdf)

**Doi**: [DOI Link of Paper](https://doi.org/10.51483/IJAIML.2.2.2022.56-61)

**Other Sources**:
- [Arxiv.org](https://arxiv.org/abs/2209.06119)
- [Research Gate](https://www.researchgate.net/publication/364734055_APTx_Better_Activation_Function_than_MISH_SWISH_and_ReLU's_Variants_used_in_Deep_Learning), [Research Gate - Preprint](https://www.researchgate.net/publication/383019098_APTx_better_activation_function_than_MISH_SWISH_and_ReLU's_variants_used_in_deep_learning)
- [Osf.io](https://osf.io/3249p_v1/)
- [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4346892)
- [Internet Archive](https://archive.org/details/aptx-activation-function-in-deep-learning-published-paper), [Internet Archive - Preprint](https://archive.org/details/aptx-activation-function)

#### Github Repositories: 
- **Github Repository** (Python Package- Pytorch Implementation): [Python Package](https://github.com/mr-ravin/aptx_activation)

#### Cite Paper as:
```
Ravin Kumar (2022). APTx: Better Activation Function than MISH, SWISH, and ReLU’s Variants used in Deep Learning. International Journal of Artificial Intelligence and Machine Learning, 2(2), 56-61. doi: 10.51483/IJAIML.2.2.2022.56-61
```
---
## Mathematical Definition
The APTx activation function is defined as:

$\mathrm{APTx}(x) = (\alpha + \tanh(\beta x)) \cdot \gamma x$

where:
- $\alpha$ controls the baseline shift (default: 1.0)
- $\beta$ scales the input inside the tanh function (default: 1.0)
- $\gamma$ controls the output amplitude (default: 0.5)



At $\alpha$ = 1, $\beta$ = ½ and $\gamma$ = ½ APTx closely maps the negative domain part of the APTx with the negative part of MISH. When $\alpha$ = 1 , $\beta$ = 1 and $\gamma$ = ½ the positive domain part of APTx closely maps with the
positive part of MISH. 

So, we can use $\alpha$ = 1 , $\beta$ = ½ and $\gamma$ = ½ values for the negative part, and $\alpha$ = 1 , $\beta$ = 1 and $\gamma$ = ½ for the positive part in case we want to closely approximate the MISH activation function.

Interestingly, APTx function with parameters $\alpha$ = 1 , $\beta$ = ½ and $\gamma$ = ½ behaves like the SWISH(x, 1) activation function, and APTx with $\alpha$ = 1 , $\beta$ = 1 and $\gamma$ = ½ behaves like SWISH(x, 2).

---
## 📥 Installation
```bash
pip install aptx_activation
```
or,

```bash
pip install git+https://github.com/mr-ravin/aptx_activation.git
```

### 📌 **Dependencies:**
- Python >= 3.7
- Pytorch >= 1.8.0

## Usage

#### 1. APTx with parameters values

On Default Device:
```python
import torch
from aptx_activation import APTx

# Example Usage
aptx = APTx(alpha=1.0, beta=1.0, gamma=0.5) # default values in APTx
tensor = torch.randn(5)
output = aptx(tensor)
print(output)
```

On GPU Device:
```python
import torch
from aptx_activation import APTx

# Example Usage
aptx = APTx(alpha=1.0, beta=1.0, gamma=0.5).to("cuda") # default values in APTx
tensor = torch.randn(5).to("cuda")
output = aptx(tensor)
print(output)
```
##### 2. APTx with parameters - Similar to SWISH

```python
import torch
from aptx_activation import APTx

# Example Usage
aptx = APTx(alpha=1.0, beta=0.5, gamma=0.5) # Behaves like SWISH(x, 1)
tensor = torch.randn(5)
output = aptx(tensor)
print(output)
```

##### 3. APTx with trainable parameters
APTx allows for trainable parameters to adapt dynamically during training:
```python
from aptx_activation import APTx
aptx = APTx(trainable=True)  # Learnable α, β, and γ
```

---
## Key Benefits of APTx
- **Efficient Computation**: Requires fewer mathematical operations compared to MISH and SWISH.
- **Faster Training**: The reduced complexity speeds up both forward and backward propagation.
- **Lower Hardware Requirements**: Optimized for edge devices and low-end computing hardware.
- **Parameter Flexibility - SWISH**: By setting $\alpha = 1$, $\beta = 0.5$, and $\gamma = 0.5$, APTx can replicate the SWISH(x, 1) activation function.
- **Parameter Flexibility - MISH**:
  - By setting $\alpha = 1$, $\beta = 0.5$, and $\gamma = 0.5$, APTx can replicate the `negative domain` part of MISH activation function.
  - By setting $\alpha = 1$, $\beta = 1$, and $\gamma = 0.5$, APTx can replicate the `positive domain` part of MISH activation function.

## Comparison of APTx with MISH, SWISH, and ReLU's variants
- SWISH generally outperforms ReLU (and its variants) in deeper networks because it is smooth and non-monotonic, allowing better gradient flow.
- MISH vs SWISH: 
  - MISH is smoother than SWISH, helping gradient flow.
  - MISH retains more information during negative input values.
  - MISH requires more **computation**.
- **APTx offers similar performance to MISH but with significantly lower computation costs**, making it ideal for resource-constrained environments.

### 📜 Copyright License
```python
Copyright (c) 2025 Ravin Kumar
Website: https://mr-ravin.github.io

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation 
files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, 
modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the 
Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```
