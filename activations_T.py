from typing import Optional
import math
from scipy.special import erf
from tensor_T import Tensor
from autograd_T import Function
import arrays
from module_list_T import Module

class ReLUFunction(Function):
    @staticmethod
    def forward(ctx, x):
        if hasattr(x, 'data'):
            x_data = x.data
        else:
            x_asarray = arrays.asarray(x)
            x_data = x_asarray.data
        
        x_array = arrays.Array(x_data)
        zeros_array = arrays.zeros_like(x_array)
        max_result = arrays.maximum(x_array, zeros_array)
        result_asarray = arrays.asarray(max_result.data)
        result = result_asarray.data
        ctx.save_for_backward(x)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        if hasattr(x, 'data'):
            x_data = x.data
        else:
            x_asarray = arrays.asarray(x)
            x_data = x_asarray.data
            
        if hasattr(grad_output, 'data'):
            grad_data = grad_output.data
        else:
            grad_asarray = arrays.asarray(grad_output)
            grad_data = grad_asarray.data
            
        grad_x = grad_data * (x_data > 0)
        return Tensor(grad_x)

class ReLU(Module):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        self._input = None

    def forward(self, x: Tensor) -> Tensor:
        if self.inplace:
            if not isinstance(x, Tensor):
                raise TypeError("In-place operations require Tensor input")
            if not x.requires_grad:
                x.requires_grad = True
            self._input = x
            max_result = arrays.maximum(x.data, 0)
            max_asarray = arrays.asarray(max_result.data)
            x.data = max_asarray.data
            return x
        else:
            return ReLUFunction.apply(x)

    def __str__(self) -> str:
        return f"ReLU(inplace={self.inplace})"

    def backward(self, grad_output: Tensor) -> Tensor:
        if self.inplace:
            # 使用保存的输入计算梯度
            grad_input = grad_output.data * (self._input.data > 0)
            return Tensor(grad_input, requires_grad=grad_output.requires_grad)
        else:
            # ReLUFunction 会处理梯度计算
            return grad_output

    def __call__(self, input: Tensor) -> Tensor:
        return self.forward(input)

class Sigmoid(Module):
    def __init__(self) -> None:
        super().__init__()
        """
        Sigmoid 激活层 (带反向传播支持)
        Forward:  output = 1 / (1 + exp(-input))
        Backward: grad_input = grad_output * output * (1 - output)
        """
        self._output_cache: Optional[Tensor] = None

    def forward(self, input: Tensor) -> Tensor:
        self._output_cache = Tensor(1) / (Tensor(1) + (-input).exp())
        return self._output_cache

    def __call__(self, input: Tensor) -> Tensor:
        return self.forward(input)

class GELUFunction(Function):
    @staticmethod
    def forward(ctx, x, approximate: str = 'tanh'):
        ctx.save_for_backward(x)
        ctx.metadata['approximate'] = approximate
        
        if hasattr(x, 'data'):
            x_data = x.data
        else:
            x_asarray = arrays.asarray(x)
            x_data = x_asarray.data
        
        if approximate == 'tanh':
            sqrt_2_over_pi = math.sqrt(2 / math.pi)
            x_cubed = x_data * x_data * x_data
            inner = sqrt_2_over_pi * (x_data + 0.044715 * x_cubed)
            inner_array = arrays.Array(inner)
            tanh_result = arrays.tanh(inner_array)
            tanh_data = tanh_result.data
            
            # 直接计算 0.5 * x * (1 + tanh(inner))，避免数组运算
            if isinstance(tanh_data, list):
                if isinstance(tanh_data[0], list):
                    # 2D 数据
                    result = [[0.5 * x_data[i][j] * (1 + tanh_data[i][j]) 
                              for j in range(len(tanh_data[i]))] 
                             for i in range(len(tanh_data))]
                else:
                    # 1D 数据
                    result = [0.5 * x_data[i] * (1 + tanh_data[i]) 
                             for i in range(len(tanh_data))]
            else:
                # 标量
                result = 0.5 * x_data * (1 + tanh_data)
            
            return result
        else:
            x_over_sqrt2 = x_data / math.sqrt(2)
            return 0.5 * x_data * (1 + erf(x_over_sqrt2))
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        approximate = ctx.metadata['approximate']
        
        if hasattr(x, 'data'):
            x_data = x.data
        else:
            x_asarray = arrays.asarray(x)
            x_data = x_asarray.data
            
        if hasattr(grad_output, 'data'):
            grad_data = grad_output.data
        else:
            grad_asarray = arrays.asarray(grad_output)
            grad_data = grad_asarray.data
        
        if approximate == 'tanh':
            sqrt_2_over_pi = math.sqrt(2 / math.pi)
            x_sq = x_data * x_data
            x_cubed = x_data * x_sq
            inner = sqrt_2_over_pi * (x_data + 0.044715 * x_cubed)
            inner_array = arrays.Array(inner)
            tanh_result = arrays.tanh(inner_array)
            tanh_inner = tanh_result.data
            
            # 导数计算
            deriv = 0.5 * (1 + tanh_inner) + \
                   0.5 * x_data * (1 - tanh_inner * tanh_inner) * \
                   sqrt_2_over_pi * (1 + 3 * 0.044715 * x_sq)
        else:
            # 精确GELU的导数
            sqrt_2 = math.sqrt(2)
            x_over_sqrt2 = x_data / sqrt_2
            erf_term = erf(x_over_sqrt2)
            x_squared = x_data * x_data
            half_neg_x_squared = -0.5 * x_squared
            exp_input = arrays.Array(half_neg_x_squared)
            exp_result = arrays.exp(exp_input)
            exp_term = exp_result.data
            deriv = 0.5 * (1 + erf_term) + x_data * exp_term / math.sqrt(2 * math.pi)
        
        return Tensor(grad_data * deriv)

class GELU(Module):
    def __init__(self, approximate: str = 'tanh') -> None:
        super().__init__()
        """
        GELU 激活层 (带反向传播支持)
        Forward (tanh近似):
            0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        Backward:
            grad_input = grad_output * Δ (详见代码实现)
        """
        assert approximate in ['tanh', None], "approximate must be 'tanh' or None"
        self.approximate = approximate

    def forward(self, input: Tensor) -> Tensor:
        return GELUFunction.apply(input, self.approximate)

    def __call__(self, input: Tensor) -> Tensor:
        return self.forward(input)

class LeakyReLU(Module):
    def __init__(self, negative_slope: float = 0.01, inplace: bool = False) -> None:
        super().__init__()
        """
        泄漏ReLU层 (带反向传播支持)
        Forward:  output = max(0, x) + negative_slope * min(0, x)
        Backward: grad_input = grad_output * (x > 0) + negative_slope * grad_output * (x <= 0)
        """
        self.negative_slope = negative_slope
        self.inplace = inplace
        self._input_cache: Optional[Tensor] = None

    def forward(self, input: Tensor) -> Tensor:
        self._input_cache = input if not self.inplace else input.clone()
        
        if self.inplace:
            input.relu_()
            input.add_((self._input_cache < 0).type_as(input) * self.negative_slope * self._input_cache)
            return input
        else:
            pos = input.relu()
            neg = (input < 0).type_as(input) * self.negative_slope * input
            return pos + neg

    def __call__(self, input: Tensor) -> Tensor:
        return self.forward(input)

class Swish(Module):
    def __init__(self) -> None:
        super().__init__()
        """
        Swish 激活层 (带反向传播支持)
        Forward:  output = x * sigmoid(x)
        Backward: grad_input = grad_output * (swish(x) + sigmoid(x)*(1-swish(x)))
        """
        self._input_cache: Optional[Tensor] = None
        self._sigmoid_cache: Optional[Tensor] = None

    def forward(self, input: Tensor) -> Tensor:
        self._input_cache = input.clone()
        self._sigmoid_cache = Tensor(1) / (Tensor(1) + (-input).exp())
        return input * self._sigmoid_cache

    def __call__(self, input: Tensor) -> Tensor:
        return self.forward(input)

# 函数式接口 (带自动微分支持)
def relu(input: Tensor, inplace: bool = False) -> Tensor:
    return ReLU(inplace=inplace).forward(input)

def gelu(input: Tensor, approximate: str = 'tanh') -> Tensor:
    return GELU(approximate=approximate).forward(input)

def leaky_relu(input: Tensor, negative_slope: float = 0.01, inplace: bool = False) -> Tensor:
    return LeakyReLU(negative_slope=negative_slope, inplace=inplace).forward(input)
