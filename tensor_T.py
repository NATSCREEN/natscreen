# local_torch/_tensor/tensor.py

import arrays
from typing import Optional, List, Union, Any, Tuple  
import weakref
import math
import strong_two

class Tensor:
    def __init__(
        self,
        data: Union[Any, Any, float, int, 'Tensor', 'arrays.Array'],
        requires_grad: bool = False,
        dtype: Optional[type] = None,
        _grad_fn: Optional[type] = None,  # 修改为 type 而不是 'Function'
        _children: Optional[List['Tensor']] = None
    ):
        # 数据标准化
        if isinstance(data, (list, tuple)):
            # 简化的数据清理，避免复杂递归
            def clean_data(d):
                if isinstance(d, list):
                    if len(d) == 1 and isinstance(d[0], (int, float)):
                        return float(d[0])
                    else:
                        return [clean_data(item) for item in d]
                elif hasattr(d, 'shape') and hasattr(d, 'dtype'):
                    # 检查是否为numpy标量（有shape和dtype但shape为()）
                    if hasattr(d, 'shape') and d.shape == ():
                        # numpy标量，直接转换为Python float
                        return float(d)
                    elif hasattr(d, '__iter__') and not isinstance(d, str):
                        # 可迭代对象（但不是字符串）
                        return [clean_data(item) for item in d]
                    else:
                        # 其他有shape和dtype但不可迭代的对象，尝试转换为float
                        try:
                            return float(d)
                        except (ValueError, TypeError):
                            return d
                elif hasattr(d, 'data') and hasattr(d, 'shape'):  # arrays.Array对象
                    # 直接提取数据，避免嵌套
                    if isinstance(d.data, list) and len(d.data) == 1:
                        return float(d.data[0])
                    elif isinstance(d.data, (int, float)):
                        return float(d.data)
                    else:
                        return d.data
                elif isinstance(d, (int, float)):
                    return float(d)
                else:
                    # 尝试转换为float，如果失败则返回原值
                    try:
                        return float(d)
                    except (ValueError, TypeError):
                        return d
            
            cleaned_data = clean_data(data)
            data_array = arrays.asarray_numpy_compatible(cleaned_data, dtype=dtype if dtype else float)
            self._data = data_array.data
        elif isinstance(data, (float, int)):
            data_array = arrays.asarray_numpy_compatible([data], dtype=dtype if dtype else float)
            self._data = data_array.data
        elif isinstance(data, Tensor):
            # 如果传入的是Tensor，取其底层数据
            if dtype:
                self._data = data.data.astype(dtype)
            else:
                data_array = arrays.asarray_numpy_compatible(data.data)
                self._data = data_array.data
        elif isinstance(data, arrays.Array):
            if len(data.shape) > 1:
                # 多维数组，需要保持形状
                data_array = arrays.asarray_numpy_compatible(data.data, dtype=dtype if dtype else float)
                self._data = data_array.data.reshape(data.shape)
            else:
                # 一维数组
                data_array = arrays.asarray_numpy_compatible(data.data, dtype=dtype if dtype else float)
                self._data = data_array.data
        else:
            if hasattr(data, 'copy') and hasattr(data, 'shape') and hasattr(data, 'dtype'):
                self._data = data.astype(dtype) if dtype else data.copy()
            else:
                # 最后的后备方案
                try:
                    data_array = arrays.asarray_numpy_compatible(data, dtype=dtype if dtype else float)
                    self._data = data_array.data
                except (ValueError, TypeError):
                    # 如果直接转换失败，尝试通过arrays.array
                    array_result = arrays.asarray_numpy_compatible(data, dtype=dtype if dtype else float)
                    self._data = array_result.data

        # 梯度属性
        self.requires_grad = bool(requires_grad)
        self.grad: Optional[Tensor] = None
        
        # 计算图构建
        self._grad_fn = _grad_fn
        self._children = _children if _children is not None else []
        self._ctx: Optional[Any] = None
        self._output_refs: List[weakref.ref] = []

        # 元数据
        self.shape = self._data.shape
        self.dtype = self._data.dtype
        self.device = 'cpu'

        self._id = id(self)  # 添加一个唯一标识符

    @classmethod
    def empty(cls, *shape: int, dtype: type = float, requires_grad: bool = False) -> 'Tensor':
        """创建未初始化的张量（类似torch.empty）"""
        empty_array = arrays.empty(shape, dtype=dtype)
        data_array = arrays.asarray_numpy_compatible(empty_array.data)
        return cls(data_array.data.reshape(shape), requires_grad=requires_grad)
    
    @classmethod
    def zeros(cls, *shape, requires_grad=False):
        """创建全零张量"""
        return cls([[0.0]*shape[1] for _ in range(shape[0])], requires_grad=requires_grad)
    
    @classmethod
    def ones(cls, *shape, requires_grad=False):
        """创建全一张量"""
        return cls([[1.0]*shape[1] for _ in range(shape[0])], requires_grad=requires_grad)
    
    @classmethod
    def randn(cls, *shape, requires_grad=False):
        """创建随机张量(简单实现)"""
        import math
        import time
        seed = int(time.time() * 1000) % 1000000
        def simple_random():
            nonlocal seed
            seed = (seed * 9301 + 49297) % 233280
            return seed / 233280.0
        
        # 简单的正态分布近似
        data = []
        for _ in range(shape[0]):
            row = []
            for _ in range(shape[1]):
                # Box-Muller变换生成正态随机数
                u1 = simple_random()
                u2 = simple_random()
                z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2)
                row.append(z0)
            data.append(row)
        return cls(data, requires_grad=requires_grad)
    
    # === 核心属性 ===
    @property
    def data(self) -> Any:
        """获取底层numpy数组"""
        return self._data

    @data.setter
    def data(self, value: Any) -> None:
        """设置底层numpy数组"""
        self._data = value

    @property
    def ndim(self) -> int:
        """张量维度数"""
        return len(self.shape)

    @property
    def size(self) -> int:
        """元素总数"""
        return self._data.size

    # === 类型转换 ===
    def float(self) -> 'Tensor':
        """转换为float32类型"""
        return self._apply_unary_op(lambda x: x.astype(float))

    def long(self) -> 'Tensor':
        """转换为int64类型"""
        return self._apply_unary_op(lambda x: x.astype(arrays.int64))

    def int(self) -> 'Tensor':
        """转换为int32类型"""
        return self._apply_unary_op(lambda x: x.astype(arrays.int32))

    # === 数学运算 ===
    def __add__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        from operations_T import add
        return add(self, other if isinstance(other, Tensor) else Tensor(other))

    def __mul__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        from operations_T import mul
        return mul(self, other if isinstance(other, Tensor) else Tensor(other))

    def pow(self, exponent: Union['Tensor', float, int]) -> 'Tensor':
        """幂运算"""
        from operations_T import pow
        return pow(self, exponent if isinstance(exponent, Tensor) else Tensor(exponent))

    def __pow__(self, exponent: Union['Tensor', float, int]) -> 'Tensor':
        return self.pow(exponent)

    def sqrt(self) -> 'Tensor':
        """平方根运算"""
        return self.pow(0.5)

    def exp(self) -> 'Tensor':
        """指数运算"""
        from operations_T import exp
        return exp(self)

    # === 梯度计算 ===
    
    def backward(self, gradient: Optional['Tensor'] = None) -> None:
        """反向传播计算梯度
        
        Args:
            gradient: 上游梯度，如果为None则使用单位梯度
        """
        from autograd_T import backward as autograd_backward
        
        # 确保梯度是Tensor类型
        if gradient is not None and not isinstance(gradient, Tensor):
            gradient = Tensor(gradient)
            
        # 检查梯度是否有效
        if gradient is not None:
            grad_array = arrays.Array(gradient.data.flatten())
            isnan_result = arrays.isnan(grad_array)
            isinf_result = arrays.isinf(grad_array)
            if any(isnan_result.data) or any(isinf_result.data):
                print("Warning: Invalid gradient detected in backward pass")
                gradient = Tensor(arrays.zeros_like(gradient.data))
        
        # 如果当前张量没有梯度函数但需要梯度，使用默认梯度函数
        if (not hasattr(self, '_grad_fn') or self._grad_fn is None) and self.requires_grad:
            # 尝试建立计算图连接
            from operations_T import IdentityFunction
            self._grad_fn = IdentityFunction
            
        # 如果当前张量没有梯度函数，直接返回
        if not hasattr(self, '_grad_fn') or self._grad_fn is None:
            return
            
        # 调用autograd引擎执行反向传播
        autograd_backward(self, gradient)
        
        # 检查梯度是否正确计算
        if self.requires_grad and self.grad is None:
            self.grad = Tensor(arrays.zeros_like(self.data))
        elif self.requires_grad and self.grad is not None:
            # 检查梯度是否有效
            grad_array = arrays.Array(self.grad.data.flatten())
            isnan_result = arrays.isnan(grad_array)
            isinf_result = arrays.isinf(grad_array)
            if any(isnan_result.data) or any(isinf_result.data):
                print("Warning: Invalid gradient after backward pass")
                self.grad = Tensor(arrays.zeros_like(self.data))

    def dist_context(self):
        from autograd_T import _engine
        return getattr(_engine, '_dist_context', None)

    # === 内部方法 ===
    def _apply_unary_op(self, op) -> 'Tensor':
        """应用一元操作并保留梯度信息"""
        return Tensor(
            op(self._data),
            requires_grad=self.requires_grad,
            _grad_fn=self._grad_fn,
            _children=self._children
        )

    def __repr__(self) -> str:
        return f"Tensor({self._data}, shape={self.shape}, dtype={self.dtype}, " \
               f"requires_grad={self.requires_grad})"

    # === 就地操作 ===
    def zero_(self) -> 'Tensor':
        """原地置零"""
        self._data.fill(0)
        return self

    def fill_(self, value: float) -> 'Tensor':
        """原地填充"""
        self._data.fill(value)
        return self

    # === 比较运算 ===
    def __eq__(self, other):
        if not isinstance(other, Tensor):
            return False
        return self._id == other._id

    # === 其他关键运算 ===
    def abs(self) -> 'Tensor':
        """绝对值"""
        from operations_T import abs
        return abs(self)

    def maximum(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """最大值"""
        from operations_T import maximum
        return maximum(self, other if isinstance(other, Tensor) else Tensor(other))

    # === 链式方法 ===
    def clamp(self, min_val: float, max_val: float) -> 'Tensor':
        """裁剪到范围[min_val, max_val]"""
        from operations_T import where, maximum, minimum
        # 使用operations中的函数确保计算图连通
        min_tensor = min_val if isinstance(min_val, Tensor) else Tensor(min_val)
        max_tensor = max_val if isinstance(max_val, Tensor) else Tensor(max_val)
        
        # 先应用最小值裁剪，再应用最大值裁剪
        result = maximum(self, min_tensor)
        result = minimum(result, max_tensor)
        return result

    # === 张量操作 ===
    def detach(self) -> 'Tensor':
        """返回脱离计算图的新张量"""
        return Tensor(self._data.copy(), requires_grad=False)

    def cpu(self) -> 'Tensor':
        """返回CPU上的张量副本（兼容PyTorch API）"""
        # 在我们的实现中，所有张量都在CPU上，所以直接返回自身
        return self

    def clone(self) -> 'Tensor':
        """返回完全拷贝（保留计算图）"""
        return Tensor(
            self._data.copy(),
            requires_grad=self.requires_grad,
            _grad_fn=self._grad_fn,
            _children=self._children
        )

    # === 特殊方法 ===
    def __neg__(self) -> 'Tensor':
        return self.__mul__(-1)

    def __sub__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        from operations_T import sub
        return sub(self, other if isinstance(other, Tensor) else Tensor(other))

    def __rsub__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """处理右减操作，如 1 - tensor"""
        from operations_T import sub
        return sub(Tensor(other), self)

    def __truediv__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        from operations_T import div
        return div(self, other if isinstance(other, Tensor) else Tensor(other))

    # === 索引操作 ===
    def __getitem__(self, indices) -> 'Tensor':
        """索引操作 - 使用operations_T确保梯度传播"""
        # 不应该直接创建Tensor，而应该使用indexing操作来保持梯度链
        from operations_T import indexing
        try:
            return indexing(self, indices)
        except (ImportError, AttributeError):
            # 如果indexing不存在，至少不要复制错误的梯度信息
            result = Tensor(
                self._data[indices],
                requires_grad=self.requires_grad
            )
            # 建立正确的parent-child关系
            if self.requires_grad:
                result._children = [self]
                # 设置一个简单的梯度函数用于indexing
                from autograd_T import Function
                class IndexFunction(Function):
                    @staticmethod
                    def forward(ctx, x, indices):
                        ctx.indices = indices
                        ctx.input_shape = x.shape
                        return x._data[indices]
                    
                    @staticmethod
                    def backward(ctx, grad_output):
                        # 创建与输入相同形状的零梯度
                        import arrays
                        zeros_array = arrays.zeros(ctx.input_shape, dtype='float32')
                        grad_array = arrays.asarray_numpy_compatible(zeros_array.data, dtype=grad_output.data.dtype)
                        grad_input = grad_array.data
                        # 在对应位置填入梯度
                        grad_input[ctx.indices] = grad_output.data
                        return Tensor(grad_input)
                
                result._grad_fn = IndexFunction
            return result

    # === 矩阵操作 ===
    def t(self) -> 'Tensor':
        """矩阵转置"""
        from operations_T import transpose
        return transpose(self)

    def dot(self, other: 'Tensor') -> 'Tensor':
        """点积运算"""
        from operations_T import matmul
        return matmul(self, other)

    # === 归约操作 ===
    def mean(self, dim=None, keepdim=False):
        from operations_T import mean
        return mean(self, dim, keepdim)

    def sum(self, dim: Optional[int] = None, keepdim: bool = False) -> 'Tensor':
        """求和"""
        from operations_T import sum
        return sum(self, dim, keepdim)

    def max(self, dim=None, keepdim=False):
        if dim is None:
            data_array = arrays.Array(self._data)
            max_result = arrays.max(data_array)
            return Tensor(max_result)
        else:
            axis = dim
            data_array = arrays.Array(self._data)
            values = arrays.max(data_array, axis=axis, keepdims=keepdim)
            indices = arrays.argmax(data_array, axis=axis)
            if keepdim:
                indices_array = arrays.Array(indices)
                expanded_result = arrays.expand_dims(indices_array, axis=axis)
                indices = arrays.array(expanded_result.data)
            return Tensor(values), Tensor(indices)

    def var(self, dim: Optional[int] = None, keepdim: bool = False, unbiased: bool = True) -> 'Tensor':
        """计算指定维度的方差"""
        from operations_T import sum
        mean = self.mean(dim, keepdim)
        return sum((self - mean) ** 2, dim, keepdim) / (self._data.size if dim is None else self.shape[dim] - (1 if unbiased else 0))

    # === 激活函数 ===
    def relu(self) -> 'Tensor':
        """ReLU激活函数"""
        from operations_T import maximum
        return maximum(self, 0)

    def tanh(self) -> 'Tensor':
        """双曲正切函数"""
        from operations_T import exp
        return (exp(Tensor(2)*self) - Tensor(1)) / (exp(Tensor(2)*self) + Tensor(1))

    # === 就地操作 ===
    def zero_grad_(self) -> None:
        """清零梯度"""
        if self.grad is not None:
            self.grad.zero_()
        self.grad = None

    def clip_grad_norm_(self, max_norm: float) -> None:
        """裁剪梯度范数
        
        Args:
            max_norm: 最大范数
        """
        if self.grad is None:
            return
        
        grad_squared_array = arrays.Array(self.grad.data ** 2)
        sum_result = arrays.sum(grad_squared_array)
        sum_array = arrays.Array([sum_result])
        sqrt_result = arrays.sqrt(sum_array)
        total_norm = sqrt_result
        if total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-6)
            self.grad.data *= scale

    def uniform_(self, low: float = 0.0, high: float = 1.0) -> 'Tensor':
        """均匀分布初始化"""
        uniform_array = arrays.random.uniform(low, high, self.shape)
        data_array = arrays.asarray_numpy_compatible(uniform_array.data, dtype=self.dtype)
        self._data = data_array.data.reshape(self.shape)
        return self

    def normal_(self, mean: float = 0.0, std: float = 1.0) -> 'Tensor':
        """正态分布初始化"""
        normal_array = arrays.random.normal(mean, std, self.shape)
        data_array = arrays.asarray_numpy_compatible(normal_array.data, dtype=self.dtype)
        self._data = data_array.data.reshape(self.shape)
        return self

    # === 类型转换 ===
    def numpy(self) -> Any:
        """转换为numpy数组（脱离计算图）"""
        return self._data.copy()

    def item(self) -> Union[float, int]:
        """提取标量值"""
        if self.size != 1:
            raise ValueError("只能从单元素张量提取值")
        return float(self._data) if strong_two.issubdtype(self.dtype, strong_two.floating) else int(self._data)

    # === 打印优化 ===
    def __array__(self, dtype=None) -> Any:
        """支持numpy的数组接口
        
        Args:
            dtype: numpy可能传递的数据类型参数（可选）
        """
        if dtype is not None:
            return self._data.astype(dtype)
        return self._data

    def __str__(self) -> str:
        return f"Tensor({self._data}, requires_grad={self.requires_grad})"

    def isnan(self) -> 'Tensor':
        """检查是否为NaN"""
        data_array = arrays.Array(self._data)
        isnan_result = arrays.isnan(data_array)
        return Tensor(arrays.array(isnan_result.data))

    def isinf(self) -> 'Tensor':
        """检查是否为无穷大"""
        data_array = arrays.Array(self._data)
        isinf_result = arrays.isinf(data_array)
        return Tensor(arrays.array(isinf_result.data))

    def any(self) -> bool:
        """检查是否有任何True值"""
        data_array = arrays.Array(self._data)
        any_result = arrays.any(data_array)
        return bool(any_result)

    def clamp_min(self, min_val: float) -> 'Tensor':
        """裁剪到最小值"""
        from operations_T import maximum
        return maximum(self, min_val)

    def clamp_min_(self, min_val: float) -> 'Tensor':
        """原地裁剪到最小值"""
        data_array = arrays.Array(self._data)
        min_val_array = arrays.Array([min_val] * len(data_array.data))
        max_result = arrays.maximum(data_array, min_val_array)
        result_array = arrays.asarray_numpy_compatible(max_result.data)
        self._data = result_array.data.reshape(self._data.shape)
        return self

    def __gt__(self, other: Union['Tensor', float, int, list]) -> 'Tensor':
        """Greater than comparison"""
        if hasattr(other, 'shape') and hasattr(other, 'dtype'):
            other = Tensor(other)
        if isinstance(other, Tensor):
            return Tensor(self._data > other._data)
        return Tensor(self._data > other)

    def __lt__(self, other: Union['Tensor', float, int, list]) -> 'Tensor':
        """Less than comparison"""
        if hasattr(other, 'shape') and hasattr(other, 'dtype'):
            other = Tensor(other)
        if isinstance(other, Tensor):
            return Tensor(self._data < other._data)
        return Tensor(self._data < other)

    def __ge__(self, other: Union['Tensor', float, int, list]) -> 'Tensor':
        """Greater than or equal comparison"""
        if hasattr(other, 'shape') and hasattr(other, 'dtype'):
            other = Tensor(other)
        if isinstance(other, Tensor):
            return Tensor(self._data >= other._data)
        return Tensor(self._data >= other)

    def __le__(self, other: Union['Tensor', float, int, list]) -> 'Tensor':
        """Less than or equal comparison"""
        if hasattr(other, 'shape') and hasattr(other, 'dtype'):
            other = Tensor(other)
        if isinstance(other, Tensor):
            return Tensor(self._data <= other._data)
        return Tensor(self._data <= other)

    def __rgt__(self, other: Union['Tensor', float, int, list]) -> 'Tensor':
        """Reverse greater than comparison"""
        if hasattr(other, 'shape') and hasattr(other, 'dtype'):
            other = Tensor(other)
        if isinstance(other, Tensor):
            return Tensor(other._data > self._data)
        return Tensor(other > self._data)

    def __rlt__(self, other: Union['Tensor', float, int, list]) -> 'Tensor':
        """Reverse less than comparison"""
        if hasattr(other, 'shape') and hasattr(other, 'dtype'):
            other = Tensor(other)
        if isinstance(other, Tensor):
            return Tensor(other._data < self._data)
        return Tensor(other < self._data)

    def __rge__(self, other: Union['Tensor', float, int, list]) -> 'Tensor':
        """Reverse greater than or equal comparison"""
        if hasattr(other, 'shape') and hasattr(other, 'dtype'):
            other = Tensor(other)
        if isinstance(other, Tensor):
            return Tensor(other._data >= self._data)
        return Tensor(other >= self._data)

    def __rle__(self, other: Union['Tensor', float, int, list]) -> 'Tensor':
        """Reverse less than or equal comparison"""
        if hasattr(other, 'shape') and hasattr(other, 'dtype'):
            other = Tensor(other)
        if isinstance(other, Tensor):
            return Tensor(other._data <= self._data)
        return Tensor(other <= self._data)

    def type_as(self, other: 'Tensor') -> 'Tensor':
        """转换为与other相同类型的张量"""
        return Tensor(self._data.astype(other.dtype), requires_grad=self.requires_grad)

    def erf(self) -> 'Tensor':
        """误差函数"""
        from scipy.special import erf
        from autograd_T import Function
        class Erf(Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return Tensor(erf(x._data))
            
            @staticmethod
            def backward(ctx, grad_output):
                x, = ctx.saved_tensors
                return grad_output * (2 / math.sqrt(math.pi)) * (-x._data * x._data).exp()
        
        return Erf.apply(self)

    @property
    def T(self) -> 'Tensor':
        """返回转置张量"""
        # 使用transpose操作而不是直接创建新Tensor，以保持梯度传播
        return self.transpose()

    def log(self) -> 'Tensor':
        from operations_T import log
        return log(self)

    # ==================== 初始化方法 ====================
    @classmethod
    def _calculate_fan_in_and_fan_out(cls, tensor: 'Tensor') -> Tuple[int, int]:
        """计算张量的fan_in（输入维度）和fan_out（输出维度）"""
        dimensions = tensor.ndim
        if dimensions < 2:
            raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

        num_input_fmaps = tensor.shape[1]
        num_output_fmaps = tensor.shape[0]
        receptive_field_size = 1
        if dimensions > 2:
            receptive_field_size = tensor.data[0][0].size
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
        return fan_in, fan_out

    @classmethod
    def xavier_uniform_(cls, tensor: 'Tensor', gain: float = 1.0) -> 'Tensor':
        """
        Xavier均匀分布初始化 (U(-a, a), a = gain * sqrt(6/(fan_in + fan_out)))
        适用于线性层和卷积层
        """
        fan_in, fan_out = cls._calculate_fan_in_and_fan_out(tensor)
        std = gain * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        return tensor.uniform_(-a, a)

    @classmethod
    def xavier_normal_(cls, tensor: 'Tensor', gain: float = 1.0) -> 'Tensor':
        """
        Xavier正态分布初始化 (N(0, std^2), std = gain * sqrt(2/(fan_in + fan_out)))
        """
        fan_in, fan_out = cls._calculate_fan_in_and_fan_out(tensor)
        std = gain * math.sqrt(2.0 / (fan_in + fan_out))
        return tensor.normal_(0, std)

    @classmethod
    def kaiming_uniform_(
        cls,
        tensor: 'Tensor',
        a: float = 0,
        mode: str = 'fan_in',
        nonlinearity: str = 'leaky_relu'
    ) -> 'Tensor':
        """
        Kaiming均匀分布初始化 (U(-bound, bound))
        
        Args:
            a:     负斜率（仅对leaky_relu有效）
            mode:  使用fan_in（前向）或fan_out（反向）
            nonlinearity: 激活函数 ('relu'|'leaky_relu'|'linear'|'sigmoid'|'tanh')
        """
        fan = cls._calculate_fan_in_and_fan_out(tensor)[0 if mode == 'fan_in' else 1]
        
        # 计算推荐增益值
        if nonlinearity == 'relu':
            gain = math.sqrt(2.0)
        elif nonlinearity == 'leaky_relu':
            gain = math.sqrt(2.0 / (1 + a ** 2))
        elif nonlinearity in ['sigmoid', 'tanh']:
            gain = 1.0
        else:  # linear
            gain = 1.0
            
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std
        uniform_array = arrays.random.uniform(-bound, bound, tensor.shape)
        data_array = arrays.asarray_numpy_compatible(uniform_array.data, dtype=tensor.dtype)
        tensor._data = data_array.data.reshape(tensor.shape)
        return tensor

    @classmethod
    def kaiming_normal_(
        cls,
        tensor: 'Tensor',
        a: float = 0,
        mode: str = 'fan_in',
        nonlinearity: str = 'leaky_relu'
    ) -> 'Tensor':
        """
        Kaiming正态分布初始化 (N(0, std^2))
        """
        fan = cls._calculate_fan_in_and_fan_out(tensor)[0 if mode == 'fan_in' else 1]
        
        if nonlinearity == 'relu':
            gain = math.sqrt(2.0)
        elif nonlinearity == 'leaky_relu':
            gain = math.sqrt(2.0 / (1 + a ** 2))
        elif nonlinearity in ['sigmoid', 'tanh']:
            gain = 1.0
        else:
            gain = 1.0
            
        std = gain / math.sqrt(fan)
        normal_array = arrays.random.normal(0, std, tensor.shape)
        data_array = arrays.asarray_numpy_compatible(normal_array.data, dtype=tensor.dtype)
        tensor._data = data_array.data.reshape(tensor.shape)
        return tensor

    # ==================== 初始化验证器 ====================
    @staticmethod
    def validate_init(tensor: 'Tensor', 
                    expected_std: float, 
                    context: str = "",
                    mean_tol: float = 0.15,
                    std_tol: float = 0.3) -> bool:
        """
        初始化质量验证器
        Args:
            tensor: 要验证的张量
            expected_std: 理论标准差
            context: 标识信息(用于错误提示)
            mean_tol: 均值允许误差范围
            std_tol: 标准差允许误差范围(比例)
        Returns:
            bool: 是否通过验证
        """
        data = tensor.data.flatten()
        data_array = arrays.Array(data)
        actual_mean = arrays.mean(data_array)
        actual_std = arrays.std(data_array)
        
        passed = True
        if abs(actual_mean) > mean_tol:
            print(f"⛔ {context} 均值超出范围: {actual_mean:.4f} (允许±{mean_tol})")
            passed = False
            
        std_min = (1 - std_tol) * expected_std
        std_max = (1 + std_tol) * expected_std
        if not (std_min < actual_std < std_max):
            print(f"⛔ {context} 标准差异常: {actual_std:.4f} (期望范围: {std_min:.4f}~{std_max:.4f})")
            passed = False
        
        if passed:
            print(f"✅ {context} 初始化验证通过 (mean={actual_mean:.4f}, std={actual_std:.4f})")
        return passed

    def reshape(self, shape):
        """改变张量形状"""
        from operations_T import reshape
        return reshape(self, shape)

    def __hash__(self):
        return id(self)

    @classmethod
    def stack(cls, tensors, dim=0):
        """将一组Tensor沿新维度拼接"""
        processed_arrays = []
        for t in tensors:
            if isinstance(t, Tensor):
                processed_arrays.append(t.data)
            else:
                array_result = arrays.asarray_numpy_compatible(t)
                processed_arrays.append(array_result.data)
        
        arrays_list = [arrays.Array(arr) for arr in processed_arrays]
        stacked_result = arrays.stack(arrays_list, axis=dim)
        stacked_array = arrays.asarray_numpy_compatible(stacked_result.data)
        stacked = stacked_array.data.reshape(stacked_result.shape)
        return cls(stacked)

    def transpose(self, dim0=None, dim1=None):
        """转置张量
        
        Args:
            dim0: 第一个维度
            dim1: 第二个维度
            
        Returns:
            转置后的张量
        """
        from operations_T import transpose
        
        if dim0 is not None and dim1 is not None:
            axes = list(range(len(self.shape)))
            axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
            t = transpose(self, tuple(axes))
        elif dim0 is not None:
            axes = list(range(len(self.shape)))
            axes[dim0], axes[-1] = axes[-1], axes[dim0]
            t = transpose(self, tuple(axes))
        else:
            t = transpose(self, None)
        # 注册view
        import weakref
        base = self
        while hasattr(base, '_base') and base._base is not None:
            base = base._base
        if isinstance(base, Parameter):
            if not hasattr(base, '_views'):
                base._views = set()
            base._views.add(weakref.ref(t))
        return t

    def chunk(self, chunks: int, dim: int = -1) -> List['Tensor']:
        """将张量分割成指定数量的块
        
        Args:
            chunks: 要分割的块数
            dim: 要分割的维度
            
        Returns:
            包含分割后张量的列表
        """
        if dim < 0:
            dim = len(self.shape) + dim
            
        if chunks <= 0:
            raise ValueError("chunks must be positive")
            
        size = self.shape[dim]
        if size % chunks != 0:
            raise ValueError(f"tensor size {size} in dimension {dim} is not divisible by chunks {chunks}")
            
        chunk_size = size // chunks
        result = []
        
        # 使用numpy的split函数进行分割
        # 确保dim是适合numpy的类型
        axis = dim
        array_data = arrays.Array(self._data)
        splits = arrays.split(array_data, chunks, axis=axis)
        for split in splits:
            result.append(Tensor(arrays.array(split.data), requires_grad=self.requires_grad))
            
        return result

    def squeeze(self, dim: Optional[int] = None) -> 'Tensor':
        """移除大小为1的维度
        
        Args:
            dim: 要移除的维度，如果为None则移除所有大小为1的维度
            
        Returns:
            移除维度后的新张量
        """
        if dim is not None:
            if dim < 0:
                dim = len(self.shape) + dim
            if dim >= len(self.shape):
                raise ValueError(f"dimension {dim} out of range")
            if self.shape[dim] != 1:
                return self
            new_shape = list(self.shape)
            new_shape.pop(dim)
            # 使用reshape操作来保持梯度链接
            from operations_T import reshape
            return reshape(self, new_shape)
        else:
            new_shape = tuple(s for s in self.shape if s != 1)
            if new_shape == self.shape:
                return self
            # 使用reshape操作来保持梯度链接
            from operations_T import reshape
            return reshape(self, new_shape)

    def __matmul__(self, other):
        """@ 运算符重载"""
        return self.matmul(other)

    def __rmatmul__(self, other):
        """反向 @ 运算符重载"""
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return other.matmul(self)

    def __rmul__(self, other):
        from operations_T import mul
        return mul(Tensor(other), self)

    def __radd__(self, other):
        from operations_T import add
        return add(Tensor(other), self)

    def __rtruediv__(self, other):
        from operations_T import div
        return div(Tensor(other), self)

    def __and__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Tensor(arrays.logical_and(self._data, other._data))

    def __rand__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Tensor(arrays.logical_and(other._data, self._data))

    def min(self):
        data_array = arrays.Array(self._data)
        min_result = arrays.min(data_array)
        return Tensor(min_result)

    def tolist(self):
        """将Tensor转换为Python列表
        
        Returns:
            list: 张量数据对应的Python列表
        """
        return self._data.tolist()

    def attach_module_reference(self, module, visited=None):
        """将模块引用附加到张量及其所有子张量
        
        Args:
            module: 要附加的模块引用
            visited: 已访问的张量集合，用于防止循环引用
            
        Returns:
            self: 便于链式调用
        """
        if visited is None:
            visited = set()
        
        # 如果已经访问过这个张量，直接返回
        if id(self) in visited:
            return self
        
        # 标记为已访问
        visited.add(id(self))
        
        # 附加模块引用
        self._module = module
        
        # 递归遍历计算图中的子节点
        for child in getattr(self, '_children', []):
            if child is not None and id(child) != id(self):  # 避免自引用
                if hasattr(child, 'attach_module_reference'):
                    child.attach_module_reference(module, visited)
        
        # 尝试传递到梯度张量
        if hasattr(self, 'grad') and self.grad is not None:
            if hasattr(self.grad, 'attach_module_reference'):
                self.grad.attach_module_reference(module, visited)
        
        # 尝试传递到上下文中保存的张量
        ctx = getattr(self, '_ctx', None)
        if ctx is not None and hasattr(ctx, 'saved_tensors'):
            for tensor in ctx.saved_tensors:
                if tensor is not None and id(tensor) != id(self):  # 避免自引用
                    if hasattr(tensor, 'attach_module_reference'):
                        tensor.attach_module_reference(module, visited)
        
        return self
    
    def get_module_reference(self):
        """获取关联到此张量的模块引用
        
        Returns:
            module: 关联的模块，如果没有则返回None
        """
        return getattr(self, '_module', None)
    
    def ensure_has_grad(self):
        """确保张量有梯度，如果没有则创建零梯度"""
        if self.requires_grad and self.grad is None:
            self.grad = Tensor(arrays.zeros_like(self.data))
        return self

    def matmul(self, other):
        """矩阵乘法"""
        if not isinstance(other, Tensor):
            other = Tensor(other)
        from autograd_T import MatMul
        return MatMul.apply(self, other)

    def __getattr__(self, name):
        if name == 'grad':
            grad = self.__dict__.get('grad', None)
            if grad is None and hasattr(self, '_base') and self._base is not None:
                return getattr(self._base, 'grad', None)
            return grad
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

class Parameter(Tensor):
    """Parameter类，用于存储需要梯度的参数"""
    def __init__(self, data_or_shape, init_method='xavier_uniform', device='cpu', dtype='float32'):
        # 处理输入是Tensor的情况
        if isinstance(data_or_shape, Tensor):
            super().__init__(data_or_shape.data, requires_grad=True)
            return
            
        # 处理输入是numpy数组的情况
        if hasattr(data_or_shape, 'shape') and hasattr(data_or_shape, 'dtype'):
            super().__init__(data_or_shape, requires_grad=True)
            return
            
        # 处理输入是arrays.Array的情况
        if hasattr(data_or_shape, 'data') and hasattr(data_or_shape, 'shape'):
            super().__init__(data_or_shape, requires_grad=True)
            return
            
        # 处理输入是形状的情况
        shape = data_or_shape
        # 确保shape是整数元组
        def safe_int(x):
            try:
                # 处理numpy标量
                if hasattr(x, 'item'):
                    return int(x.item())
                # 处理numpy 0维数组
                elif hasattr(x, 'shape') and len(x.shape) == 0:
                    return int(x)
                # 处理单元素数组/列表
                elif hasattr(x, '__len__') and len(x) == 1:
                    return int(x[0])
                # 处理普通数字
                else:
                    return int(x)
            except (ValueError, TypeError, AttributeError):
                # 如果转换失败，返回默认值
                return 1
        
        if isinstance(shape, (list, tuple)):
            shape = tuple(safe_int(dim) for dim in shape)
        elif hasattr(shape, '__iter__') and not isinstance(shape, str):
            shape = tuple(safe_int(dim) for dim in shape)
        else:
            shape = (safe_int(shape),)
        # 根据初始化方法生成初始值
        import arrays
        if init_method == 'xavier_uniform':
            # Xavier/Glorot uniform initialization
            try:
                fan_in = int(shape[0]) if len(shape) > 1 else 1
                fan_out = int(shape[1]) if len(shape) > 1 else 1
            except:
                fan_in = 1
                fan_out = 1
            # 防止除零和负数
            denominator = max(fan_in + fan_out, 1e-6)
            sqrt_input = arrays.Array([6.0 / denominator])
            sqrt_result = arrays.sqrt(sqrt_input)
            scale = float(sqrt_result.data[0])
            # 限制scale的范围以防止溢出
            scale = max(min(scale, 1e6), -1e6)
            uniform_array = arrays.random.uniform(-scale, scale, tuple(shape))
            # 处理高维数组
            flat_data = uniform_array.data
            if len(flat_data) != arrays.prod(arrays.Array(shape)):
                # 如果数据长度不匹配，截断或填充
                expected_size = int(arrays.prod(arrays.Array(shape)))
                if len(flat_data) > expected_size:
                    flat_data = flat_data[:expected_size]
                else:
                    flat_data.extend([0.0] * (expected_size - len(flat_data)))
            # 处理维度限制问题
            total_size = int(arrays.prod(arrays.Array(shape)))
            if len(flat_data) != total_size:
                flat_data = flat_data[:total_size] if len(flat_data) > total_size else flat_data + [0.0] * (total_size - len(flat_data))
            
            # 处理numpy的维度限制问题
            try:
                data_array = arrays.asarray_numpy_compatible(flat_data, dtype=float)
                data = data_array.data.reshape(shape)
            except ValueError as e:
                if "maximum supported dimension" in str(e):
                    # 对于超过numpy维度限制的情况，使用分块处理
                    print(f"Warning: Shape {shape} exceeds numpy limits, using chunked approach")
                    if len(shape) == 2:
                        rows, cols = shape
                        # 分块处理大矩阵
                        chunk_size = min(64, rows)
                        data_chunks = []
                        for i in range(0, rows, chunk_size):
                            end_i = min(i + chunk_size, rows)
                            chunk_data = flat_data[i*cols:(end_i)*cols]
                            chunk_array = arrays.asarray_numpy_compatible(chunk_data, dtype=float)
                            chunk = chunk_array.data.reshape(end_i - i, cols)
                            data_chunks.append(chunk)
                        # 垂直拼接所有块
                        data = arrays.vstack(data_chunks)
                    else:
                        # 1D情况，直接转换
                        data_array = arrays.asarray_numpy_compatible(flat_data, dtype=float)
                        data = data_array.data
                else:
                    raise e
        elif init_method == 'zeros':
            pass
            zeros_array = arrays.zeros(shape, dtype='float32')
            data_array = arrays.asarray_numpy_compatible(zeros_array.data, dtype=float)
            data = data_array.data
        elif init_method == 'ones':
            ones_array = arrays.ones(shape, dtype='float32')
            data_array = arrays.asarray_numpy_compatible(ones_array.data, dtype=float)
            data = data_array.data
        else:
            raise ValueError(f"Unknown initialization method: {init_method}")
        super().__init__(data, requires_grad=True)
    
    def __repr__(self):
        return f"Parameter(shape={self.shape}, requires_grad={self.requires_grad})"

    @property
    def T(self) -> 'Tensor':
        """矩阵转置"""
        return self.transpose()

    @property
    def grad(self):
        grad = self.__dict__.get('_grad', None)
        if grad is not None:
            return grad
        # 查找所有view的grad
        views = getattr(self, '_views', set())
        for ref in list(views):
            v = ref()
            if v is not None:
                g = getattr(v, 'grad', None)
                if g is not None:
                    return g
        return None

    @grad.setter
    def grad(self, value):
        self.__dict__['_grad'] = value

    def __getattr__(self, name):
        if name == 'T':
            return self.transpose()
        # 调用父类的__getattr__
        return super().__getattr__(name)
