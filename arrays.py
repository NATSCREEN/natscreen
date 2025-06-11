#!/usr/bin/env python3
"""数组操作库 - 自定义实现，减少对外部依赖"""

import math
import random
from typing import List, Union, Tuple, Optional, Any
import pure_random  # 使用我们的纯Python随机数生成器替代random
import strong_nan

def uniform(low: float = 0.0, high: float = 1.0, size: Optional[Tuple[int, ...]] = None, dtype: Optional[type] = None) -> 'Array':
    """生成均匀分布的随机数
    
    Args:
        low: 最小值
        high: 最大值
        size: 输出数组的形状
        dtype: 输出数组的数据类型
        
    Returns:
        均匀分布的随机数数组
    """
    
    if size is None:
        size = (1,)
    if dtype is None:
        dtype = float
        
    # 计算总大小
    if isinstance(size, int):
        total_size = size
    else:
        total_size = 1
        for dim in size:
            total_size *= dim
    
    # 使用我们的纯Python随机数生成器
    data = [pure_random.uniform(low, high) for _ in range(total_size)]
    
    result = Array(data, dtype=dtype)
    if isinstance(size, tuple) and len(size) > 1:
        result = result.reshape(*size)
    
    return result

class Array:
    def __init__(self, data: Union[List, Tuple, float, int, 'Array'], dtype=None):

        if 'torch' in str(type(data)):
            pass
        elif isinstance(data, Array):
            self.data = data.data.copy()
        elif isinstance(data, (list, tuple)):
            # 处理多维嵌套数据，保持原始结构
            def convert_nested_to_float(nested_data):
                if isinstance(nested_data, (list, tuple)):
                    return [convert_nested_to_float(item) for item in nested_data]
                elif hasattr(nested_data, 'data'):
                    return convert_nested_to_float(nested_data.data)
                else:
                    return float(nested_data)
            
            self.data = convert_nested_to_float(data)
        elif hasattr(data, 'shape') and hasattr(data, 'flatten'):
            if len(data.shape) == 0:
                self.data = [float(data)]
            elif len(data.shape) == 1:
                self.data = [float(x) for x in data]
            else:
                self.data = data.tolist()
        else:
            self.data = [float(data)]
        self.shape = self._compute_shape()
        self.dtype = dtype or float
        
    def _flatten_and_convert(self, data: Union[List, Tuple]) -> List:
        """将嵌套列表或元组展平并转换为浮点数列表"""
        if not isinstance(data, (list, tuple)):
            return [float(data)]
        result = []
        for x in data:
            if isinstance(x, (list, tuple)):
                result.extend(self._flatten_and_convert(x))
            else:
                result.append(float(x))
        return result
    
    def _compute_shape(self) -> Tuple[int, ...]:
        """计算数组的形状"""
        if not isinstance(self.data, list):
            return (1,)
        if not self.data:
            return (0,)
            
        # 递归计算多维数组的形状
        def get_nested_shape(data):
            if not isinstance(data, list):
                return ()
            if not data:
                return (0,)
            
            first_shape = (len(data),)
            if isinstance(data[0], list):
                # 递归获取子维度的形状
                sub_shape = get_nested_shape(data[0])
                return first_shape + sub_shape
            else:
                return first_shape
        
        shape = get_nested_shape(self.data)
        
        # 验证所有子数组的形状是否一致
        def validate_shape(data, expected_shape):
            if len(expected_shape) == 1:
                return len(data) == expected_shape[0]
            if len(data) != expected_shape[0]:
                return False
            if len(expected_shape) > 1:
                for item in data:
                    if not validate_shape(item, expected_shape[1:]):
                        return False
            return True
        
        if not validate_shape(self.data, shape):
            raise ValueError("不规则数组：所有子数组的形状必须一致")
            
        return shape
    
    def reshape(self, *shape: int) -> 'Array':
        """重塑数组形状
        
        Args:
            *shape: 新的形状，可以使用-1表示自动计算该维度
            
        Returns:
            重塑后的数组
        """
        # 处理-1的情况
        total_size = len(self.data)
        
        # 处理输入形状
        if len(shape) == 1:
            if isinstance(shape[0], (list, tuple)):
                shape = shape[0]
            elif hasattr(shape[0], '__iter__'):  # 处理任何可迭代对象
                shape = tuple(shape[0])
        
        # 确保shape是列表
        shape = list(shape)
        
        # 将所有维度转换为整数
        shape = [int(dim) for dim in shape]
        
        # 首先确保我们有正确的扁平数据
        flat_data = []
        if hasattr(self, '_flat_data') and self._flat_data:
            # 如果有_flat_data属性，优先使用
            flat_data = self._flat_data[:]
        else:
            # 否则，手动扁平化
            flat_data = []  # 确保flat_data在正确的作用域内定义
            def flatten_recursive(data):
                if isinstance(data, list):
                    for item in data:
                        flatten_recursive(item)
                else:
                    flat_data.append(float(data))
            
            flatten_recursive(self.data)
        
        total_size = len(flat_data)
        
        # 计算-1的位置和值
        if -1 in shape:
            if shape.count(-1) > 1:
                raise ValueError("只能有一个维度为-1")
            idx = shape.index(-1)
            other_dims = 1
            for i, dim in enumerate(shape):
                if i != idx and dim != -1:
                    other_dims *= dim
            if other_dims == 0:
                shape[idx] = 0
            else:
                shape[idx] = total_size // other_dims
                if total_size % other_dims != 0:
                    raise ValueError(f"无法将大小为{total_size}的数组重塑为形状{shape}")
        else:
            # 计算总大小
            total_shape = 1
            for dim in shape:
                total_shape *= dim
            if total_shape != total_size:
                raise ValueError(f"cannot reshape array of size {total_size} into shape {shape}")
            
        # 创建新数组
        new_array = Array.__new__(Array)
        new_array.dtype = self.dtype
        new_array.shape = tuple(shape)
        
        # 重新构造嵌套数据结构，使用正确的扁平数据
        if len(shape) == 1:
            # 1D数组保持扁平结构
            new_array.data = flat_data[:]
        elif len(shape) == 2:
            # 2D数组，确保数据是正确的嵌套列表形式
            rows, cols = shape
            nested_data = []
            for i in range(rows):
                row = []
                for j in range(cols):
                    idx = i * cols + j
                    if idx < len(flat_data):
                        row.append(flat_data[idx])
                    else:
                        row.append(0.0)  # 填充默认值
                nested_data.append(row)
            new_array.data = nested_data
        elif len(shape) == 3:
            # 3D数组，构造三层嵌套结构
            d0, d1, d2 = shape
            nested_data = []
            for i in range(d0):
                layer = []
                for j in range(d1):
                    row = []
                    for k in range(d2):
                        idx = i * d1 * d2 + j * d2 + k
                        if idx < len(flat_data):
                            row.append(flat_data[idx])
                        else:
                            row.append(0.0)  # 填充默认值
                    layer.append(row)
                nested_data.append(layer)
            new_array.data = nested_data
        elif len(shape) == 4:
            # 4D数组，构造四层嵌套结构
            d0, d1, d2, d3 = shape
            nested_data = []
            for i in range(d0):
                batch = []
                for j in range(d1):
                    layer = []
                    for k in range(d2):
                        row = []
                        for l in range(d3):
                            idx = i * d1 * d2 * d3 + j * d2 * d3 + k * d3 + l
                            if idx < len(flat_data):
                                row.append(flat_data[idx])
                            else:
                                row.append(0.0)  # 填充默认值
                        layer.append(row)
                    batch.append(layer)
                nested_data.append(batch)
            new_array.data = nested_data
        else:
            # 对于更高维度，使用通用递归方法
            new_array.data = self._reshape_recursive(flat_data, shape, 0)
            
        return new_array
    
    def transpose(self):
        """转置数组"""
        if len(self.shape) != 2:
            raise ValueError("transpose requires 2D array")
        rows, cols = self.shape
        result = []
        for j in range(cols):
            row = []
            for i in range(rows):
                # 正确处理嵌套列表数据结构
                if isinstance(self.data[0], list):
                    # 对于嵌套列表，直接使用二维索引
                    row.append(self.data[i][j])
                else:
                    # 对于扁平列表，使用计算的索引
                    row.append(self.data[i * cols + j])
            result.append(row)
        return Array(result)
    
    @property
    def T(self):
        """转置属性"""
        return self.transpose()
    
    def __add__(self, other: Union['Array', float, int]) -> 'Array':
        if isinstance(other, (int, float)):
            # 标量加法，需要处理嵌套列表
            if isinstance(self.data, list):
                if len(self.data) > 0 and isinstance(self.data[0], list):  # 处理嵌套列表
                    return Array([[float(x + other) for x in row] for row in self.data], dtype=self.dtype)
                else:  # 处理一维列表
                    return Array([float(x + other) for x in self.data], dtype=self.dtype)
            else:  # 处理单个值
                return Array([float(self.data + other)], dtype=self.dtype)
        if isinstance(other, Array):
            if self.shape != other.shape:
                raise ValueError("shapes do not match")
            
            # 修复：对二维数组执行正确的逐元素加法
            if isinstance(self.data[0], list) and isinstance(other.data[0], list):
                # 两个都是二维数组，逐元素加法
                return Array([[float(a + b) for a, b in zip(row_a, row_b)] 
                            for row_a, row_b in zip(self.data, other.data)], dtype=self.dtype)
            elif not isinstance(self.data[0], list) and not isinstance(other.data[0], list):
                # 两个都是一维数组，逐元素加法
                return Array([float(a + b) for a, b in zip(self.data, other.data)], dtype=self.dtype)
            else:
                # 混合情况，按原逻辑处理
                return Array([a + b for a, b in zip(self.data, other.data)], dtype=self.dtype)
                
        raise TypeError(f"unsupported operand type(s) for +: 'Array' and '{type(other)}'")
    
    def __mul__(self, other):
        """乘法运算"""
        if hasattr(other, 'shape') and hasattr(other, 'tolist'):
            try:
                other = Array(other.tolist())
            except Exception:
                pass
        
        if isinstance(other, (int, float)):
            # 标量乘法，使用递归方法处理任意维度
            def mul_recursive(data):
                if isinstance(data, list):
                    if len(data) > 0 and isinstance(data[0], list):
                        # 嵌套列表，递归处理
                        return [mul_recursive(item) for item in data]
                    else:
                        # 一维列表，批量处理
                        return [float(x * other) for x in data]
                else:
                    # 单个值
                    return float(data * other)
            
            if isinstance(self.data, list):
                result_data = mul_recursive(self.data)
                return Array(result_data, dtype=self.dtype)
            else:  # 处理单个值
                return Array([float(self.data * other)], dtype=self.dtype)
        elif isinstance(other, Array):
            # 广播或元素级乘法
            # 检查是否需要广播
            if self.shape == other.shape:
                # 形状相同，执行元素级乘法
                if isinstance(self.data[0], list) and isinstance(other.data[0], list):
                    # 两个都是二维数组
                    return Array([[float(a * b) for a, b in zip(row_a, row_b)] 
                                for row_a, row_b in zip(self.data, other.data)], dtype=self.dtype)
                elif not isinstance(self.data[0], list) and not isinstance(other.data[0], list):
                    # 两个都是一维数组
                    return Array([float(a * b) for a, b in zip(self.data, other.data)], dtype=self.dtype)
            
            # 尝试执行广播
            # 情况1: self是标量，other是数组
            if len(self.data) == 1 and len(other.data) > 1:
                scalar_value = self.data[0]
                return other * scalar_value  # 递归调用，但顺序反过来
            
            # 情况2: other是标量，self是数组
            if len(other.data) == 1 and len(self.data) > 1:
                scalar_value = other.data[0]
                return self * scalar_value  # 递归调用，转为标量乘法
            
            # 特殊情况：一维数组(N,) 与二维数组(N,M)相乘
            # 对一维数组进行广播，使其形状与二维数组的行数匹配
            if isinstance(self.data[0], list) and not isinstance(other.data[0], list):
                if len(other.data) == len(self.data):  # 一维数组长度等于二维数组的行数
                    result = []
                    for i, row in enumerate(self.data):
                        result.append([float(cell * other.data[i]) for cell in row])
                    return Array(result, dtype=self.dtype)
            
            # 情况相反：二维数组(N,M)与一维数组(N,)相乘
            if not isinstance(self.data[0], list) and isinstance(other.data[0], list):
                if len(self.data) == len(other.data):  # 一维数组长度等于二维数组的行数
                    result = []
                    for i, row in enumerate(other.data):
                        result.append([float(self.data[i] * cell) for cell in row])
                    return Array(result, dtype=self.dtype)
            
            # 如果形状不匹配且无法广播，引发异常
            raise ValueError(f"shapes do not match for multiplication: {self.shape} vs {other.shape}")
        else:
            raise TypeError(f"unsupported operand type(s) for *: 'Array' and '{type(other)}'")
    
    def __rmul__(self, other):
        """反向乘法运算（处理标量 * Array的情况）"""
        if isinstance(other, (int, float)):
            return self.__mul__(other)
        raise TypeError(f"unsupported operand type(s) for *: '{type(other)}' and 'Array'")
    
    def __sub__(self, other: Union['Array', float, int]) -> 'Array':
        """减法运算"""
        if isinstance(other, (int, float)):
            # 标量减法，直接应用到每个元素
            if isinstance(self.data, list):
                if isinstance(self.data[0], list):  # 处理嵌套列表
                    return Array([[float(x - other) for x in row] for row in self.data], dtype=self.dtype)
                else:  # 处理一维列表
                    return Array([float(x - other) for x in self.data], dtype=self.dtype)
            else:  # 处理单个值
                return Array([float(self.data - other)], dtype=self.dtype)
        elif isinstance(other, Array):
            # 广播或元素级减法
            # 检查是否需要广播
            if self.shape == other.shape:
                # 形状相同，执行元素级减法
                if isinstance(self.data[0], list) and isinstance(other.data[0], list):
                    # 两个都是二维数组
                    return Array([[float(a - b) for a, b in zip(row_a, row_b)] 
                                for row_a, row_b in zip(self.data, other.data)], dtype=self.dtype)
                elif not isinstance(self.data[0], list) and not isinstance(other.data[0], list):
                    # 两个都是一维数组
                    return Array([float(a - b) for a, b in zip(self.data, other.data)], dtype=self.dtype)
            
            # 尝试执行广播
            # 情况1: self是标量，other是数组
            if len(self.data) == 1 and len(other.data) > 1:
                scalar_value = self.data[0]
                if isinstance(other.data[0], list):  # other是二维数组
                    return Array([[float(scalar_value - cell) for cell in row] for row in other.data], dtype=self.dtype)
                else:  # other是一维数组
                    return Array([float(scalar_value - x) for x in other.data], dtype=self.dtype)
            
            # 情况2: other是标量，self是数组
            if len(other.data) == 1 and len(self.data) > 1:
                scalar_value = other.data[0]
                return self - scalar_value  # 转换为标量减法
            
            # 特殊情况：一维数组(N,) 与二维数组(N,M)相减
            # 对一维数组进行广播，使其形状与二维数组的行数匹配
            if isinstance(self.data[0], list) and not isinstance(other.data[0], list):
                if len(other.data) == len(self.data):  # 一维数组长度等于二维数组的行数
                    result = []
                    for i, row in enumerate(self.data):
                        result.append([float(cell - other.data[i]) for cell in row])
                    return Array(result, dtype=self.dtype)
            
            # 情况相反：二维数组(N,M)与一维数组(N,)相减
            if not isinstance(self.data[0], list) and isinstance(other.data[0], list):
                if len(self.data) == len(other.data):  # 一维数组长度等于二维数组的行数
                    result = []
                    for i, row in enumerate(other.data):
                        result.append([float(self.data[i] - cell) for cell in row])
                    return Array(result, dtype=self.dtype)
            
            # 如果形状不匹配且无法广播，引发异常
            raise ValueError(f"shapes do not match for subtraction: {self.shape} vs {other.shape}")
        else:
            raise TypeError(f"unsupported operand type(s) for -: 'Array' and '{type(other)}'")
    
    def __rsub__(self, other: Union[float, int]) -> 'Array':
        """反向减法运算（处理标量 - Array的情况）"""
        if isinstance(other, (int, float)):
            # 对每个元素应用减法，支持多维数组
            def rsub_recursive(data):
                if isinstance(data, list):
                    if len(data) > 0 and isinstance(data[0], list):
                        # 嵌套列表，递归处理
                        return [rsub_recursive(item) for item in data]
                    else:
                        # 一维列表
                        return [float(other - x) for x in data]
                else:
                    # 单个值
                    return float(other - data)
            
            if isinstance(self.data, list):
                result_data = rsub_recursive(self.data)
                return Array(result_data, dtype=self.dtype)
            else:  # 处理单个值
                return Array([float(other - self.data)], dtype=self.dtype)
        else:
            raise TypeError(f"unsupported operand type(s) for -: '{type(other)}' and 'Array'")
    
    def __truediv__(self, other: Union['Array', float, int]) -> 'Array':
        """除法运算 - 优化版本"""
        if hasattr(other, 'shape') and hasattr(other, 'tolist'):
            try:
                other = Array(other.tolist())
            except Exception:
                pass
        
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("division by zero")
            # 标量除法，使用优化的递归方法
            def div_recursive(data):
                if isinstance(data, list):
                    if len(data) > 0 and isinstance(data[0], list):
                        # 嵌套列表，递归处理
                        return [div_recursive(item) for item in data]
                    else:
                        # 一维列表，批量处理
                        return [float(x / other) for x in data]
                else:
                    # 单个值
                    return float(data / other)
            
            if isinstance(self.data, list):
                result_data = div_recursive(self.data)
                return Array(result_data, dtype=self.dtype)
            else:  # 处理单个值
                return Array([float(self.data / other)], dtype=self.dtype)
        elif isinstance(other, Array):
            # 广播或元素级除法
            # 检查是否需要广播
            if self.shape == other.shape:
                # 形状相同，执行元素级除法
                def div_elementwise(data_a, data_b):
                    if isinstance(data_a, list) and isinstance(data_b, list):
                        if len(data_a) > 0 and isinstance(data_a[0], list):
                            # 多维数组，递归处理
                            return [div_elementwise(row_a, row_b) for row_a, row_b in zip(data_a, data_b)]
                        else:
                            # 一维数组，逐元素处理
                            return [float(a / b) if b != 0 else float('inf') for a, b in zip(data_a, data_b)]
                    else:
                        # 标量情况
                        return float(data_a / data_b) if data_b != 0 else float('inf')
                
                result_data = div_elementwise(self.data, other.data)
                return Array(result_data, dtype=self.dtype)
            
            # 尝试执行广播
            # 情况1: self是标量，other是数组 (标量/数组)
            if len(self.data) == 1 and len(other.data) > 1:
                # 这种情况很特殊，每个元素是 scalar / array_element
                scalar_value = self.data[0]
                if isinstance(other.data[0], list):  # other是二维数组
                    return Array([[float(scalar_value / cell) if cell != 0 else float('inf') for cell in row] 
                                 for row in other.data], dtype=self.dtype)
                else:  # other是一维数组
                    return Array([float(scalar_value / x) if x != 0 else float('inf') for x in other.data], dtype=self.dtype)
            
            # 情况2: other是标量，self是数组 (数组/标量)
            if len(other.data) == 1 and len(self.data) > 1:
                scalar_value = other.data[0]
                if scalar_value == 0:
                    raise ZeroDivisionError("division by zero")
                return self * (1.0 / scalar_value)  # 转换为乘法
            
            # 特殊情况：一维数组(N,) 与二维数组(N,M)相除
            # 对一维数组进行广播，使其形状与二维数组的行数匹配
            if isinstance(self.data[0], list) and not isinstance(other.data[0], list):
                if len(other.data) == len(self.data):  # 一维数组长度等于二维数组的行数
                    result = []
                    for i, row in enumerate(self.data):
                        if other.data[i] == 0:
                            result.append([float('inf')] * len(row))
                        else:
                            result.append([float(cell / other.data[i]) for cell in row])
                    return Array(result, dtype=self.dtype)
            
            # 情况相反：二维数组(N,M)与一维数组(N,)相除
            if not isinstance(self.data[0], list) and isinstance(other.data[0], list):
                if len(self.data) == len(other.data):  # 一维数组长度等于二维数组的行数
                    result = []
                    for i, row in enumerate(other.data):
                        result.append([float(self.data[i] / cell) if cell != 0 else float('inf') for cell in row])
                    return Array(result, dtype=self.dtype)
            
            # 如果形状不匹配且无法广播，引发异常
            raise ValueError(f"shapes do not match for division: {self.shape} vs {other.shape}")
        else:
            raise TypeError(f"unsupported operand type(s) for /: 'Array' and '{type(other)}'")
    
    def __rtruediv__(self, other: Union[float, int]) -> 'Array':
        """反向除法运算（处理标量 / Array的情况）"""
        if isinstance(other, (int, float)):
            # 对每个元素应用除法
            if isinstance(self.data, list):
                if isinstance(self.data[0], list):  # 处理嵌套列表
                    return Array([[float(other / x) if x != 0 else float('inf') for x in row] for row in self.data], dtype=self.dtype)
                else:  # 处理一维列表
                    return Array([float(other / x) if x != 0 else float('inf') for x in self.data], dtype=self.dtype)
            else:  # 处理单个值
                return Array([float(other / self.data) if self.data != 0 else float('inf')], dtype=self.dtype)
        else:
            raise TypeError(f"unsupported operand type(s) for /: '{type(other)}' and 'Array'")
    
    def __pow__(self, other: Union['Array', float, int]) -> 'Array':
        if isinstance(other, (int, float)):
            return Array([x ** other for x in self.data], dtype=self.dtype)
        if isinstance(other, Array):
            if self.shape != other.shape:
                raise ValueError("shapes do not match")
            return Array([a ** b for a, b in zip(self.data, other.data)], dtype=self.dtype)
        raise TypeError(f"unsupported operand type(s) for **: 'Array' and '{type(other)}'")
    
    def sum(self, axis: Optional[int] = None, keepdims: bool = False) -> Union['Array', float]:
        """计算数组的总和
        
        Args:
            axis: 计算总和的轴，如果为None则计算全局总和
            keepdims: 是否保持维度
            
        Returns:
            总和或总和数组
        """
        # 处理空数组
        if not self.data:
            return 0.0
            
        if axis is None:
            # 全局求和，需要处理嵌套结构
            if len(self.data) > 0 and isinstance(self.data[0], list):
                # 多维数组，递归展平后求和
                flat_data = self._flatten_recursive(self.data)
                return __builtins__['sum'](flat_data)
            else:
                # 1D数组
                return __builtins__['sum'](self.data)
        
        if axis >= len(self.shape):
            raise ValueError("axis out of bounds")
        
        # 实现按轴求和
        if len(self.shape) == 1:
            result = __builtins__['sum'](self.data)
            if keepdims:
                return Array([result])
            return result
        
        # 2D数组的轴求和
        if len(self.shape) == 2:
            rows, cols = self.shape
            if axis == 0:
                # 沿行求和，每列求和
                result = []
                for j in range(cols):
                    col_sum = __builtins__['sum'](self.data[i][j] for i in range(rows))
                    result.append(col_sum)
                result_array = Array(result)
                if keepdims:
                    # 保持维度：(cols,) -> (1, cols)
                    result_array = result_array.reshape(1, cols)
                return result_array
            else:
                # 沿列求和，每行求和
                result = []
                for i in range(rows):
                    row_sum = __builtins__['sum'](self.data[i])
                    result.append(row_sum)
                result_array = Array(result)
                if keepdims:
                    # 保持维度：(rows,) -> (rows, 1)
                    result_array = result_array.reshape(rows, 1)
                return result_array
        
        # 3D数组的轴求和
        if len(self.shape) == 3:
            d0, d1, d2 = self.shape
            if axis == 0:
                # 沿第一个轴求和
                result = []
                for i in range(d1):
                    row = []
                    for j in range(d2):
                        sum_val = __builtins__['sum'](self.data[k][i][j] for k in range(d0))
                        row.append(sum_val)
                    result.append(row)
                result_array = Array(result)
                if keepdims:
                    # 保持维度：(d1, d2) -> (1, d1, d2)
                    result_array = result_array.reshape(1, d1, d2)
                return result_array
            elif axis == 1:
                # 沿第二个轴求和
                result = []
                for i in range(d0):
                    row = []
                    for j in range(d2):
                        sum_val = __builtins__['sum'](self.data[i][k][j] for k in range(d1))
                        row.append(sum_val)
                    result.append(row)
                result_array = Array(result)
                if keepdims:
                    # 保持维度：(d0, d2) -> (d0, 1, d2)
                    # 需要重新构造数据结构
                    new_data = []
                    for i in range(d0):
                        new_data.append([result_array.data[i]])
                    result_array = Array(new_data)
                return result_array
            else:  # axis == 2
                # 沿第三个轴求和
                result = []
                for i in range(d0):
                    row = []
                    for j in range(d1):
                        sum_val = __builtins__['sum'](self.data[i][j][k] for k in range(d2))
                        row.append(sum_val)
                    result.append(row)
                result_array = Array(result)
                if keepdims:
                    # 保持维度：(d0, d1) -> (d0, d1, 1)
                    # 需要重新构造数据结构
                    new_data = []
                    for i in range(d0):
                        layer = []
                        for j in range(d1):
                            layer.append([result_array.data[i][j]])
                        new_data.append(layer)
                    result_array = Array(new_data)
                return result_array
        
        # 对于更高维度的数组，使用通用方法
        return self._sum_general(axis, keepdims)
    
    def _sum_general(self, axis: int, keepdims: bool = False):
        """通用的多维sum实现"""
        # 对于4维及以上的数组，我们使用递归方法
        if len(self.shape) == 4:
            return self._sum_4d(axis, keepdims)
        
        # 对于更高维度，使用通用递归方法
        return self._sum_recursive(axis, keepdims)
    
    def _sum_4d(self, axis: int, keepdims: bool = False):
        """4维数组的sum实现"""
        d0, d1, d2, d3 = self.shape
        
        if axis == 0:
            # 沿第一个轴求和
            result = []
            for i in range(d1):
                layer = []
                for j in range(d2):
                    row = []
                    for k in range(d3):
                        sum_val = __builtins__['sum'](self.data[l][i][j][k] for l in range(d0))
                        row.append(sum_val)
                    layer.append(row)
                result.append(layer)
            result_array = Array(result)
            if keepdims:
                # 保持维度：(d1, d2, d3) -> (1, d1, d2, d3)
                # 需要重新构造数据结构
                new_data = [result_array.data]
                result_array = Array(new_data)
            return result_array
        elif axis == 1:
            # 沿第二个轴求和
            result = []
            for i in range(d0):
                layer = []
                for j in range(d2):
                    row = []
                    for k in range(d3):
                        sum_val = __builtins__['sum'](self.data[i][l][j][k] for l in range(d1))
                        row.append(sum_val)
                    layer.append(row)
                result.append(layer)
            result_array = Array(result)
            if keepdims:
                # 保持维度：(d0, d2, d3) -> (d0, 1, d2, d3)
                # 需要重新构造数据结构
                new_data = []
                for i in range(d0):
                    new_data.append([result_array.data[i]])
                result_array = Array(new_data)
            return result_array
        elif axis == 2:
            # 沿第三个轴求和
            result = []
            for i in range(d0):
                layer = []
                for j in range(d1):
                    row = []
                    for k in range(d3):
                        sum_val = __builtins__['sum'](self.data[i][j][l][k] for l in range(d2))
                        row.append(sum_val)
                    layer.append(row)
                result.append(layer)
            result_array = Array(result)
            if keepdims:
                # 保持维度：(d0, d1, d3) -> (d0, d1, 1, d3)
                # 需要重新构造数据结构
                new_data = []
                for i in range(d0):
                    batch = []
                    for j in range(d1):
                        batch.append([result_array.data[i][j]])
                    new_data.append(batch)
                result_array = Array(new_data)
            return result_array
        else:  # axis == 3
            # 沿第四个轴求和
            result = []
            for i in range(d0):
                layer = []
                for j in range(d1):
                    row = []
                    for k in range(d2):
                        sum_val = __builtins__['sum'](self.data[i][j][k][l] for l in range(d3))
                        row.append(sum_val)
                    layer.append(row)
                result.append(layer)
            result_array = Array(result)
            if keepdims:
                # 保持维度：(d0, d1, d2) -> (d0, d1, d2, 1)
                # 需要重新构造数据结构
                new_data = []
                for i in range(d0):
                    batch = []
                    for j in range(d1):
                        layer = []
                        for k in range(d2):
                            layer.append([result_array.data[i][j][k]])
                        batch.append(layer)
                    new_data.append(batch)
                result_array = Array(new_data)
            return result_array
    
    def _sum_recursive(self, axis: int, keepdims: bool = False):
        """递归实现的通用sum方法"""
        # 对于非常高维的数组，我们使用一个简化的方法
        # 将数组展平，然后重新组织
        flat_data = self._flatten_recursive(self.data)
        
        # 计算沿指定轴的总和
        # 这是一个简化实现，可能不完全正确，但可以处理基本情况
        if axis == len(self.shape) - 1:
            # 如果是最后一个轴，直接计算
            return __builtins__['sum'](flat_data)
        else:
            # 对于其他轴，返回全局总和作为后备
            return __builtins__['sum'](flat_data)
    
    def mean(self, axis: Optional[int] = None) -> Union['Array', float]:
        """计算数组的平均值
        
        Args:
            axis: 计算平均值的轴，如果为None则计算所有元素
            
        Returns:
            平均值
        """
        # 验证数据
        if not self.data:
            raise ValueError("无法计算空数组的平均值")
            
        # 验证轴
        if axis is not None and axis >= len(self.shape):
            raise ValueError(f"axis {axis} 超出数组维度 {len(self.shape)}")
            
        try:
            if axis is None:
                # 处理1D数组的平均值
                if len(self.shape) == 1:
                    return __builtins__['sum'](self.data) / len(self.data)
                else:
                    # 多维数组，先展平再计算
                    flat_data = self._flatten_recursive(self.data)
                    return __builtins__['sum'](flat_data) / len(flat_data)
            result = self.sum(axis)
            if isinstance(result, Array):
                return result / self.shape[axis]
            return result
        except Exception as e:
            raise RuntimeError(f"计算平均值失败: {str(e)}")
    
    def max(self, axis: Optional[int] = None, keepdims: bool = False) -> Union['Array', float]:
        """计算数组的最大值
        
        Args:
            axis: 计算最大值的轴，如果为None则计算全局最大值
            keepdims: 是否保持维度
            
        Returns:
            最大值或最大值数组
        """
        if axis is None:
            # 全局最大值
            if isinstance(self.data[0], list):
                # 多维数组，需要递归展平
                flat_data = self._flatten_recursive(self.data)
                return __builtins__['max'](flat_data)
            else:
                # 一维数组
                return __builtins__['max'](self.data)
        
        if axis >= len(self.shape):
            raise ValueError("axis out of bounds")
        
        if len(self.shape) == 1:
            return __builtins__['max'](self.data)
        
        if len(self.shape) == 2:
            rows, cols = self.shape
            if axis == 0:
                # 沿第一个轴（行）计算，返回每列的最大值
                result = []
                for j in range(cols):
                    col_max = self.data[0][j]
                    for i in range(1, rows):
                        if self.data[i][j] > col_max:
                            col_max = self.data[i][j]
                    result.append(col_max)
                return Array(result)
            else:  # axis == 1
                # 沿第二个轴（列）计算，返回每行的最大值
                result = []
                for i in range(rows):
                    row_max = self.data[i][0]
                    for j in range(1, cols):
                        if self.data[i][j] > row_max:
                            row_max = self.data[i][j]
                    result.append(row_max)
                return Array(result)
        
        if len(self.shape) == 3:
            # 3D数组支持
            d0, d1, d2 = self.shape
            if axis == 0:
                # 沿第一个轴计算
                result = []
                for i in range(d1):
                    row = []
                    for j in range(d2):
                        max_val = self.data[0][i][j]
                        for k in range(1, d0):
                            if self.data[k][i][j] > max_val:
                                max_val = self.data[k][i][j]
                        row.append(max_val)
                    result.append(row)
                return Array(result)
            elif axis == 1:
                # 沿第二个轴计算
                result = []
                for i in range(d0):
                    row = []
                    for j in range(d2):
                        max_val = self.data[i][0][j]
                        for k in range(1, d1):
                            if self.data[i][k][j] > max_val:
                                max_val = self.data[i][k][j]
                        row.append(max_val)
                    result.append(row)
                return Array(result)
            else:  # axis == 2
                # 沿第三个轴计算
                result = []
                for i in range(d0):
                    row = []
                    for j in range(d1):
                        max_val = self.data[i][j][0]
                        for k in range(1, d2):
                            if self.data[i][j][k] > max_val:
                                max_val = self.data[i][j][k]
                        row.append(max_val)
                    result.append(row)
                return Array(result)
        
        # 对于更高维度的数组，使用通用方法
        return self._max_general(axis, keepdims)
    
    def _flatten_recursive(self, data):
        """递归展平嵌套列表"""
        result = []
        for item in data:
            if isinstance(item, list):
                result.extend(self._flatten_recursive(item))
            else:
                result.append(item)
        return result
    
    def _max_general(self, axis: int, keepdims: bool = False):
        """通用的多维max实现"""
        # 对于4维及以上的数组，我们使用递归方法
        if len(self.shape) == 4:
            return self._max_4d(axis, keepdims)
        
        # 对于更高维度，使用通用递归方法
        return self._max_recursive(axis, keepdims)
    
    def _max_4d(self, axis: int, keepdims: bool = False):
        """4维数组的max实现"""
        d0, d1, d2, d3 = self.shape
        
        if axis == 0:
            # 沿第一个轴计算
            result = []
            for i in range(d1):
                layer = []
                for j in range(d2):
                    row = []
                    for k in range(d3):
                        max_val = self.data[0][i][j][k]
                        for l in range(1, d0):
                            if self.data[l][i][j][k] > max_val:
                                max_val = self.data[l][i][j][k]
                        row.append(max_val)
                    layer.append(row)
                result.append(layer)
            return Array(result)
        elif axis == 1:
            # 沿第二个轴计算
            result = []
            for i in range(d0):
                layer = []
                for j in range(d2):
                    row = []
                    for k in range(d3):
                        max_val = self.data[i][0][j][k]
                        for l in range(1, d1):
                            if self.data[i][l][j][k] > max_val:
                                max_val = self.data[i][l][j][k]
                        row.append(max_val)
                    layer.append(row)
                result.append(layer)
            return Array(result)
        elif axis == 2:
            # 沿第三个轴计算
            result = []
            for i in range(d0):
                layer = []
                for j in range(d1):
                    row = []
                    for k in range(d3):
                        max_val = self.data[i][j][0][k]
                        for l in range(1, d2):
                            if self.data[i][j][l][k] > max_val:
                                max_val = self.data[i][j][l][k]
                        row.append(max_val)
                    layer.append(row)
                result.append(layer)
            return Array(result)
        else:  # axis == 3
            # 沿第四个轴计算
            result = []
            for i in range(d0):
                layer = []
                for j in range(d1):
                    row = []
                    for k in range(d2):
                        max_val = self.data[i][j][k][0]
                        for l in range(1, d3):
                            if self.data[i][j][k][l] > max_val:
                                max_val = self.data[i][j][k][l]
                        row.append(max_val)
                    layer.append(row)
                result.append(layer)
            result_array = Array(result)
            if keepdims:
                # 保持维度：(d0, d1, d2) -> (d0, d1, d2, 1)
                # 需要重新构造数据结构
                new_data = []
                for i in range(d0):
                    layer = []
                    for j in range(d1):
                        row = []
                        for k in range(d2):
                            row.append([result_array.data[i][j][k]])
                        layer.append(row)
                    new_data.append(layer)
                result_array = Array(new_data)
            return result_array
    
    def _max_recursive(self, axis: int, keepdims: bool = False):
        """递归实现的通用max方法"""
        # 对于非常高维的数组，我们使用一个简化的方法
        # 将数组展平，然后重新组织
        flat_data = self._flatten_recursive(self.data)
        
        # 计算沿指定轴的最大值
        # 这是一个简化实现，可能不完全正确，但可以处理基本情况
        if axis == len(self.shape) - 1:
            # 如果是最后一个轴，直接计算
            return __builtins__['max'](flat_data)
        else:
            # 对于其他轴，返回全局最大值作为后备
            return __builtins__['max'](flat_data)
    
    def min(self, axis: Optional[int] = None) -> Union['Array', float]:
        if axis is None:
            return __builtins__['min'](self.data)
        if axis >= len(self.shape):
            raise ValueError("axis out of bounds")
        if len(self.shape) == 1:
            return __builtins__['min'](self.data)
        if len(self.shape) == 2:
            rows, cols = self.shape
            if axis == 0:
                return Array([__builtins__['min'](self.data[i::cols]) for i in range(cols)])
            else:
                return Array([__builtins__['min'](self.data[i*cols:(i+1)*cols]) for i in range(rows)])
        raise NotImplementedError("min for arrays with more than 2 dimensions not implemented")
    
    def dot(self, other: 'Array') -> 'Array':
        if not isinstance(other, Array):
            raise TypeError("dot product requires Array operand")
        
        # 处理不同维度的矩阵乘法
        if len(self.shape) == 2 and len(other.shape) == 2:
            # 标准2D @ 2D矩阵乘法
            if self.shape[1] != other.shape[0]:
                raise ValueError("shapes not aligned for dot product")
            return self._dot_2d_2d(other)
        
        elif len(self.shape) == 3 and len(other.shape) == 2:
            # 3D @ 2D 批量矩阵乘法
            # (batch, seq, features) @ (features, out_features) -> (batch, seq, out_features)
            if self.shape[2] != other.shape[0]:
                raise ValueError("shapes not aligned for dot product")
            return self._dot_3d_2d(other)
        
        elif len(self.shape) == 3 and len(other.shape) == 3:
            # 3D @ 3D 批量矩阵乘法
            if self.shape[0] != other.shape[0] or self.shape[2] != other.shape[1]:
                raise ValueError("shapes not aligned for dot product")
            return self._dot_3d_3d(other)
        
        elif len(self.shape) == 4 and len(other.shape) == 4:
            # 4D @ 4D 批量矩阵乘法
            # (batch, heads, seq, features) @ (batch, heads, features, seq) -> (batch, heads, seq, seq)
            if (self.shape[0] != other.shape[0] or self.shape[1] != other.shape[1] or 
                self.shape[3] != other.shape[2]):
                raise ValueError("shapes not aligned for 4D dot product")
            return self._dot_4d_4d(other)
        
        else:
            raise ValueError(f"unsupported dot product dimensions: {len(self.shape)}D @ {len(other.shape)}D")
    
    def _dot_2d_2d(self, other: 'Array') -> 'Array':
        """2D @ 2D 矩阵乘法"""
        rows_a, cols_a = self.shape
        rows_b, cols_b = other.shape
        
        result = []
        for i in range(rows_a):
            row = []
            for j in range(cols_b):
                sum_val = 0
                for k in range(cols_a):
                    # 正确处理嵌套列表数据
                    if isinstance(self.data[0], list):
                        val_a = self.data[i][k]
                    else:
                        val_a = self.data[i * cols_a + k]
                    
                    if isinstance(other.data[0], list):
                        val_b = other.data[k][j]
                    else:
                        val_b = other.data[k * cols_b + j]
                    
                    sum_val += val_a * val_b
                row.append(sum_val)
            result.append(row)
        return Array(result)
    
    def _dot_3d_2d(self, other: 'Array') -> 'Array':
        """3D @ 2D 批量矩阵乘法"""
        batch_size, seq_len, features = self.shape
        out_features = other.shape[1]
        
        result = []
        for b in range(batch_size):
            batch_result = []
            for s in range(seq_len):
                row = []
                for o in range(out_features):
                    sum_val = 0
                    for f in range(features):
                        # 获取3D数组中的值: self.data[b][s][f]
                        val_a = self.data[b][s][f]
                        # 获取2D数组中的值: other.data[f][o]
                        if isinstance(other.data[0], list):
                            val_b = other.data[f][o]
                        else:
                            val_b = other.data[f * out_features + o]
                        sum_val += val_a * val_b
                    row.append(sum_val)
                batch_result.append(row)
            result.append(batch_result)
        return Array(result)
    
    def _dot_3d_3d(self, other: 'Array') -> 'Array':
        """3D @ 3D 批量矩阵乘法"""
        batch_size = self.shape[0]
        result = []
        for b in range(batch_size):
            # 提取每个批次的2D矩阵并进行乘法
            a_2d = Array(self.data[b])
            b_2d = Array(other.data[b])
            batch_result = a_2d._dot_2d_2d(b_2d)
            result.append(batch_result.data)
        return Array(result)
    
    def _dot_4d_4d(self, other: 'Array') -> 'Array':
        """4D @ 4D 批量矩阵乘法"""
        batch_size, num_heads, seq_len, features = self.shape
        _, _, features_other, seq_len_other = other.shape
        
        result = []
        for b in range(batch_size):
            batch_result = []
            for h in range(num_heads):
                # 提取每个头的2D矩阵并进行乘法
                a_2d = Array(self.data[b][h])  # (seq_len, features)
                b_2d = Array(other.data[b][h])  # (features, seq_len_other)
                head_result = a_2d._dot_2d_2d(b_2d)  # (seq_len, seq_len_other)
                batch_result.append(head_result.data)
            result.append(batch_result)
        return Array(result)
    
    def __matmul__(self, other: 'Array') -> 'Array':
        return self.dot(other)
    
    def __getitem__(self, key: Union[int, slice, Tuple]) -> 'Array':
        if isinstance(key, int):
            if key >= len(self.data):
                raise IndexError("index out of bounds")
            return Array([self.data[key]], dtype=self.dtype)
        if isinstance(key, slice):
            return Array(self.data[key], dtype=self.dtype)
        if isinstance(key, tuple):
            # 实现多维索引
            if len(key) != len(self.shape):
                raise ValueError("number of indices must match array dimensions")
            # 简化版本：只支持2D数组的索引
            if len(self.shape) == 2:
                rows, cols = self.shape
                if isinstance(key[0], int) and isinstance(key[1], int):
                    return Array([self.data[key[0] * cols + key[1]]], dtype=self.dtype)
                # 处理切片
                row_slice = key[0] if isinstance(key[0], slice) else slice(key[0], key[0] + 1)
                col_slice = key[1] if isinstance(key[1], slice) else slice(key[1], key[1] + 1)
                result = []
                for i in range(*row_slice.indices(rows)):
                    for j in range(*col_slice.indices(cols)):
                        result.append(self.data[i * cols + j])
                return Array(result, dtype=self.dtype)
        raise TypeError("invalid index type")
    
    def __setitem__(self, key: Union[int, slice, Tuple], value: Union['Array', float, int]) -> None:
        if isinstance(key, int):
            if key >= len(self.data):
                raise IndexError("index out of bounds")
            self.data[key] = float(value)
        elif isinstance(key, slice):
            if isinstance(value, (int, float)):
                for i in range(*key.indices(len(self.data))):
                    self.data[i] = float(value)
            elif isinstance(value, Array):
                for i, v in zip(range(*key.indices(len(self.data))), value.data):
                    self.data[i] = float(v)
        elif isinstance(key, tuple):
            if len(key) != len(self.shape):
                raise ValueError("number of indices must match array dimensions")
            if len(self.shape) == 2:
                rows, cols = self.shape
                if isinstance(key[0], int) and isinstance(key[1], int):
                    self.data[key[0] * cols + key[1]] = float(value)
                else:
                    row_slice = key[0] if isinstance(key[0], slice) else slice(key[0], key[0] + 1)
                    col_slice = key[1] if isinstance(key[1], slice) else slice(key[1], key[1] + 1)
                    if isinstance(value, (int, float)):
                        for i in range(*row_slice.indices(rows)):
                            for j in range(*col_slice.indices(cols)):
                                self.data[i * cols + j] = float(value)
                    elif isinstance(value, Array):
                        idx = 0
                        for i in range(*row_slice.indices(rows)):
                            for j in range(*col_slice.indices(cols)):
                                self.data[i * cols + j] = float(value.data[idx])
                                idx += 1
        else:
            raise TypeError("invalid index type")
    
    def __repr__(self) -> str:
        return f"Array({self.data}, shape={self.shape}, dtype={self.dtype})"
    
    @property
    def ndim(self) -> int:
        """返回数组的维度数"""
        return len(self.shape)
    
    def __str__(self) -> str:
        return self.__repr__()
    
    def copy(self) -> 'Array':
        return Array(self.data.copy(), dtype=self.dtype)
    
    def astype(self, dtype) -> 'Array':
        return Array(self.data, dtype=dtype)
    
    def fill(self, value: float) -> None:
        self.data = [float(value)] * len(self.data)
    
    def clip(self, min_val: float, max_val: float) -> 'Array':
        """限制数组元素在指定范围内，处理嵌套列表"""
        def clip_recursive(data):
            if isinstance(data, list):
                return [clip_recursive(item) for item in data]
            else:
                return __builtins__['max'](min_val, __builtins__['min'](data, max_val))
        
        return Array(clip_recursive(self.data), dtype=self.dtype)
    
    def exp(self) -> 'Array':
        """计算数组的指数，支持任意维度"""
        def exp_recursive(data):
            if isinstance(data, list):
                return [exp_recursive(item) for item in data]
            else:
                try:
                    return math.exp(float(data))
                except (ValueError, TypeError):
                    return 0.0
        
        result_data = exp_recursive(self.data)
        result = Array(result_data, dtype=self.dtype)
        # 确保保持原始形状
        if hasattr(self, 'shape'):
            result.shape = self.shape
        return result
    
    def log(self) -> 'Array':
        """计算数组的自然对数，支持任意维度"""
        def log_recursive(data):
            if isinstance(data, list):
                return [log_recursive(item) for item in data]
            else:
                try:
                    val = float(data)
                    if val <= 0:
                        return float('-inf')  # 或者返回一个小的负数
                    return math.log(val)
                except (ValueError, TypeError):
                    return 0.0
        
        result_data = log_recursive(self.data)
        result = Array(result_data, dtype=self.dtype)
        # 确保保持原始形状
        if hasattr(self, 'shape'):
            result.shape = self.shape
        return result
    
    def sqrt(self) -> 'Array':
        """计算数组的平方根，支持任意维度"""
        def sqrt_recursive(data):
            if isinstance(data, list):
                return [sqrt_recursive(item) for item in data]
            else:
                try:
                    val = float(data)
                    if val < 0:
                        return 0.0  # 或者抛出错误
                    return math.sqrt(val)
                except (ValueError, TypeError):
                    return 0.0
        
        result_data = sqrt_recursive(self.data)
        result = Array(result_data, dtype=self.dtype)
        # 确保保持原始形状
        if hasattr(self, 'shape'):
            result.shape = self.shape
        return result
    
    def abs(self) -> 'Array':
        def abs_recursive(data):
            if isinstance(data, list):
                return [abs_recursive(item) for item in data]
            else:
                return __builtins__['abs'](data)
        
        result_data = abs_recursive(self.data)
        return Array(result_data, dtype=self.dtype)
    
    def flatten(self) -> 'Array':
        """将数组展平为一维数组
        
        Returns:
            展平后的一维数组
        """
        def flatten_recursive(data):
            """递归展平嵌套列表"""
            result = []
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, list):
                        result.extend(flatten_recursive(item))
                    else:
                        result.append(item)
            else:
                result.append(data)
            return result
        
        flat_data = flatten_recursive(self.data)
        return Array(flat_data, dtype=self.dtype)
    
    def var(self, axis: Optional[int] = None) -> Union['Array', float]:
        mean_val = self.mean(axis)
        if axis is None:
            return __builtins__['sum']((x - mean_val) ** 2 for x in self.data) / len(self.data)
        # 实现按轴计算方差
        if len(self.shape) == 1:
            return __builtins__['sum']((x - mean_val) ** 2 for x in self.data) / len(self.data)
        if len(self.shape) == 2:
            rows, cols = self.shape
            if axis == 0:
                return Array([__builtins__['sum']((self.data[i::cols] - mean_val.data[i]) ** 2 for i in range(cols)) / rows])
            else:
                return Array([__builtins__['sum']((self.data[i*cols:(i+1)*cols] - mean_val.data[i]) ** 2 for i in range(rows)) / cols])
        raise NotImplementedError("var for arrays with more than 2 dimensions not implemented")

    def _ensure_2d(self) -> None:
        """确保数组是2D的，如果不是则转换"""
        if len(self.shape) == 1:
            self.data = [self.data]
            self.shape = (1, len(self.data[0]))

    def argmax(self, axis: Optional[int] = None) -> Union['Array', int]:
        """返回最大值的索引
        
        Args:
            axis: 计算最大值索引的轴，如果为None则返回全局最大值索引
            
        Returns:
            最大值的索引
        """
        if axis is None:
            # 返回全局最大值的索引 - 需要处理嵌套列表情况
            if isinstance(self.data[0], list):
                # 2D数组情况
                max_val = self.data[0][0]
                max_idx = 0
                idx = 0
                for i, row in enumerate(self.data):
                    for j, val in enumerate(row):
                        if val > max_val:
                            max_val = val
                            max_idx = idx
                        idx += 1
                return max_idx
            else:
                # 1D数组情况
                max_val = self.data[0]
                max_idx = 0
                for i, val in enumerate(self.data):
                    if val > max_val:
                        max_val = val
                        max_idx = i
                return max_idx
        
        if axis >= len(self.shape):
            raise ValueError("axis out of bounds")
        
        if len(self.shape) == 1:
            return self.argmax()
        
        if len(self.shape) == 2:
            rows, cols = self.shape
            if axis == 0:
                # 沿着第一个轴（行）查找每列的最大值索引
                result = []
                for j in range(cols):
                    max_val = self.data[0][j]
                    max_idx = 0
                    for i in range(rows):
                        if self.data[i][j] > max_val:
                            max_val = self.data[i][j]
                            max_idx = i
                    result.append(max_idx)
                return Array(result)
            else:  # axis == 1
                # 沿着第二个轴（列）查找每行的最大值索引
                result = []
                for i in range(rows):
                    max_val = self.data[i][0]
                    max_idx = 0
                    for j in range(cols):
                        if self.data[i][j] > max_val:
                            max_val = self.data[i][j]
                            max_idx = j
                    result.append(max_idx)
                return Array(result)
        
        raise NotImplementedError("argmax for arrays with more than 2 dimensions not implemented")

    def any(self) -> bool:
        """检查数组中是否有任何True值
        
        Returns:
            如果有任何True值则返回True，否则返回False
        """
        if isinstance(self.data[0], list):
            # 2D数组情况
            for row in self.data:
                for val in row:
                    if val:
                        return True
            return False
        else:
            # 1D数组情况
            for val in self.data:
                if val:
                    return True
            return False

# 工厂函数
def array(data: Union[List, Tuple, float, int, Array], dtype=None) -> Array:
    return Array(data, dtype=dtype)

def zeros(shape: Union[int, Tuple[int, ...]], dtype: str = 'float32') -> Array:
    """
    创建指定形状的全零数组
    
    Args:
        shape: 数组形状，可以是整数或整数元组
        dtype: 数据类型，默认为'float32'
        
    Returns:
        Array: 全零数组
        
    Raises:
        ValueError: 如果形状参数无效
    """
    # 验证形状参数
    if isinstance(shape, int):
        if shape <= 0:
            # 对于非正数，创建最小的有效形状
            shape = (1,)
        else:
            shape = (shape,)
    elif isinstance(shape, tuple):
        # 修复非正数维度
        fixed_shape = []
        for dim in shape:
            if not isinstance(dim, int) or dim <= 0:
                fixed_shape.append(1)  # 将非正数维度改为1
            else:
                fixed_shape.append(dim)
        shape = tuple(fixed_shape)
    else:
        raise ValueError(f"shape必须是整数或整数元组，得到: {type(shape)}")
        
    # 创建数组
    try:
        # 计算总大小
        if isinstance(shape, int):
            total_size = shape
        else:
            total_size = 1
            for dim in shape:
                total_size *= dim
                
        # 创建全零数据
        data = [0.0] * total_size
        result = Array(data, dtype=dtype)
        
        # 如果是多维，重塑形状
        if isinstance(shape, tuple) and len(shape) > 1:
            result = result.reshape(*shape)
            
        return result
    except Exception as e:
        raise RuntimeError(f"创建零数组失败: {str(e)}")

def ones(shape: Union[int, Tuple[int, ...]], dtype: str = 'float32') -> Array:
    """
    创建指定形状的全1数组
    
    Args:
        shape: 数组形状，可以是整数或整数元组
        dtype: 数据类型，默认为'float32'
        
    Returns:
        Array: 全1数组
        
    Raises:
        ValueError: 如果形状参数无效
    """
    # 验证形状参数
    if isinstance(shape, int):
        if shape <= 0:
            # 对于非正数，创建最小的有效形状
            shape = (1,)
        else:
            shape = (shape,)
    elif isinstance(shape, tuple):
        # 修复非正数维度
        fixed_shape = []
        for dim in shape:
            if not isinstance(dim, int) or dim <= 0:
                fixed_shape.append(1)  # 将非正数维度改为1
            else:
                fixed_shape.append(dim)
        shape = tuple(fixed_shape)
    else:
        raise ValueError(f"shape必须是整数或整数元组，得到: {type(shape)}")
        
    # 创建数组
    try:
        # 计算总大小
        if isinstance(shape, int):
            total_size = shape
        else:
            total_size = 1
            for dim in shape:
                total_size *= dim
                
        # 创建全1数据
        data = [1.0] * total_size
        result = Array(data, dtype=dtype)
        
        # 如果是多维，重塑形状
        if isinstance(shape, tuple) and len(shape) > 1:
            result = result.reshape(*shape)
            
        return result
    except Exception as e:
        raise RuntimeError(f"创建全1数组失败: {str(e)}")

def empty(shape: Union[int, Tuple[int, ...]], dtype=None) -> Array:
    return zeros(shape, dtype=dtype)

def random_normal(shape: Union[int, Tuple[int, ...]], mean=0.0, std=1.0, dtype=None) -> Array:
    """生成服从正态分布的随机数组
    
    Args:
        shape: 输出数组的形状，可以是整数或元组
        mean: 正态分布的均值
        std: 正态分布的标准差
        dtype: 数据类型
        
    Returns:
        生成的随机数组
        
    Raises:
        ValueError: 如果形状参数无效
        RuntimeError: 如果生成随机数失败
    """
    # 验证输入
    if isinstance(shape, int):
        if shape <= 0:
            raise ValueError(f"shape必须为正数，得到: {shape}")
        shape = (shape,)
    elif isinstance(shape, tuple):
        if not shape:
            raise ValueError(f"shape不能为空，得到: {shape}")
        for dim in shape:
            if dim <= 0:
                raise ValueError(f"shape的所有维度必须为正数，得到: {shape}")
    else:
        raise ValueError(f"shape必须是整数或整数元组，得到: {type(shape)}")
            
    # 生成随机数据
    try:
        # 计算总大小
        if isinstance(shape, int):
            total_size = shape
        else:
            total_size = 1
            for dim in shape:
                total_size *= dim
                
        # 使用批量生成提高效率
        data = pure_random.normal_batch(total_size, mean, std)
        
        # 直接创建Array对象，跳过复杂的验证
        result = Array.__new__(Array)
        result.data = data
        result.shape = shape
        result.dtype = dtype or float
        
        return result
    except Exception as e:
        raise RuntimeError(f"生成随机数失败: {str(e)}")

def random_uniform(shape: Union[int, Tuple[int, ...]], low=0.0, high=1.0, dtype=None) -> Array:
    if isinstance(shape, int):
        size = shape
    else:
        size = 1
        for dim in shape:
            size *= dim
    data = pure_random.uniform_batch(size, low, high)  # 使用批量生成提高效率
    
    # 直接创建Array对象，跳过复杂的验证
    result = Array.__new__(Array)
    result.data = data
    result.shape = shape if isinstance(shape, tuple) else (shape,)
    result.dtype = dtype or float
    
    return result

def randn(*shape: int, dtype=None) -> Array:
    """生成标准正态分布的随机数组 (均值=0, 标准差=1)
    
    Args:
        *shape: 输出数组的形状，可以是多个整数参数
        dtype: 数据类型
        
    Returns:
        生成的标准正态分布随机数组
        
    Examples:
        randn(3, 4)  # 生成3x4的标准正态分布数组
        randn(10)    # 生成长度为10的一维标准正态分布数组
    """
    if len(shape) == 0:
        raise ValueError("至少需要一个形状参数")
    
    # 验证所有形状参数都是正整数
    for dim in shape:
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"所有形状参数必须是正整数，得到: {dim}")
    
    # 如果只有一个参数，创建一维数组
    if len(shape) == 1:
        return random_normal(shape[0], mean=0.0, std=1.0, dtype=dtype)
    else:
        # 多维数组
        return random_normal(shape, mean=0.0, std=1.0, dtype=dtype)

class random:
    
    @staticmethod
    def uniform(low=0.0, high=1.0, size=None):
        """生成均匀分布的随机数组
        
        Args:
            low: 最小值
            high: 最大值
            size: 输出数组的形状，可以是整数、元组或None
            
        Returns:
            均匀分布的随机数组
        """
        if size is None:
            # 返回单个随机数
            return pure_random.uniform(low, high)
        
        if isinstance(size, int):
            size = (size,)
        elif isinstance(size, (list, tuple)):
            size = tuple(size)
        else:
            raise ValueError(f"size必须是整数、元组或None，得到: {type(size)}")
        
        return random_uniform(size, low, high)
    
    @staticmethod
    def randn(*args):
        """生成标准正态分布的随机数组"""
        return randn(*args)
    
    @staticmethod
    def normal(loc=0.0, scale=1.0, size=None):
        """生成正态分布的随机数组
        
        Args:
            loc: 均值
            scale: 标准差
            size: 输出数组的形状
            
        Returns:
            正态分布的随机数组
        """
        if size is None:
            return pure_random.normal(loc, scale)
        
        if isinstance(size, int):
            size = (size,)
        elif isinstance(size, (list, tuple)):
            size = tuple(size)
        else:
            raise ValueError(f"size必须是整数、元组或None，得到: {type(size)}")
        
        return random_normal(size, loc, scale)

def eye(n: int, dtype=None) -> Array:
    data = [0.0] * (n * n)
    for i in range(n):
        data[i * n + i] = 1.0
    return Array(data, dtype=dtype).reshape(n, n)

def linspace(start: float, stop: float, num: int, dtype=None) -> Array:
    step = (stop - start) / (num - 1) if num > 1 else 0
    data = [start + i * step for i in range(num)]
    return Array(data, dtype=dtype)

def arange(start: float, stop: float, step: float = 1.0, dtype=None) -> Array:
    data = []
    current = start
    while current < stop:
        data.append(current)
        current += step
    return Array(data, dtype=dtype)

def concatenate(arrays: List[Array], axis: int = 0) -> Array:
    if not arrays:
        raise ValueError("empty sequence")
    if axis >= len(arrays[0].shape):
        raise ValueError("axis out of bounds")
    # 简化版本：只支持1D和2D数组的连接
    if len(arrays[0].shape) == 1:
        result = []
        for arr in arrays:
            result.extend(arr.data)
        return Array(result, dtype=arrays[0].dtype)
    if len(arrays[0].shape) == 2:
        if axis == 0:
            result = []
            for arr in arrays:
                result.extend(arr.data)
            return Array(result, dtype=arrays[0].dtype).reshape(sum(arr.shape[0] for arr in arrays), arrays[0].shape[1])
        else:
            result = []
            rows = arrays[0].shape[0]
            for i in range(rows):
                for arr in arrays:
                    result.extend(arr.data[i * arr.shape[1]:(i + 1) * arr.shape[1]])
            return Array(result, dtype=arrays[0].dtype).reshape(rows, sum(arr.shape[1] for arr in arrays))
    raise NotImplementedError("concatenate for arrays with more than 2 dimensions not implemented")

def stack(arrays: List[Array], axis: int = 0) -> Array:
    if not arrays:
        raise ValueError("empty sequence")
    if axis > len(arrays[0].shape):
        raise ValueError("axis out of bounds")
    # 简化版本：只支持1D和2D数组的堆叠
    if len(arrays[0].shape) == 1:
        if axis == 0:
            return Array([x for arr in arrays for x in arr.data], dtype=arrays[0].dtype).reshape(len(arrays), -1)
        else:
            return Array([x for arr in arrays for x in arr.data], dtype=arrays[0].dtype).reshape(-1, len(arrays))
    if len(arrays[0].shape) == 2:
        if axis == 0:
            return Array([x for arr in arrays for x in arr.data], dtype=arrays[0].dtype).reshape(len(arrays), *arrays[0].shape)
        elif axis == 1:
            return Array([x for arr in arrays for x in arr.data], dtype=arrays[0].dtype).reshape(arrays[0].shape[0], len(arrays), arrays[0].shape[1])
        else:
            return Array([x for arr in arrays for x in arr.data], dtype=arrays[0].dtype).reshape(arrays[0].shape[0], arrays[0].shape[1], len(arrays))
    raise NotImplementedError("stack for arrays with more than 2 dimensions not implemented")

def maximum(a: Union[Array, float], b: Union[Array, float]) -> Array:
    # 确保输入是Array对象
    if not isinstance(a, Array) and not isinstance(a, (int, float)):
        a = Array(a)
    if not isinstance(b, Array) and not isinstance(b, (int, float)):
        b = Array(b)
        
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return Array([max(a, b)], dtype=float)
    if isinstance(a, (int, float)):
        # a是标量，b是数组
        if len(b.data) > 0 and isinstance(b.data[0], list):  # b是多维数组
            result = []
            for row in b.data:
                result.append([__builtins__['max'](a, x) for x in row])
            return Array(result, dtype=b.dtype)
        else:  # b是一维数组
            return Array([__builtins__['max'](a, x) for x in b.data], dtype=b.dtype)
    if isinstance(b, (int, float)):
        # b是标量，a是数组
        if len(a.data) > 0 and isinstance(a.data[0], list):  # a是多维数组
            result = []
            for row in a.data:
                result.append([__builtins__['max'](x, b) for x in row])
            return Array(result, dtype=a.dtype)
        else:  # a是一维数组
            return Array([__builtins__['max'](x, b) for x in a.data], dtype=a.dtype)
    
    # 两个都是数组
    if a.shape != b.shape:
        raise ValueError("shapes do not match")
    
    # 检查是否为多维数组
    if len(a.data) > 0 and len(b.data) > 0:
        if isinstance(a.data[0], list) and isinstance(b.data[0], list):
            # 两个都是多维数组
            result = []
            for row_a, row_b in zip(a.data, b.data):
                result.append([__builtins__['max'](x, y) for x, y in zip(row_a, row_b)])
            return Array(result, dtype=a.dtype)
        elif not isinstance(a.data[0], list) and not isinstance(b.data[0], list):
            # 两个都是一维数组
            return Array([__builtins__['max'](x, y) for x, y in zip(a.data, b.data)], dtype=a.dtype)
        else:
            raise ValueError("Inconsistent array dimensions")
    else:
        # 处理空数组的情况
        return Array([], dtype=a.dtype if hasattr(a, 'dtype') else float)

def minimum(a: Union[Array, float], b: Union[Array, float]) -> Array:
    # 确保输入是Array对象
    if not isinstance(a, Array) and not isinstance(a, (int, float)):
        a = Array(a)
    if not isinstance(b, Array) and not isinstance(b, (int, float)):
        b = Array(b)
        
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return Array([__builtins__['min'](a, b)], dtype=float)
    if isinstance(a, (int, float)):
        # a是标量，b是数组
        if len(b.data) > 0 and isinstance(b.data[0], list):  # b是多维数组
            result = []
            for row in b.data:
                result.append([__builtins__['min'](a, x) for x in row])
            return Array(result, dtype=b.dtype)
        else:  # b是一维数组
            return Array([__builtins__['min'](a, x) for x in b.data], dtype=b.dtype)
    if isinstance(b, (int, float)):
        # b是标量，a是数组
        if len(a.data) > 0 and isinstance(a.data[0], list):  # a是多维数组
            result = []
            for row in a.data:
                result.append([__builtins__['min'](x, b) for x in row])
            return Array(result, dtype=a.dtype)
        else:  # a是一维数组
            return Array([__builtins__['min'](x, b) for x in a.data], dtype=a.dtype)
    
    # 两个都是数组
    if a.shape != b.shape:
        raise ValueError("shapes do not match")
    
    # 检查是否为多维数组
    if len(a.data) > 0 and len(b.data) > 0:
        if isinstance(a.data[0], list) and isinstance(b.data[0], list):
            # 两个都是多维数组
            result = []
            for row_a, row_b in zip(a.data, b.data):
                result.append([__builtins__['min'](x, y) for x, y in zip(row_a, row_b)])
            return Array(result, dtype=a.dtype)
        elif not isinstance(a.data[0], list) and not isinstance(b.data[0], list):
            # 两个都是一维数组
            return Array([__builtins__['min'](x, y) for x, y in zip(a.data, b.data)], dtype=a.dtype)
        else:
            raise ValueError("Inconsistent array dimensions")
    else:
        # 处理空数组的情况
        return Array([], dtype=a.dtype if hasattr(a, 'dtype') else float)

def where(condition: Array, x: Union[Array, float], y: Union[Array, float]) -> Array:
    if not isinstance(condition, Array):
        raise TypeError("condition must be an Array")
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return Array([x if c else y for c in condition.data], dtype=float)
    if isinstance(x, (int, float)):
        if condition.shape != y.shape:
            raise ValueError("shapes do not match")
        return Array([x if c else y_val for c, y_val in zip(condition.data, y.data)], dtype=y.dtype)
    if isinstance(y, (int, float)):
        if condition.shape != x.shape:
            raise ValueError("shapes do not match")
        return Array([x_val if c else y for c, x_val in zip(condition.data, x.data)], dtype=x.dtype)
    if condition.shape != x.shape or condition.shape != y.shape:
        raise ValueError("shapes do not match")
    return Array([x_val if c else y_val for c, x_val, y_val in zip(condition.data, x.data, y.data)], dtype=x.dtype)

def einsum(equation: str, *arrays: Array) -> Array:
    # 简化版本：只支持基本的矩阵乘法
    if equation == 'ij,jk->ik':
        if len(arrays) != 2:
            raise ValueError("einsum requires 2 arrays for matrix multiplication")
        return arrays[0].dot(arrays[1])
    raise NotImplementedError("only basic matrix multiplication is supported for einsum")

def bmm(x: Array, y: Array) -> Array:
    if len(x.shape) != 3 or len(y.shape) != 3:
        raise ValueError("bmm requires 3D arrays")
    if x.shape[0] != y.shape[0] or x.shape[2] != y.shape[1]:
        raise ValueError("shapes not aligned for batch matrix multiplication")
    result = []
    for i in range(x.shape[0]):
        x_i = Array(x.data[i * x.shape[1] * x.shape[2]:(i + 1) * x.shape[1] * x.shape[2]], dtype=x.dtype).reshape(x.shape[1], x.shape[2])
        y_i = Array(y.data[i * y.shape[1] * y.shape[2]:(i + 1) * y.shape[1] * y.shape[2]], dtype=y.dtype).reshape(y.shape[1], y.shape[2])
        result.extend(x_i.dot(y_i).data)
    return Array(result, dtype=x.dtype).reshape(x.shape[0], x.shape[1], y.shape[2])

def conv2d(input: Array, weight: Array, bias: Optional[Array] = None, stride: Tuple[int, int] = (1, 1), padding: Tuple[int, int] = (0, 0)) -> Array:
    if len(input.shape) != 4 or len(weight.shape) != 4:
        raise ValueError("conv2d requires 4D arrays")
    if input.shape[1] != weight.shape[1]:
        raise ValueError("input channels must match weight channels")
    if bias is not None and bias.shape[0] != weight.shape[0]:
        raise ValueError("bias size must match output channels")
    
    batch_size, in_channels, in_height, in_width = input.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    
    out_height = (in_height + 2 * pad_h - kernel_height) // stride_h + 1
    out_width = (in_width + 2 * pad_w - kernel_width) // stride_w + 1
    
    result = []
    for b in range(batch_size):
        for oc in range(out_channels):
            for oh in range(out_height):
                for ow in range(out_width):
                    sum_val = 0.0
                    for ic in range(in_channels):
                        for kh in range(kernel_height):
                            for kw in range(kernel_width):
                                ih = oh * stride_h + kh - pad_h
                                iw = ow * stride_w + kw - pad_w
                                if 0 <= ih < in_height and 0 <= iw < in_width:
                                    input_idx = b * in_channels * in_height * in_width + ic * in_height * in_width + ih * in_width + iw
                                    weight_idx = oc * in_channels * kernel_height * kernel_width + ic * kernel_height * kernel_width + kh * kernel_width + kw
                                    sum_val += input.data[input_idx] * weight.data[weight_idx]
                    if bias is not None:
                        sum_val += bias.data[oc]
                    result.append(sum_val)
    
    return Array(result, dtype=input.dtype).reshape(batch_size, out_channels, out_height, out_width)

def mean(x: Union[Array, List, Tuple], axis: Optional[int] = None) -> Union[Array, float]:
    """计算数组的平均值
    
    Args:
        x: 输入数组
        axis: 计算平均值的轴，如果为None则计算所有元素
        
    Returns:
        平均值
    """
    if isinstance(x, Array):
        return x.mean(axis)
    return __builtins__['sum'](x) / len(x)

def std(x: Union[Array, List, Tuple], axis: Optional[int] = None) -> Union[Array, float]:
    """计算标准差"""
    if isinstance(x, Array):
        return x.var(axis=axis) ** 0.5
    x_array = array(x)
    return x_array.var(axis=axis) ** 0.5

def histogram(x: Union[Array, List, Tuple], bins: int = 10) -> Tuple[Array, Array]:
    """计算直方图
    
    Args:
        x: 输入数组
        bins: 直方图的箱数
        
    Returns:
        (hist, bin_edges): 直方图值和箱边缘
    """
    if isinstance(x, Array):
        data = x.data
    else:
        data = x
        
    # 计算数据范围
    min_val = __builtins__['min'](data)
    max_val = __builtins__['max'](data)
    
    # 计算箱边缘
    bin_edges = linspace(min_val, max_val, bins + 1)
    
    # 计算每个箱的计数
    hist = zeros(bins)
    for val in data:
        for i in range(bins):
            if bin_edges[i] <= val < bin_edges[i + 1]:
                hist.data[i] += 1
                break
    
    return Array(hist.data), Array(bin_edges.data)

def sqrt(x: Union[Array, float, List[float]]) -> Array:
    """计算数组的平方根 - 高效版本
    
    Args:
        x: 输入数组或数值
        
    Returns:
        平方根数组
    """
    if isinstance(x, (int, float)):
        if x < 0:
            raise ValueError("不能对负数计算平方根")
        return array([math.sqrt(x)])
    
    if isinstance(x, list):
        x = array(x)
    
    if hasattr(x, 'shape') and hasattr(x, 'flatten'):
        try:
            import strong_sqrt
            strong_result = strong_sqrt.replace_np_sqrt(x)
            if len(x.shape) == 0:
                return array([float(strong_result)])
            elif len(x.shape) == 1:
                return array(strong_result.tolist() if hasattr(strong_result, 'tolist') else strong_result)
            else:
                return array(strong_result.tolist() if hasattr(strong_result, 'tolist') else strong_result)
        except Exception as e:
            print(f"Warning: strong_sqrt failed, falling back to math.sqrt: {e}")
            pass
    
    if isinstance(x, Array):
        # 使用递归方法处理任意维度的数组 - 优化版本
        def sqrt_recursive(data):
            if isinstance(data, list):
                if len(data) > 0 and isinstance(data[0], list):
                    # 嵌套列表，递归处理
                    return [sqrt_recursive(item) for item in data]
                else:
                    # 一维列表，批量处理
                    return [math.sqrt(float(val)) for val in data]
            else:
                # 单个值
                val = float(data)
                if val < 0:
                    raise ValueError("不能对负数计算平方根")
                return math.sqrt(val)
        
        try:
            result_data = sqrt_recursive(x.data)
            return array(result_data)
        except Exception as e:
            # 如果出错，使用原来的逐个处理方法
            if isinstance(x.data, list):
                if len(x.data) > 0 and isinstance(x.data[0], list):  # 处理嵌套列表
                    result = []
                    for row in x.data:
                        row_result = []
                        for val in row:
                            if val < 0:
                                raise ValueError("不能对负数计算平方根")
                            row_result.append(math.sqrt(val))
                        result.append(row_result)
                    return array(result)
                else:  # 处理一维列表
                    result = []
                    for val in x.data:
                        if val < 0:
                            raise ValueError("不能对负数计算平方根")
                        result.append(math.sqrt(val))
                    return array(result)
            else:  # 处理单个值
                if x.data < 0:
                    raise ValueError("不能对负数计算平方根")
                return array([math.sqrt(x.data)])
    
    raise TypeError(f"unsupported operand type(s) for sqrt: '{type(x)}'")

def exp(x: Union[Array, float, List[float]]) -> Array:
    """计算数组的指数
    
    Args:
        x: 输入数组或数值
        
    Returns:
        指数数组
    """
    if isinstance(x, (int, float)):
        return array([math.exp(x)])
    
    if isinstance(x, list):
        x = array(x)
    
    if isinstance(x, Array):
        # 使用递归方法处理任意维度的数组
        def exp_recursive(data):
            if isinstance(data, list):
                return [exp_recursive(item) for item in data]
            elif hasattr(data, 'data'):
                # 处理嵌套的Array对象
                return exp_recursive(data.data)
            else:
                try:
                    return math.exp(float(data))
                except (ValueError, TypeError):
                    return 0.0
        
        result_data = exp_recursive(x.data)
        return array(result_data)
    
    raise TypeError(f"unsupported operand type(s) for exp: '{type(x)}'")

def log(x: Union[Array, float, List[float]]) -> Array:
    """计算数组的自然对数
    
    Args:
        x: 输入数组或数值
        
    Returns:
        自然对数数组
    """
    if isinstance(x, (int, float)):
        if x <= 0:
            raise ValueError("不能对非正数计算对数")
        return array([math.log(x)])
    
    if isinstance(x, list):
        x = array(x)
    
    result = []
    for val in x.data:
        if val <= 0:
            raise ValueError("不能对非正数计算对数")
        result.append(math.log(val))
    return array(result)

def argmax(x: Array, axis: Optional[int] = None) -> Union[Array, int]:
    """返回数组中最大值的索引
    
    Args:
        x: 输入数组
        axis: 计算最大值索引的轴，如果为None则返回全局最大值索引
        
    Returns:
        最大值的索引
    """
    if not isinstance(x, Array):
        raise TypeError("输入必须是Array类型")
    
    return x.argmax(axis)

def zeros_like(x: Array, dtype: Optional[type] = None) -> Array:
    """创建与输入数组形状相同的零数组
    
    Args:
        x: 输入数组
        dtype: 数据类型
        
    Returns:
        零数组
    """
    if not isinstance(x, Array):
        x = Array(x)
    return zeros(x.shape, dtype=dtype or x.dtype)

def ones_like(x: Array, dtype: Optional[type] = None) -> Array:
    """创建与输入数组形状相同的一数组
    
    Args:
        x: 输入数组
        dtype: 数据类型
        
    Returns:
        一数组
    """
    if not isinstance(x, Array):
        x = Array(x)
    return ones(x.shape, dtype=dtype or x.dtype)

def max(x: Array, axis: Optional[int] = None, keepdims: bool = False) -> Union[Array, float]:
    """计算最大值
    
    Args:
        x: 输入数组
        axis: 计算轴
        keepdims: 是否保持维度
        
    Returns:
        最大值
    """
    if not isinstance(x, Array):
        x = Array(x)
    # 直接使用Array类的max方法，它已经正确处理了keepdims
    return x.max(axis, keepdims)

def min(x: Array, axis: Optional[int] = None, keepdims: bool = False) -> Union[Array, float]:
    """计算最小值
    
    Args:
        x: 输入数组
        axis: 计算轴
        keepdims: 是否保持维度
        
    Returns:
        最小值
    """
    if not isinstance(x, Array):
        x = Array(x)
    result = x.min(axis)
    if axis is not None and keepdims and isinstance(result, Array):
        # 如果需要保持维度，重塑结果
        new_shape = list(x.shape)
        new_shape[axis] = 1
        result = result.reshape(*new_shape)
    return result

def sum(x: Array, axis: Optional[int] = None, keepdims: bool = False) -> Union[Array, float]:
    """计算总和
    
    Args:
        x: 输入数组
        axis: 计算轴
        keepdims: 是否保持维度
        
    Returns:
        总和
    """
    if not isinstance(x, Array):
        x = Array(x)
    return x.sum(axis, keepdims)

def asarray(data, dtype=None) -> Array:
    """将输入转换为数组，增强对特殊对象的处理
    
    Args:
        data: 输入数据
        dtype: 数据类型
        
    Returns:
        数组
    """
    # 处理特殊对象（如memoryview）
    if hasattr(data, '__array__'):
        # 如果对象有__array__方法，先转换为数组
        import memasarray
        converted_data = memasarray.ult_asarray(data)
        return Array(converted_data, dtype=dtype)

    
    # 处理memoryview对象
    if isinstance(data, memoryview):
        import memasarray
        converted_data = memasarray.ult_asarray(data)
        return Array(converted_data, dtype=dtype)

    # 处理其他特殊对象
    if hasattr(data, 'tolist'):
        try:
            return Array(data.tolist(), dtype=dtype)
        except:
            pass
    
    if hasattr(data, 'item'):
        try:
            return Array([data.item()], dtype=dtype)
        except:
            pass
    
    # 默认处理
    try:
        return Array(data, dtype=dtype)
    except Exception as e:
        # 如果所有方法都失败，返回一个安全的默认值
        print(f"Warning: asarray转换失败，使用默认值: {e}")
        return Array([0.0], dtype=dtype)

def isnan(x: Array) -> Array:
    """检查是否为NaN
    
    Args:
        x: 输入数组
        
    Returns:
        布尔数组
    """
    if not isinstance(x, Array):
        x = Array(x)
    
    result = []
    for val in x.data:
        result.append(math.isnan(val) if isinstance(val, (int, float)) else False)
    return Array(result)

def isinf(x: Array) -> Array:
    """检查是否为无穷大
    
    Args:
        x: 输入数组
        
    Returns:
        布尔数组
    """
    if not isinstance(x, Array):
        x = Array(x)
    
    result = []
    for val in x.data:
        result.append(math.isinf(val) if isinstance(val, (int, float)) else False)
    return Array(result)

def any(x: Array) -> bool:
    """检查是否有任何True值
    
    Args:
        x: 输入数组
        
    Returns:
        布尔值
    """
    if not isinstance(x, Array):
        x = Array(x)
    
    for val in x.data:
        if val:
            return True
    return False

def tanh(x: Union[Array, float, List[float]]) -> Array:
    """计算双曲正切函数
    
    Args:
        x: 输入数组或数值
        
    Returns:
        双曲正切函数值数组
    """
    if isinstance(x, (int, float)):
        # 使用数学定义: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
        exp_x = math.exp(x)
        exp_neg_x = math.exp(-x)
        return array([(exp_x - exp_neg_x) / (exp_x + exp_neg_x)])
    
    if isinstance(x, list):
        x = array(x)
    
    if isinstance(x, Array):
        if isinstance(x.data, list):
            if isinstance(x.data[0], list):  # 处理嵌套列表
                result = []
                for row in x.data:
                    row_result = []
                    for val in row:
                        exp_val = math.exp(val)
                        exp_neg_val = math.exp(-val)
                        row_result.append((exp_val - exp_neg_val) / (exp_val + exp_neg_val))
                    result.append(row_result)
                return array(result)
            else:  # 处理一维列表
                result = []
                for val in x.data:
                    exp_val = math.exp(val)
                    exp_neg_val = math.exp(-val)
                    result.append((exp_val - exp_neg_val) / (exp_val + exp_neg_val))
                return array(result)
        else:  # 处理单个值
            exp_val = math.exp(x.data)
            exp_neg_val = math.exp(-x.data)
            return array([(exp_val - exp_neg_val) / (exp_val + exp_neg_val)])
    
    raise TypeError(f"unsupported operand type(s) for tanh: '{type(x)}'")

def erf(x: Union[Array, float, List[float]]) -> Array:
    """计算误差函数（错误函数）
    使用Abramowitz和Stegun的近似公式
    
    Args:
        x: 输入数组或数值
        
    Returns:
        误差函数值数组
    """
    def _erf_single(x_val):
        """单个值的误差函数计算"""
        # 使用Abramowitz和Stegun的近似公式
        # erf(x) ≈ 1 - (a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5)*exp(-x^2)
        # 其中 t = 1 / (1 + p*|x|), p = 0.3275911
        
        # 系数
        a1 =  0.254829592
        a2 = -0.284496736
        a3 =  1.421413741
        a4 = -1.453152027
        a5 =  1.061405429
        p  =  0.3275911
        
        # 保存符号
        sign = 1 if x_val >= 0 else -1
        x_abs = abs(x_val)
        
        # 计算
        t = 1.0 / (1.0 + p * x_abs)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x_abs * x_abs)
        
        return sign * y
    
    if isinstance(x, (int, float)):
        return array([_erf_single(x)])
    
    if isinstance(x, list):
        x = array(x)
    
    if isinstance(x, Array):
        if isinstance(x.data, list):
            if isinstance(x.data[0], list):  # 处理嵌套列表
                result = []
                for row in x.data:
                    row_result = []
                    for val in row:
                        row_result.append(_erf_single(val))
                    result.append(row_result)
                return array(result)
            else:  # 处理一维列表
                result = []
                for val in x.data:
                    result.append(_erf_single(val))
                return array(result)
        else:  # 处理单个值
            return array([_erf_single(x.data)])
    
    raise TypeError(f"unsupported operand type(s) for erf: '{type(x)}'")

def matmul(a: Union[Array, List, Tuple], b: Union[Array, List, Tuple]) -> Array:
    """矩阵乘法
    
    Args:
        a: 第一个矩阵
        b: 第二个矩阵
        
    Returns:
        矩阵乘法的结果
    """
    # 确保输入是Array对象
    if not isinstance(a, Array):
        a = Array(a)
    if not isinstance(b, Array):
        b = Array(b)
    
    # 使用Array类的dot方法（已实现的矩阵乘法）
    return a.dot(b)

def broadcast_to(array: Union[Array, List, Tuple], shape: Tuple[int, ...]) -> Array:
    """将数组广播到指定形状
    
    Args:
        array: 输入数组
        shape: 目标形状
        
    Returns:
        广播后的数组
    """
    if not isinstance(array, Array):
        array = Array(array)
    
    # 如果形状已经匹配，直接返回
    if array.shape == shape:
        return array.copy()
    
    # 检查是否可以广播
    if len(array.shape) > len(shape):
        raise ValueError(f"cannot broadcast array with shape {array.shape} to shape {shape}")
    
    # 从右边开始比较维度
    for i in range(1, len(array.shape) + 1):
        if array.shape[-i] != 1 and array.shape[-i] != shape[-i]:
            raise ValueError(f"cannot broadcast array with shape {array.shape} to shape {shape}")
    
    # 执行广播
    result_data = []
    
    if len(shape) == 1:
        # 1D广播
        if len(array.shape) == 1:
            if array.shape[0] == 1:
                # 从单元素广播到多元素
                result_data = [array.data[0]] * shape[0]
            else:
                result_data = array.data.copy()
        else:
            raise ValueError(f"cannot broadcast array with shape {array.shape} to shape {shape}")
    
    elif len(shape) == 2:
        # 2D广播
        rows, cols = shape
        if len(array.shape) == 1:
            if array.shape[0] == 1:
                # 从(1,)广播到(rows, cols)
                result_data = [[array.data[0]] * cols for _ in range(rows)]
            elif array.shape[0] == cols:
                # 从(cols,)广播到(rows, cols)
                result_data = [array.data.copy() for _ in range(rows)]
            else:
                raise ValueError(f"cannot broadcast array with shape {array.shape} to shape {shape}")
        elif len(array.shape) == 2:
            if array.shape == (1, 1):
                # 从(1,1)广播到(rows, cols)
                result_data = [[array.data[0][0]] * cols for _ in range(rows)]
            elif array.shape[0] == 1 and array.shape[1] == cols:
                # 从(1, cols)广播到(rows, cols)
                result_data = [array.data[0].copy() for _ in range(rows)]
            elif array.shape[0] == rows and array.shape[1] == 1:
                # 从(rows, 1)广播到(rows, cols)
                result_data = [[array.data[i][0]] * cols for i in range(rows)]
            else:
                result_data = array.data.copy()
        else:
            raise ValueError(f"cannot broadcast array with shape {array.shape} to shape {shape}")
    
    elif len(shape) == 3:
        # 3D广播（增强实现）
        d0, d1, d2 = shape
        if len(array.shape) == 1:
            if array.shape[0] == 1:
                # 从(1,)广播到(d0, d1, d2)
                result_data = [[[array.data[0]] * d2 for _ in range(d1)] for _ in range(d0)]
            elif array.shape[0] == d2:
                # 从(d2,)广播到(d0, d1, d2)
                result_data = [[[array.data[k] for k in range(d2)] for _ in range(d1)] for _ in range(d0)]
            else:
                raise ValueError(f"cannot broadcast array with shape {array.shape} to shape {shape}")
        elif len(array.shape) == 2:
            if array.shape == (1, 1):
                # 从(1,1)广播到(d0, d1, d2)
                result_data = [[[array.data[0][0]] * d2 for _ in range(d1)] for _ in range(d0)]
            elif array.shape[0] == d0 and array.shape[1] == 1:
                # 从(d0, 1)广播到(d0, d1, d2)
                result_data = [[[array.data[i][0]] * d2 for _ in range(d1)] for i in range(d0)]
            elif array.shape[0] == 1 and array.shape[1] == d2:
                # 从(1, d2)广播到(d0, d1, d2)
                result_data = [[[array.data[0][k] for k in range(d2)] for _ in range(d1)] for _ in range(d0)]
            elif array.shape[0] == d1 and array.shape[1] == d2:
                # 从(d1, d2)广播到(d0, d1, d2)
                result_data = [[array.data[j] for j in range(d1)] for _ in range(d0)]
            else:
                raise ValueError(f"cannot broadcast array with shape {array.shape} to shape {shape}")
        elif len(array.shape) == 3:
            if array.shape[0] == d0 and array.shape[1] == 1 and array.shape[2] == 1:
                # 从(d0, 1, 1)广播到(d0, d1, d2)
                result_data = [[[array.data[i][0][0]] * d2 for _ in range(d1)] for i in range(d0)]
            elif array.shape[0] == 1 and array.shape[1] == d1 and array.shape[2] == 1:
                # 从(1, d1, 1)广播到(d0, d1, d2)
                result_data = [[[array.data[0][j][0]] * d2 for j in range(d1)] for _ in range(d0)]
            elif array.shape[0] == 1 and array.shape[1] == 1 and array.shape[2] == d2:
                # 从(1, 1, d2)广播到(d0, d1, d2)
                result_data = [[[array.data[0][0][k] for k in range(d2)] for _ in range(d1)] for _ in range(d0)]
            elif array.shape[0] == d0 and array.shape[1] == d1 and array.shape[2] == 1:
                # 从(d0, d1, 1)广播到(d0, d1, d2)
                result_data = [[[array.data[i][j][0]] * d2 for j in range(d1)] for i in range(d0)]
            elif array.shape[0] == d0 and array.shape[1] == 1 and array.shape[2] == d2:
                # 从(d0, 1, d2)广播到(d0, d1, d2)
                result_data = [[[array.data[i][0][k] for k in range(d2)] for _ in range(d1)] for i in range(d0)]
            elif array.shape[0] == 1 and array.shape[1] == d1 and array.shape[2] == d2:
                # 从(1, d1, d2)广播到(d0, d1, d2)
                result_data = [[array.data[0][j] for j in range(d1)] for _ in range(d0)]
            elif array.shape == shape:
                # 形状已经匹配
                result_data = array.data
            else:
                raise ValueError(f"cannot broadcast array with shape {array.shape} to shape {shape}")
        else:
            raise ValueError(f"cannot broadcast array with shape {array.shape} to shape {shape}")
    
    else:
        # 更高维度的广播（通用实现）
        # 使用递归方法处理任意维度的广播
        def broadcast_recursive(arr_shape, target_shape, data):
            if len(arr_shape) == 0:
                # 标量情况
                if len(target_shape) == 0:
                    return data
                else:
                    # 标量广播到多维
                    result = data
                    for dim_size in reversed(target_shape):
                        result = [result] * dim_size
                    return result
            elif len(arr_shape) == 1:
                # 一维数组
                if arr_shape[0] == 1:
                    # 从(1,)广播
                    result = data[0]
                    for dim_size in reversed(target_shape):
                        result = [result] * dim_size
                    return result
                elif arr_shape[0] == target_shape[-1]:
                    # 从(n,)广播到(..., n)
                    result = data
                    for dim_size in reversed(target_shape[:-1]):
                        result = [result] * dim_size
                    return result
                else:
                    raise ValueError(f"cannot broadcast array with shape {arr_shape} to shape {target_shape}")
            else:
                # 多维数组，递归处理
                if arr_shape[0] == 1:
                    # 第一维是1，可以广播
                    inner_result = broadcast_recursive(arr_shape[1:], target_shape[1:], data[0])
                    return [inner_result] * target_shape[0]
                elif arr_shape[0] == target_shape[0]:
                    # 第一维匹配，递归处理剩余维度
                    result = []
                    for i in range(arr_shape[0]):
                        inner_result = broadcast_recursive(arr_shape[1:], target_shape[1:], data[i])
                        result.append(inner_result)
                    return result
                else:
                    raise ValueError(f"cannot broadcast array with shape {arr_shape} to shape {target_shape}")
        
        try:
            result_data = broadcast_recursive(array.shape, shape, array.data)
        except Exception as e:
            raise ValueError(f"cannot broadcast array with shape {array.shape} to shape {shape}: {e}")
    
    return Array(result_data)

def expand_dims(array: Union[Array, List, Tuple], axis: int) -> Array:
    """在指定位置增加一个维度
    
    Args:
        array: 输入数组
        axis: 要插入新维度的位置
        
    Returns:
        扩展维度后的数组
    """
    if not isinstance(array, Array):
        array = Array(array)
    
    # 处理负数轴
    if axis < 0:
        axis = len(array.shape) + axis + 1
    
    # 验证轴的有效性
    if axis < 0 or axis > len(array.shape):
        raise ValueError(f"axis {axis} is out of bounds for array of dimension {len(array.shape)}")
    
    # 创建新的形状
    new_shape = list(array.shape)
    new_shape.insert(axis, 1)
    
    # 重塑数组
    result = array.copy()
    result.shape = tuple(new_shape)
    
    # 如果需要重新组织数据结构
    if axis == 0 and len(array.shape) == 1:
        # 在最前面添加维度：(n,) -> (1, n)
        result.data = [array.data]
    elif axis == 1 and len(array.shape) == 1:
        # 在第二个位置添加维度：(n,) -> (n, 1)
        result.data = [[x] for x in array.data]
    elif len(array.shape) == 2 and axis == 0:
        # 在3D数组的第一个位置添加维度：(m, n) -> (1, m, n)
        result.data = [array.data]
    elif len(array.shape) == 2 and axis == 1:
        # 在3D数组的第二个位置添加维度：(m, n) -> (m, 1, n)
        result.data = [[row] for row in array.data]
    elif len(array.shape) == 2 and axis == 2:
        # 在3D数组的第三个位置添加维度：(m, n) -> (m, n, 1)
        result.data = [[[x] for x in row] for row in array.data]
    
    return result

def power(base: Union[Array, float, List[float]], exponent: Union[Array, float, List[float]]) -> Array:
    """
    计算数组的幂运算，增强错误处理
    
    Args:
        base: 底数
        exponent: 指数
        
    Returns:
        幂运算结果
    """
    if not isinstance(base, Array):
        base = Array(base)
    
    if isinstance(exponent, (int, float)):
        # 标量指数
        try:
            if isinstance(base.data[0], list):
                # 多维数组
                def power_recursive(data):
                    if isinstance(data[0], list):
                        return [power_recursive(row) for row in data]
                    else:
                        result = []
                        for x in data:
                            try:
                                # 处理特殊情况
                                if x == 0 and exponent < 0:
                                    result.append(float('inf'))
                                elif x < 0 and not isinstance(exponent, int) and exponent != int(exponent):
                                    # 负数的非整数幂，取绝对值
                                    result.append(abs(x) ** exponent)
                                else:
                                    result.append(x ** exponent)
                            except (OverflowError, ZeroDivisionError, ValueError):
                                # 处理溢出、除零等错误
                                if x == 0:
                                    result.append(0.0 if exponent > 0 else float('inf'))
                                elif abs(x) > 1 and exponent > 100:
                                    result.append(float('inf') if x > 0 else float('-inf'))
                                elif abs(x) < 1 and exponent < -100:
                                    result.append(float('inf'))
                                else:
                                    result.append(1.0)  # 安全默认值
                        return result
                result_data = power_recursive(base.data)
            else:
                # 一维数组
                result_data = []
                for x in base.data:
                    try:
                        # 处理特殊情况
                        if x == 0 and exponent < 0:
                            result_data.append(float('inf'))
                        elif x < 0 and not isinstance(exponent, int) and exponent != int(exponent):
                            # 负数的非整数幂，取绝对值
                            result_data.append(abs(x) ** exponent)
                        else:
                            result_data.append(x ** exponent)
                    except (OverflowError, ZeroDivisionError, ValueError):
                        # 处理溢出、除零等错误
                        if x == 0:
                            result_data.append(0.0 if exponent > 0 else float('inf'))
                        elif abs(x) > 1 and exponent > 100:
                            result_data.append(float('inf') if x > 0 else float('-inf'))
                        elif abs(x) < 1 and exponent < -100:
                            result_data.append(float('inf'))
                        else:
                            result_data.append(1.0)  # 安全默认值
            
            result = Array(result_data, dtype=base.dtype)
            return result
        except Exception as e:
            # 完全失败时的后备方案
            print(f"Power计算失败，使用安全默认值: {e}")
            if isinstance(base.data[0], list):
                result_data = [[1.0 for _ in row] for row in base.data]
            else:
                result_data = [1.0 for _ in base.data]
            return Array(result_data, dtype=base.dtype)
    
    elif isinstance(exponent, Array):
        # 数组指数
        if base.shape != exponent.shape:
            raise ValueError(f"shapes do not match for power: {base.shape} vs {exponent.shape}")
        
        try:
            if isinstance(base.data[0], list) and isinstance(exponent.data[0], list):
                # 两个都是多维数组
                result_data = []
                for base_row, exp_row in zip(base.data, exponent.data):
                    result_row = []
                    for b, e in zip(base_row, exp_row):
                        try:
                            if b == 0 and e < 0:
                                result_row.append(float('inf'))
                            elif b < 0 and not isinstance(e, int) and e != int(e):
                                result_row.append(abs(b) ** e)
                            else:
                                result_row.append(b ** e)
                        except (OverflowError, ZeroDivisionError, ValueError):
                            if b == 0:
                                result_row.append(0.0 if e > 0 else float('inf'))
                            elif abs(b) > 1 and e > 100:
                                result_row.append(float('inf') if b > 0 else float('-inf'))
                            elif abs(b) < 1 and e < -100:
                                result_row.append(float('inf'))
                            else:
                                result_row.append(1.0)
                    result_data.append(result_row)
            else:
                # 一维数组
                result_data = []
                for b, e in zip(base.data, exponent.data):
                    try:
                        if b == 0 and e < 0:
                            result_data.append(float('inf'))
                        elif b < 0 and not isinstance(e, int) and e != int(e):
                            result_data.append(abs(b) ** e)
                        else:
                            result_data.append(b ** e)
                    except (OverflowError, ZeroDivisionError, ValueError):
                        if b == 0:
                            result_data.append(0.0 if e > 0 else float('inf'))
                        elif abs(b) > 1 and e > 100:
                            result_data.append(float('inf') if b > 0 else float('-inf'))
                        elif abs(b) < 1 and e < -100:
                            result_data.append(float('inf'))
                        else:
                            result_data.append(1.0)
            
            result = Array(result_data, dtype=base.dtype)
            return result
        except Exception as e:
            print(f"Array power计算失败，使用安全默认值: {e}")
            if isinstance(base.data[0], list):
                result_data = [[1.0 for _ in row] for row in base.data]
            else:
                result_data = [1.0 for _ in base.data]
            return Array(result_data, dtype=base.dtype)
    
    else:
        # 转换为Array
        exponent = Array(exponent)
        return power(base, exponent)


def nan_to_num(x: Union[Array, float, List[float]], nan: float = 0.0, posinf: float = None, neginf: float = None) -> Array:
    """
    将NaN和无穷大替换为有限数值
    
    Args:
        x: 输入数组或数值
        nan: 替换NaN的值，默认为0.0
        posinf: 替换正无穷的值，默认为很大的正数
        neginf: 替换负无穷的值，默认为很大的负数
        
    Returns:
        替换后的数组
    """
    if posinf is None:
        posinf = 1e38
    if neginf is None:
        neginf = -1e38
        
    if isinstance(x, (int, float)):
        if math.isnan(x):
            return array([nan])
        elif math.isinf(x):
            if x > 0:
                return array([posinf])
            else:
                return array([neginf])
        else:
            return array([x])
    
    if hasattr(x, 'shape') and hasattr(x, 'dtype'):
        try:
            if hasattr(x, 'tolist'):
                x_data = x.tolist()
                result_data = strong_nan.perfect_nan_to_num(x_data, nan=nan, posinf=posinf, neginf=neginf)
                return array(result_data)
            else:
                x_data = x.data if hasattr(x, 'data') else x
                result_data = strong_nan.perfect_nan_to_num(x_data, nan=nan, posinf=posinf, neginf=neginf)
                return array(result_data)
        except Exception as e:
            print(f"Warning: strong_nan failed, using fallback nan_to_num: {e}")
            pass
    
    if isinstance(x, list):
        x = array(x)
    
    if isinstance(x, Array):
        # 使用递归方法处理任意维度的数组
        def nan_to_num_recursive(data):
            if isinstance(data, list):
                return [nan_to_num_recursive(item) for item in data]
            else:
                if math.isnan(data):
                    return nan
                elif math.isinf(data):
                    if data > 0:
                        return posinf
                    else:
                        return neginf
                else:
                    return data
        
        result_data = nan_to_num_recursive(x.data)
        return array(result_data)
    
    raise TypeError(f"unsupported operand type(s) for nan_to_num: '{type(x)}'")

def prod(x: Union[Array, List[float]], axis=None) -> Union[Array, float]:
    """
    计算数组元素的乘积
    
    Args:
        x: 输入数组
        axis: 计算乘积的轴，None表示全部元素
        
    Returns:
        乘积结果
    """
    if not isinstance(x, Array):
        x = Array(x)
    
    if axis is None:
        # 计算所有元素的乘积
        def prod_recursive(data):
            if isinstance(data[0], list):
                result = 1
                for row in data:
                    result *= prod_recursive(row)
                return result
            else:
                result = 1
                for val in data:
                    result *= val
                return result
        
        return prod_recursive(x.data)
    else:
        # 沿指定轴计算乘积
        # 简化实现：只支持2D数组
        if isinstance(x.data[0], list):
            if axis == 0:
                # 沿行计算
                result = []
                for col in range(len(x.data[0])):
                    prod_val = 1
                    for row in range(len(x.data)):
                        prod_val *= x.data[row][col]
                    result.append(prod_val)
                return Array(result)
            elif axis == 1:
                # 沿列计算
                result = []
                for row in x.data:
                    prod_val = 1
                    for val in row:
                        prod_val *= val
                    result.append(prod_val)
                return Array(result)
        else:
            # 1D数组
            result = 1
            for val in x.data:
                result *= val
            return result

def all(x: Union[Array, List[bool]]) -> bool:
    """
    检查是否所有元素都为True
    
    Args:
        x: 输入数组
        
    Returns:
        如果所有元素都为True则返回True
    """
    if not isinstance(x, Array):
        x = Array(x)
    
    def all_recursive(data):
        if isinstance(data[0], list):
            for row in data:
                if not all_recursive(row):
                    return False
            return True
        else:
            for val in data:
                if not val:
                    return False
            return True
    
    return all_recursive(x.data)

def round_array(x: Union[Array, float, List[float]], decimals: int = 0) -> Array:
    """
    四舍五入到指定小数位
    
    Args:
        x: 输入数组或数值
        decimals: 小数位数
        
    Returns:
        四舍五入后的数组
    """
    import builtins  # 使用内置的round函数
    
    if isinstance(x, (int, float)):
        return Array([builtins.round(x, decimals)])
    
    if not isinstance(x, Array):
        x = Array(x)
    
    def round_recursive(data):
        if isinstance(data[0], list):
            return [round_recursive(row) for row in data]
        else:
            return [builtins.round(val, decimals) for val in data]
    
    result_data = round_recursive(x.data)
    return Array(result_data, dtype=x.dtype)

def isclose(a: Union[Array, float, List[float]], b: Union[Array, float, List[float]], 
           rtol: float = 1e-05, atol: float = 1e-08) -> Array:
    """
    检查两个数组是否在容差范围内相等
    
    Args:
        a: 第一个数组
        b: 第二个数组
        rtol: 相对容差
        atol: 绝对容差
        
    Returns:
        布尔数组
    """
    if not isinstance(a, Array):
        a = Array(a)
    if not isinstance(b, Array):
        b = Array(b)
    
    def isclose_recursive(data_a, data_b):
        if isinstance(data_a[0], list):
            return [isclose_recursive(row_a, row_b) for row_a, row_b in zip(data_a, data_b)]
        else:
            result = []
            for val_a, val_b in zip(data_a, data_b):
                diff = __builtins__['abs'](val_a - val_b)
                tolerance = atol + rtol * __builtins__['abs'](val_b)
                result.append(diff <= tolerance)
            return result
    
    result_data = isclose_recursive(a.data, b.data)
    return Array(result_data)

def allclose(a: Union[Array, float, List[float]], b: Union[Array, float, List[float]], 
            rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    """
    检查两个数组是否在容差范围内全部相等
    
    Args:
        a: 第一个数组
        b: 第二个数组
        rtol: 相对容差
        atol: 绝对容差
        
    Returns:
        如果所有元素都在容差范围内相等则返回True，否则返回False
    """
    if not isinstance(a, Array):
        a = Array(a)
    if not isinstance(b, Array):
        b = Array(b)
    
    # 检查形状是否相同
    if a.shape != b.shape:
        return False
    
    def allclose_recursive(data_a, data_b):
        if isinstance(data_a, list):
            if len(data_a) != len(data_b):
                return False
            for item_a, item_b in zip(data_a, data_b):
                if not allclose_recursive(item_a, item_b):
                    return False
            return True
        else:
            # 标量比较
            diff = __builtins__['abs'](data_a - data_b)
            tolerance = atol + rtol * __builtins__['abs'](data_b)
            return diff <= tolerance
    
    return allclose_recursive(a.data, b.data)

def resize(array: Union[Array, List[float]], new_shape: Union[int, Tuple[int, ...]]) -> Array:
    """
    改变数组的形状，如果新形状更大则重复元素
    
    Args:
        array: 输入数组
        new_shape: 新的形状
        
    Returns:
        调整大小后的数组
    """
    if not isinstance(array, Array):
        array = Array(array)
    
    if isinstance(new_shape, int):
        new_shape = (new_shape,)
    
    # 计算新的总大小
    new_size = 1
    for dim in new_shape:
        new_size *= dim
    
    # 展平原数组
    flat_data = []
    def flatten_recursive(data):
        if isinstance(data, list):
            for item in data:
                flatten_recursive(item)
        else:
            flat_data.append(data)
    
    flatten_recursive(array.data)
    
    # 如果新大小更大，重复元素
    if new_size > len(flat_data):
        # 重复原数据直到达到新大小
        result_data = []
        for i in range(new_size):
            result_data.append(flat_data[i % len(flat_data)])
    else:
        # 如果新大小更小，截断数据
        result_data = flat_data[:new_size]
    
    # 创建新数组并重塑
    result = Array(result_data)
    if len(new_shape) > 1:
        result = result.reshape(*new_shape)
    
    return result

def full_like(array: Union[Array, List[float]], fill_value: float, dtype: Optional[type] = None) -> Array:
    """
    创建与输入数组形状相同的数组，并用指定值填充
    
    Args:
        array: 输入数组
        fill_value: 填充值
        dtype: 数据类型
        
    Returns:
        填充后的数组
    """
    if not isinstance(array, Array):
        array = Array(array)
    
    # 计算总大小
    total_size = 1
    for dim in array.shape:
        total_size *= dim
    
    # 创建填充数据
    data = [fill_value] * total_size
    result = Array(data, dtype=dtype or array.dtype)
    
    # 重塑为原数组的形状
    if len(array.shape) > 1:
        result = result.reshape(*array.shape)
    
    return result

def empty_like(array: Union[Array, List[float]], dtype: Optional[type] = None) -> Array:
    """
    创建与输入数组形状相同的未初始化数组
    
    Args:
        array: 输入数组
        dtype: 数据类型
        
    Returns:
        未初始化的数组（实际上用0填充）
    """
    return zeros_like(array, dtype=dtype)

def argsort(array: Union[Array, List[float]], axis: int = -1) -> Array:
    """
    返回排序后的索引数组
    
    Args:
        array: 输入数组
        axis: 排序的轴，默认为-1（最后一个轴）
        
    Returns:
        排序后的索引数组
    """
    if not isinstance(array, Array):
        array = Array(array)
    
    if axis == -1:
        axis = len(array.shape) - 1
    
    if len(array.shape) == 1:
        # 一维数组的排序
        indexed_data = [(i, val) for i, val in enumerate(array.data)]
        indexed_data.sort(key=lambda x: x[1])
        indices = [x[0] for x in indexed_data]
        return Array(indices)
    else:
        # 多维数组的排序（简化实现）
        # 这里只实现基本的功能
        if isinstance(array.data[0], list):
            # 2D数组
            if axis == 0:
                # 沿第一个轴排序
                result = []
                for col in range(len(array.data[0])):
                    column_data = [(i, array.data[i][col]) for i in range(len(array.data))]
                    column_data.sort(key=lambda x: x[1])
                    indices = [x[0] for x in column_data]
                    result.append(indices)
                # 转置结果
                transposed = [[result[j][i] for j in range(len(result))] for i in range(len(result[0]))]
                return Array(transposed)
            else:
                # 沿第二个轴排序
                result = []
                for row in array.data:
                    indexed_row = [(i, val) for i, val in enumerate(row)]
                    indexed_row.sort(key=lambda x: x[1])
                    indices = [x[0] for x in indexed_row]
                    result.append(indices)
                return Array(result)
        else:
            # 一维数组
            indexed_data = [(i, val) for i, val in enumerate(array.data)]
            indexed_data.sort(key=lambda x: x[1])
            indices = [x[0] for x in indexed_data]
            return Array(indices)

def clip(x: Union[Array, float, List[float]], min_val: float, max_val: float) -> Array:
    """
    将数组中的值裁剪到指定范围内
    
    Args:
        x: 输入数组或数值
        min_val: 最小值
        max_val: 最大值
        
    Returns:
        裁剪后的数组
    """
    if isinstance(x, (int, float)):
        clipped_val = __builtins__['max'](min_val, __builtins__['min'](x, max_val))
        return Array([clipped_val])
    
    if not isinstance(x, Array):
        x = Array(x)
    
    return x.clip(min_val, max_val)

def logical_and(a: Union[Array, float, List[float]], b: Union[Array, float, List[float]]) -> Array:
    """
    逐元素计算逻辑AND运算
    
    Args:
        a: 第一个数组或数值
        b: 第二个数组或数值
        
    Returns:
        逻辑AND运算结果的布尔数组
    """
    if not isinstance(a, Array):
        a = Array(a)
    if not isinstance(b, Array):
        b = Array(b)
    
    # 检查形状是否匹配
    if a.shape != b.shape:
        raise ValueError(f"shapes do not match for logical_and: {a.shape} vs {b.shape}")
    
    # 执行逻辑AND运算
    if isinstance(a.data[0], list) and isinstance(b.data[0], list):
        # 两个都是多维数组
        result_data = []
        for row_a, row_b in zip(a.data, b.data):
            result_row = [bool(val_a) and bool(val_b) for val_a, val_b in zip(row_a, row_b)]
            result_data.append(result_row)
    elif not isinstance(a.data[0], list) and not isinstance(b.data[0], list):
        # 两个都是一维数组
        result_data = [bool(val_a) and bool(val_b) for val_a, val_b in zip(a.data, b.data)]
    else:
        raise ValueError("Inconsistent array dimensions for logical_and")
    
    return Array(result_data)

def split(array: Union[Array, List, Tuple], indices_or_sections: Union[int, List[int]], axis: int = 0) -> List[Array]:
    """
    将数组沿指定轴分割成多个子数组
    
    Args:
        array: 输入数组
        indices_or_sections: 分割点或分割数量
            - 如果是整数，表示等分成多少份
            - 如果是列表，表示分割点的索引
        axis: 分割的轴
        
    Returns:
        分割后的数组列表
    """
    if not isinstance(array, Array):
        array = Array(array)
    
    # 处理负数轴
    if axis < 0:
        axis = len(array.shape) + axis
    
    # 验证轴的有效性
    if axis >= len(array.shape):
        raise ValueError(f"axis {axis} is out of bounds for array of dimension {len(array.shape)}")
    
    axis_size = array.shape[axis]
    
    if isinstance(indices_or_sections, int):
        # 等分情况
        sections = indices_or_sections
        if axis_size % sections != 0:
            raise ValueError(f"array split does not result in an equal division")
        
        section_size = axis_size // sections
        split_indices = [i * section_size for i in range(1, sections)]
    else:
        # 指定分割点
        split_indices = list(indices_or_sections)
    
    # 添加起始和结束点
    all_indices = [0] + split_indices + [axis_size]
    
    result = []
    
    if len(array.shape) == 1:
        # 一维数组分割
        for i in range(len(all_indices) - 1):
            start = all_indices[i]
            end = all_indices[i + 1]
            result.append(Array(array.data[start:end]))
    
    elif len(array.shape) == 2:
        # 二维数组分割
        if axis == 0:
            # 沿第一个轴（行）分割
            for i in range(len(all_indices) - 1):
                start = all_indices[i]
                end = all_indices[i + 1]
                result.append(Array(array.data[start:end]))
        else:
            # 沿第二个轴（列）分割
            for i in range(len(all_indices) - 1):
                start = all_indices[i]
                end = all_indices[i + 1]
                split_data = []
                for row in array.data:
                    split_data.append(row[start:end])
                result.append(Array(split_data))
    
    elif len(array.shape) == 3:
        # 三维数组分割
        if axis == 0:
            # 沿第一个轴分割
            for i in range(len(all_indices) - 1):
                start = all_indices[i]
                end = all_indices[i + 1]
                result.append(Array(array.data[start:end]))
        elif axis == 1:
            # 沿第二个轴分割
            for i in range(len(all_indices) - 1):
                start = all_indices[i]
                end = all_indices[i + 1]
                split_data = []
                for layer in array.data:
                    split_data.append(layer[start:end])
                result.append(Array(split_data))
        else:
            # 沿第三个轴分割
            for i in range(len(all_indices) - 1):
                start = all_indices[i]
                end = all_indices[i + 1]
                split_data = []
                for layer in array.data:
                    layer_data = []
                    for row in layer:
                        layer_data.append(row[start:end])
                    split_data.append(layer_data)
                result.append(Array(split_data))
    
    else:
        # 更高维度的数组（简化实现）
        raise NotImplementedError("split for arrays with more than 3 dimensions not implemented")
    
    return result

def abs(x: Union[Array, float, List[float]]) -> Array:
    """
    计算数组中每个元素的绝对值
    
    Args:
        x: 输入数组或数值
        
    Returns:
        绝对值数组
    """
    if isinstance(x, (int, float)):
        return Array([__builtins__['abs'](x)])
    
    if isinstance(x, Array):
        return x.abs()
    
    x = Array(x)
    return x.abs()

def sign(x: Union[Array, float, List[float]]) -> Array:
    """
    计算数组中每个元素的符号
    
    Args:
        x: 输入数组或数值
        
    Returns:
        符号数组，其中：
        - 正数返回 1.0
        - 负数返回 -1.0  
        - 零返回 0.0
    """
    if isinstance(x, (int, float)):
        if x > 0:
            return Array([1.0])
        elif x < 0:
            return Array([-1.0])
        else:
            return Array([0.0])
    
    if not isinstance(x, Array):
        x = Array(x)
    
    # 递归处理多维数组
    def sign_recursive(data):
        if isinstance(data, list):
            return [sign_recursive(item) for item in data]
        else:
            if data > 0:
                return 1.0
            elif data < 0:
                return -1.0
            else:
                return 0.0
    
    result_data = sign_recursive(x.data)
    return Array(result_data, dtype=x.dtype)


def reshape(array: Union[Array, List, Tuple], shape: Union[int, Tuple[int, ...]]) -> Array:
    """
    重塑数组形状（模块级函数），增强兼容性
    
    Args:
        array: 输入数组
        shape: 新的形状，可以是整数或整数元组
        
    Returns:
        重塑后的数组
    """
    if not isinstance(array, Array):
        array = Array(array)
    
    try:
        # 尝试使用Array的reshape方法
        return array.reshape(*shape if isinstance(shape, tuple) else (shape,))
    except (ValueError, TypeError) as e:
        # 如果失败，使用数组展平和重新构造作为后备
        try:
            # 获取数据并展平
            if hasattr(array, 'data'):
                flat_data = array.flatten().data if hasattr(array, 'flatten') else array.data
            else:
                flat_data = array
            
            # 确保是列表
            if not isinstance(flat_data, list):
                flat_data = list(flat_data)
            
            # 计算新形状的总元素数
            if isinstance(shape, tuple):
                total_elements = 1
                for dim in shape:
                    total_elements *= dim
                new_shape = shape
            else:
                total_elements = shape
                new_shape = (shape,)
            
            # 检查元素数量是否匹配
            if len(flat_data) != total_elements:
                raise ValueError(f"无法将大小为 {len(flat_data)} 的数组重塑为形状 {new_shape}")
            
            # 创建新的Array并手动设置形状
            result = Array(flat_data)
            result.shape = new_shape
            return result
        except Exception as fallback_error:
            raise e

def transpose(array: Union[Array, List, Tuple], axes: Optional[Tuple[int, ...]] = None) -> Array:
    """
    转置数组（模块级函数）
    
    Args:
        array: 输入数组
        axes: 转置的轴顺序，如果为None则完全转置
        
    Returns:
        转置后的数组
    """
    if not isinstance(array, Array):
        array = Array(array)
    
    # 如果没有指定axes，执行完全转置
    if axes is None:
        if len(array.shape) == 1:
            return array  # 1D数组转置后还是自己
        elif len(array.shape) == 2:
            return array.transpose()  # 使用现有的2D转置方法
        else:
            # 多维数组的完全转置：反转所有轴
            axes = tuple(range(len(array.shape) - 1, -1, -1))
    
    # 验证axes参数
    if len(axes) != len(array.shape):
        raise ValueError(f"axes长度({len(axes)})必须等于数组维度({len(array.shape)})")
    
    if sorted(axes) != list(range(len(array.shape))):
        raise ValueError("axes必须是所有维度的排列")
    
    # 执行转置
    return _transpose_with_axes(array, axes)

def _transpose_with_axes(array: Array, axes: Tuple[int, ...]) -> Array:
    """使用指定轴顺序进行转置的辅助函数"""
    
    # 计算新的形状
    new_shape = tuple(array.shape[axis] for axis in axes)
    
    # 创建结果数组
    result_data = _create_nested_list(new_shape)
    
    # 执行转置操作
    _fill_transposed_data(array.data, result_data, array.shape, new_shape, axes)
    
    return Array(result_data)

def _create_nested_list(shape: Tuple[int, ...]):
    """创建指定形状的嵌套列表结构"""
    if len(shape) == 1:
        return [0.0] * shape[0]
    elif len(shape) == 2:
        return [[0.0 for _ in range(shape[1])] for _ in range(shape[0])]
    elif len(shape) == 3:
        return [[[0.0 for _ in range(shape[2])] for _ in range(shape[1])] for _ in range(shape[0])]
    elif len(shape) == 4:
        return [[[[0.0 for _ in range(shape[3])] for _ in range(shape[2])] 
                 for _ in range(shape[1])] for _ in range(shape[0])]
    else:
        # 对于更高维度，使用递归方法
        return [_create_nested_list(shape[1:]) for _ in range(shape[0])]

def _fill_transposed_data(source_data, target_data, source_shape: Tuple[int, ...], 
                         target_shape: Tuple[int, ...], axes: Tuple[int, ...]):
    """填充转置后的数据"""
    
    # 对于2D数组的特殊处理
    if len(source_shape) == 2 and len(target_shape) == 2:
        rows, cols = source_shape
        for i in range(rows):
            for j in range(cols):
                if axes == (1, 0):  # 标准转置
                    target_data[j][i] = source_data[i][j]
                else:  # axes == (0, 1)，不转置
                    target_data[i][j] = source_data[i][j]
        return
    
    # 对于3D数组的处理
    if len(source_shape) == 3 and len(target_shape) == 3:
        d0, d1, d2 = source_shape
        for i in range(d0):
            for j in range(d1):
                for k in range(d2):
                    # 根据axes重新映射索引
                    new_indices = [0, 0, 0]
                    old_indices = [i, j, k]
                    for new_pos, old_pos in enumerate(axes):
                        new_indices[new_pos] = old_indices[old_pos]
                    
                    target_data[new_indices[0]][new_indices[1]][new_indices[2]] = source_data[i][j][k]
        return
    
    # 对于4D数组的处理
    if len(source_shape) == 4 and len(target_shape) == 4:
        d0, d1, d2, d3 = source_shape
        for i in range(d0):
            for j in range(d1):
                for k in range(d2):
                    for l in range(d3):
                        # 根据axes重新映射索引
                        new_indices = [0, 0, 0, 0]
                        old_indices = [i, j, k, l]
                        for new_pos, old_pos in enumerate(axes):
                            new_indices[new_pos] = old_indices[old_pos]
                        
                        target_data[new_indices[0]][new_indices[1]][new_indices[2]][new_indices[3]] = source_data[i][j][k][l]
        return
    
    # 对于更高维度，使用通用递归方法
    if len(source_shape) > 4:
        # 使用通用递归方法处理任意维度
        _fill_transposed_data_recursive(source_data, target_data, source_shape, axes, [])
        return
    
    # 如果到这里，说明有未处理的情况
    raise NotImplementedError(f"暂不支持{len(source_shape)}维数组的转置")

def _fill_transposed_data_recursive(source_data, target_data, source_shape: Tuple[int, ...], 
                                   axes: Tuple[int, ...], current_indices: List[int]):
    """递归填充转置数据的辅助函数"""
    
    if len(current_indices) == len(source_shape):
        # 到达叶子节点，执行数据复制
        # 根据axes重新映射索引
        new_indices = [0] * len(source_shape)
        for new_pos, old_pos in enumerate(axes):
            new_indices[new_pos] = current_indices[old_pos]
        
        # 获取源数据
        src_value = source_data
        for idx in current_indices:
            src_value = src_value[idx]
        
        # 设置目标数据
        target_ref = target_data
        for i, idx in enumerate(new_indices[:-1]):
            target_ref = target_ref[idx]
        target_ref[new_indices[-1]] = src_value
        return
    
    # 递归处理下一个维度
    current_dim = len(current_indices)
    for i in range(source_shape[current_dim]):
        _fill_transposed_data_recursive(source_data, target_data, source_shape, axes, 
                                       current_indices + [i])

def pad(array: Union[Array, List, Tuple], pad_width, mode: str = 'constant', constant_values: float = 0.0) -> Array:
    """
    填充数组
    
    Args:
        array: 输入数组
        pad_width: 填充宽度，格式为 ((before_1, after_1), (before_2, after_2), ...)
        mode: 填充模式，目前支持 'constant'
        constant_values: 常数填充值
        
    Returns:
        填充后的数组
    """
    if not isinstance(array, Array):
        array = Array(array)
    
    # 验证pad_width格式
    if not isinstance(pad_width, (list, tuple)):
        raise ValueError("pad_width必须是列表或元组")
    
    # 确保pad_width与数组维度匹配
    if len(pad_width) != len(array.shape):
        raise ValueError(f"pad_width长度({len(pad_width)})必须等于数组维度({len(array.shape)})")
    
    # 目前只支持constant模式
    if mode != 'constant':
        raise NotImplementedError(f"暂不支持填充模式: {mode}")
    
    # 计算新的形状
    new_shape = []
    for i, (before, after) in enumerate(pad_width):
        new_shape.append(array.shape[i] + before + after)
    new_shape = tuple(new_shape)
    
    # 创建填充后的数组
    if len(array.shape) == 1:
        return _pad_1d(array, pad_width, constant_values)
    elif len(array.shape) == 2:
        return _pad_2d(array, pad_width, constant_values)
    elif len(array.shape) == 3:
        return _pad_3d(array, pad_width, constant_values)
    elif len(array.shape) == 4:
        return _pad_4d(array, pad_width, constant_values)
    else:
        raise NotImplementedError(f"暂不支持{len(array.shape)}维数组的填充")

def _pad_1d(array: Array, pad_width, constant_values: float) -> Array:
    """1D数组填充"""
    before, after = pad_width[0]
    
    # 创建新数组
    new_data = [constant_values] * before + array.data + [constant_values] * after
    return Array(new_data)

def _pad_2d(array: Array, pad_width, constant_values: float) -> Array:
    """2D数组填充"""
    (before_0, after_0), (before_1, after_1) = pad_width
    rows, cols = array.shape
    
    # 创建新数组
    new_rows = rows + before_0 + after_0
    new_cols = cols + before_1 + after_1
    
    new_data = []
    
    # 前面的填充行
    for _ in range(before_0):
        new_data.append([constant_values] * new_cols)
    
    # 原始数据行（左右填充）
    for i in range(rows):
        row = ([constant_values] * before_1 + 
               array.data[i] + 
               [constant_values] * after_1)
        new_data.append(row)
    
    # 后面的填充行
    for _ in range(after_0):
        new_data.append([constant_values] * new_cols)
    
    return Array(new_data)

def _pad_3d(array: Array, pad_width, constant_values: float) -> Array:
    """3D数组填充"""
    (before_0, after_0), (before_1, after_1), (before_2, after_2) = pad_width
    d0, d1, d2 = array.shape
    
    # 创建新数组
    new_d0 = d0 + before_0 + after_0
    new_d1 = d1 + before_1 + after_1
    new_d2 = d2 + before_2 + after_2
    
    new_data = []
    
    # 前面的填充层
    for _ in range(before_0):
        layer = []
        for _ in range(new_d1):
            layer.append([constant_values] * new_d2)
        new_data.append(layer)
    
    # 原始数据层（上下左右填充）
    for i in range(d0):
        layer = []
        
        # 前面的填充行
        for _ in range(before_1):
            layer.append([constant_values] * new_d2)
        
        # 原始数据行（左右填充）
        for j in range(d1):
            row = ([constant_values] * before_2 + 
                   array.data[i][j] + 
                   [constant_values] * after_2)
            layer.append(row)
        
        # 后面的填充行
        for _ in range(after_1):
            layer.append([constant_values] * new_d2)
        
        new_data.append(layer)
    
    # 后面的填充层
    for _ in range(after_0):
        layer = []
        for _ in range(new_d1):
            layer.append([constant_values] * new_d2)
        new_data.append(layer)
    
    return Array(new_data)

def _pad_4d(array: Array, pad_width, constant_values: float) -> Array:
    """4D数组填充"""
    (before_0, after_0), (before_1, after_1), (before_2, after_2), (before_3, after_3) = pad_width
    d0, d1, d2, d3 = array.shape
    
    # 创建新数组
    new_d0 = d0 + before_0 + after_0
    new_d1 = d1 + before_1 + after_1
    new_d2 = d2 + before_2 + after_2
    new_d3 = d3 + before_3 + after_3
    
    new_data = []
    
    # 前面的填充批次
    for _ in range(before_0):
        batch = []
        for _ in range(new_d1):
            layer = []
            for _ in range(new_d2):
                layer.append([constant_values] * new_d3)
            batch.append(layer)
        new_data.append(batch)
    
    # 原始数据批次（各维度填充）
    for i in range(d0):
        batch = []
        
        # 前面的填充层
        for _ in range(before_1):
            layer = []
            for _ in range(new_d2):
                layer.append([constant_values] * new_d3)
            batch.append(layer)
        
        # 原始数据层（上下左右填充）
        for j in range(d1):
            layer = []
            
            # 前面的填充行
            for _ in range(before_2):
                layer.append([constant_values] * new_d3)
            
            # 原始数据行（左右填充）
            for k in range(d2):
                row = ([constant_values] * before_3 + 
                       array.data[i][j][k] + 
                       [constant_values] * after_3)
                layer.append(row)
            
            # 后面的填充行
            for _ in range(after_2):
                layer.append([constant_values] * new_d3)
            
            batch.append(layer)
        
        # 后面的填充层
        for _ in range(after_1):
            layer = []
            for _ in range(new_d2):
                layer.append([constant_values] * new_d3)
            batch.append(layer)
        
        new_data.append(batch)
    
    # 后面的填充批次
    for _ in range(after_0):
        batch = []
        for _ in range(new_d1):
            layer = []
            for _ in range(new_d2):
                layer.append([constant_values] * new_d3)
            batch.append(layer)
        new_data.append(batch)
    
    return Array(new_data)

def log1p(array: Union[Array, List, Tuple]) -> Array:
    """
    计算log(1 + x)，数值稳定版本
    
    Args:
        array: 输入数组
        
    Returns:
        log(1 + x)的结果数组
    """
    if not isinstance(array, Array):
        array = Array(array)
    
    # 递归处理嵌套结构
    result_data = _log1p_recursive(array.data)
    return Array(result_data)

def _log1p_recursive(data):
    """递归处理log1p操作"""
    if isinstance(data, list):
        return [_log1p_recursive(item) for item in data]
    elif isinstance(data, Array):
        # 如果是Array对象，递归处理其data
        return _log1p_recursive(data.data)
    else:
        # 处理标量值
        try:
            return _log1p_scalar(float(data))
        except (TypeError, ValueError):
            # 如果转换失败，尝试其他方法
            if hasattr(data, 'data'):
                return _log1p_recursive(data.data)
            else:
                return _log1p_scalar(0.0)  # 默认值

def _log1p_scalar(x) -> float:
    """
    计算单个标量的log(1 + x)，数值稳定版本
    
    Args:
        x: 输入值
        
    Returns:
        log(1 + x)的结果
    """
    import math
    
    # 确保x是数值类型
    if isinstance(x, Array):
        # 如果是Array对象，递归处理
        return _log1p_recursive(x.data)
    
    try:
        x = float(x)
    except (TypeError, ValueError):
        if hasattr(x, 'data'):
            return _log1p_recursive(x.data)
        else:
            return 0.0
    
    # 数值稳定性处理
    if x < -1.0:
        # x < -1时，1+x <= 0，log未定义，返回负无穷
        return float('-inf')
    elif x == -1.0:
        # x = -1时，1+x = 0，log(0) = -inf
        return float('-inf')
    elif __builtins__['abs'](x) < 1e-8:  # 使用Python内置abs函数
        # 当x很小时，使用泰勒展开：log(1+x) ≈ x - x²/2 + x³/3 - ...
        # 对于很小的x，主要项是x
        return x - x*x/2.0 + x*x*x/3.0
    elif x < 0.5:
        # 对于中等大小的x，使用标准公式但保持精度
        return math.log(1.0 + x)
    else:
        # 对于较大的x，直接使用log(1+x)
        return math.log(1.0 + x)

def int32(x: Union[Array, float, int, List]) -> Array:
    """
    将输入转换为32位整数数组
    
    Args:
        x: 输入数据
        
    Returns:
        转换为32位整数的数组
    """
    if not isinstance(x, Array):
        x = Array(x)
    
    # 递归处理嵌套结构
    result_data = _int_convert_recursive(x.data, 'int32')
    result = Array(result_data)
    result.dtype = int
    return result

def int64(x: Union[Array, float, int, List]) -> Array:
    """
    将输入转换为64位整数数组
    
    Args:
        x: 输入数据
        
    Returns:
        转换为64位整数的数组
    """
    if not isinstance(x, Array):
        x = Array(x)
    
    # 递归处理嵌套结构
    result_data = _int_convert_recursive(x.data, 'int64')
    result = Array(result_data)
    result.dtype = int
    return result

def _int_convert_recursive(data, dtype_name):
    """递归处理整数转换操作"""
    if isinstance(data, list):
        return [_int_convert_recursive(item, dtype_name) for item in data]
    elif isinstance(data, Array):
        # 如果是Array对象，递归处理其data
        return _int_convert_recursive(data.data, dtype_name)
    else:
        # 处理标量值
        try:
            # 转换为整数
            return int(float(data))
        except (TypeError, ValueError):
            # 如果转换失败，尝试其他方法
            if hasattr(data, 'data'):
                return _int_convert_recursive(data.data, dtype_name)
            else:
                return 0  # 默认值

def vstack(arrays_list: List[Array]) -> Array:
    """
    垂直堆叠数组（沿第0轴）
    
    Args:
        arrays_list: 要堆叠的数组列表
        
    Returns:
        垂直堆叠后的数组
    """
    if not arrays_list:
        raise ValueError("数组列表不能为空")
    
    # 确保所有输入都是Array对象
    arrays_list = [Array(arr) if not isinstance(arr, Array) else arr for arr in arrays_list]
    
    # 检查所有数组的形状（除了第0维）是否兼容
    first_shape = arrays_list[0].shape
    if len(first_shape) == 1:
        # 1D数组的情况
        for arr in arrays_list[1:]:
            if len(arr.shape) != 1:
                raise ValueError("所有数组必须具有相同的维数")
    else:
        # 多维数组的情况
        for arr in arrays_list[1:]:
            if len(arr.shape) != len(first_shape):
                raise ValueError("所有数组必须具有相同的维数")
            if arr.shape[1:] != first_shape[1:]:
                raise ValueError("除第0维外，所有数组的形状必须相同")
    
    # 执行垂直堆叠
    if len(first_shape) == 1:
        # 1D数组：简单连接
        result_data = []
        for arr in arrays_list:
            result_data.extend(arr.data)
        return Array(result_data)
    else:
        # 多维数组：沿第0轴堆叠
        result_data = []
        for arr in arrays_list:
            result_data.extend(arr.data)
        return Array(result_data)

def ndindex(*shape: int):
    """
    生成多维数组索引的迭代器
    
    Args:
        *shape: 数组的形状，可以是多个整数参数
        
    Yields:
        tuple: 每个可能的索引组合
        
    Examples:
        for idx in ndindex(2, 3):
            print(idx)  # (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)
    """
    if len(shape) == 0:
        yield ()
        return
    
    if len(shape) == 1:
        # 一维情况
        for i in range(shape[0]):
            yield (i,)
        return
    
    # 多维情况：递归生成
    def _generate_indices(dims):
        if len(dims) == 1:
            for i in range(dims[0]):
                yield [i]
        else:
            for i in range(dims[0]):
                for rest in _generate_indices(dims[1:]):
                    yield [i] + rest
    
    for indices in _generate_indices(shape):
        yield tuple(indices)

def asarray_numpy_compatible(data, dtype=None):
    import final_array
    import memasarray
    import inspect
    import os
    from datetime import datetime
    
    # 全局调用计数器初始化（确保只初始化一次）
    if not hasattr(asarray_numpy_compatible, 'call_count'):
        asarray_numpy_compatible.call_count = 0
        asarray_numpy_compatible.array_count = 0
        asarray_numpy_compatible.memoryview_count = 0
        asarray_numpy_compatible.list_count = 0
        # 初始化日志文件
        log_file = "final_array_log.txt"
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== Final Array 调用日志 ===\n")
            f.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"=" * 60 + "\n\n")
    
    # 定义日志输出函数（统一使用）
    def log_message(message):
        print(message)  # 输出到终端
        with open("final_array_log.txt", 'a', encoding='utf-8') as f:
            f.write(message + "\n")  # 写入文件
    
    # 获取调用栈信息
    frame = inspect.currentframe()
    try:
        caller_frame = frame.f_back
        
        # 获取调用栈（前3层）
        call_stack = []
        current_frame = caller_frame
        for i in range(3):
            if current_frame:
                filename = os.path.basename(current_frame.f_code.co_filename)
                function_name = current_frame.f_code.co_name
                line_number = current_frame.f_lineno
                call_stack.append(f"{filename}:{line_number} in {function_name}()")
                current_frame = current_frame.f_back
            else:
                break
    finally:
        del frame
    

    
    # 处理特殊对象（如memoryview）
    if hasattr(data, '__array__'):
        asarray_numpy_compatible.call_count += 1
        asarray_numpy_compatible.array_count += 1
        call_id = asarray_numpy_compatible.call_count
        
        log_message(f"[CALL {call_id}] 调用__array__")
        for stack_info in [f"  {info}" for info in call_stack]:
            log_message(stack_info)
        
        great_data = memasarray.ult_asarray(data, dtype=dtype)
        result = Array([0])  # 临时初始化
        result.data = great_data
        result.shape = great_data.shape
        result.dtype = dtype or great_data.dtype
        print ("成功——array——")

        log_message("")  # 空行分隔
        return result
    
    # 处理memoryview对象
    if isinstance(data, memoryview):
        asarray_numpy_compatible.call_count += 1
        asarray_numpy_compatible.memoryview_count += 1
        call_id = asarray_numpy_compatible.call_count
        
        log_message(f"[CALL {call_id}] 调用memoryview")
        for stack_info in [f"  {info}" for info in call_stack]:
            log_message(stack_info)
        
        great_data = memasarray.ult_asarray(data, dtype=dtype)
        result = Array([0])
        result.data = great_data
        result.shape = great_data.shape
        result.dtype = dtype or great_data.dtype
        print ("成功memoryview")
        
        log_message("")  # 空行分隔
        return result

    
    standard_result = asarray(data, dtype=dtype)
    
    if isinstance(standard_result.data, list):
        asarray_numpy_compatible.call_count += 1
        asarray_numpy_compatible.list_count += 1
        call_id = asarray_numpy_compatible.call_count
        
        log_message(f"[CALL {call_id}] 调用list")
        for stack_info in [f"  {info}" for info in call_stack]:
            log_message(stack_info)
        
        # 针对第70次和第71次调用输出详细数据调试信息
        if call_id == 135 or call_id == 1300 or call_id == 1270 or call_id == 1250:
            log_message(f"🔍 [CALL {call_id}] 数据类型调试:")
            log_message(f"  输入数据类型: {type(data)}")
            
            # 检查数据长度
            try:
                data_length = len(standard_result.data) if hasattr(standard_result.data, '__len__') else "无法获取"
                log_message(f"  输入数据长度: {data_length}")
            except:
                log_message(f"  输入数据长度: 无法获取")
            
            # 获取前10个数据项
            try:
                if isinstance(standard_result.data, (list, tuple)) and len(standard_result.data) > 0:
                    first_10_items = standard_result.data[:10] if len(standard_result.data) >= 10 else standard_result.data
                    log_message(f"  输入数据前10个: {first_10_items}")
                else:
                    log_message(f"  输入数据前10个: {standard_result.data} (标量或空)")
            except Exception as e:
                log_message(f"  输入数据前10个: 获取失败: {e}")
        
        # 使用final_array处理数据
        try:
            #版本1：自定义final_array
            great_data = final_array.perfect_array(standard_result.data, dtype=dtype or float)
            #版本2：标准numpy
            import numpy
            #great_data = numpy.array(standard_result.data, dtype=dtype or float)
            result = Array([0])  # 临时初始化
            result.data = great_data
            result.shape = great_data.shape
            result.dtype = dtype or great_data.dtype
            
            # 针对第70次和第71次调用输出结果调试信息
            if call_id == 135 or call_id == 1300 or call_id == 1270 or call_id == 1250:
                log_message(f"  great_data类型: {type(great_data)}")
                log_message(f"  result.data类型: {type(result.data)}")
            
            log_message("")  # 空行分隔
            
            return result
            
        except Exception as e:
            log_message(f"[CALL {call_id}] ❌ final_array.perfect_array失败: {e}")
            log_message(f"  回退到标准结果")
            log_message("")  # 空行分隔
            return standard_result
    
    print(f"✅ 直接使用standard_result")
    return standard_result

