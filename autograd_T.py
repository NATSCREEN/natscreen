# local_torch/_tensor/autograd.py

import weakref
import threading
from functools import wraps
from enum import Enum, auto
import arrays
import strong_matmul

# === Global debug flag ===
DEBUG_AUTOGRAD = False  # Set to True to enable debug output

# === 核心类型定义 ===
class GradMode(Enum):
    """梯度模式
    
    定义了两种梯度模式：
    - TRAINING: 训练模式，启用梯度计算
    - INFERENCE: 推理模式，禁用梯度计算
    """
    TRAINING = auto()
    INFERENCE = auto()

class Context:
    __slots__ = ['saved_tensors', 'non_differentiable', 'metadata', 'module_ref', 'module_ref_a', 'module_ref_b', 'module_refs']
    
    def __init__(self):
        self.saved_tensors = None  # 初始化为 None 而不是空元组
        self.non_differentiable = set()
        self.metadata = {}
        self.module_ref = None
        self.module_ref_a = None
        self.module_ref_b = None
        self.module_refs = {}  # 存储每个输入参数的模块引用
    
    def save_for_backward(self, *tensors):  # 移除Tensor类型注解
        """保存张量用于反向传播
        
        Args:
            *tensors: 要保存的张量
        """
        self.saved_tensors = tensors
    
    def mark_non_differentiable(self, *tensors):  # 移除Tensor类型注解
        """标记张量为不可微
        
        Args:
            *tensors: 要标记的张量
        """
        self.non_differentiable.update(id(t) for t in tensors)

class FunctionMeta(type):
    """函数元类
    
    用于处理函数的元数据，包括注册和包装方法。
    """
    _registry = {}
    
    def __new__(mcls, name, bases, attrs):
        """创建新的函数类
        
        Args:
            name: 类名
            bases: 基类
            attrs: 类属性
        
        Returns:
            新的函数类
        """
        for method in ['forward', 'backward']:
            if method in attrs:
                attrs[method] = FunctionMeta._wrap_method(attrs[method])
        cls = super().__new__(mcls, name, bases, attrs)
        if name != 'Function':
            mcls._registry[name] = cls
        return cls
    
    @staticmethod
    def _wrap_method(method):
        """包装方法
        
        Args:
            method: 要包装的方法
        
        Returns:
            包装后的方法
        """
        @wraps(method)
        def wrapper(ctx, *args, **kwargs):
            return method(ctx, *args, **kwargs)
        return staticmethod(wrapper)

class Function(metaclass=FunctionMeta):
    """函数基类
    
    所有支持自动微分的函数都应该继承这个类。
    """
    __slots__ = ['requires_grad', 'ctx']
    
    @staticmethod
    def forward(ctx, *args):
        """前向传播
        
        Args:
            ctx: 上下文对象
            *args: 输入参数
        
        Returns:
            前向传播的结果
        """
        raise NotImplementedError("必须实现forward方法")
    
    @staticmethod
    def backward(ctx, *grad_outputs):
        """反向传播
        
        Args:
            ctx: 上下文对象
            *grad_outputs: 上游梯度
        
        Returns:
            每个输入参数的梯度
        """
        raise NotImplementedError("必须实现backward方法")
    
    @classmethod
    def apply(cls, *args, **kwargs):
        """应用函数
        
        Args:
            *args: 输入参数
            **kwargs: 关键字参数
        
        Returns:
            函数的结果
        """
        # 延迟导入+运行时类型检查
        
        def is_tensor(obj):
            return hasattr(obj, '_data') and hasattr(obj, 'requires_grad')
        
        ctx = Context()
        tensor_args = []
        processed_args = []
        
        # 查找可能的模块引用
        source_module = None
        module_refs = {}
        
        # 首先收集所有输入张量的模块引用
        for i, arg in enumerate(args):
            if is_tensor(arg):
                if hasattr(arg, '_module') and arg._module is not None:
                    module_refs[i] = arg._module
                    if source_module is None:
                        source_module = arg._module
        
        # 处理输入参数（不再转为 .data，直接传递 Tensor）
        for i, arg in enumerate(args):
            if is_tensor(arg):
                tensor_args.append(arg)
                processed_args.append(arg)  # 直接传递 Tensor
            else:
                processed_args.append(arg)
        
        # 保存模块引用信息到上下文
        ctx.module_ref = source_module
        ctx.module_refs = module_refs
        
        raw_output = cls.forward(ctx, *processed_args, **kwargs)
        if hasattr(raw_output, '_data') and hasattr(raw_output, 'data'):
            # 如果raw_output是Tensor对象，提取其data属性
            raw_output = raw_output.data
        elif hasattr(raw_output, 'data') and hasattr(raw_output, 'shape'):
            # 如果raw_output是Array对象，提取其data属性
            raw_output = raw_output.data
        elif not isinstance(raw_output, (list, float)):
            try:
                raw_output = arrays.array(raw_output, dtype=float)
            except (ValueError, TypeError):
                # 如果转换失败，尝试其他方法
                if hasattr(raw_output, 'tolist'):
                    raw_output = raw_output.tolist()
                else:
                    raw_output = float(raw_output)
        
        # 检查当前梯度模式
        if _engine._grad_mode == GradMode.INFERENCE:
            requires_grad = False
        else:
            # 修改检测逻辑：只要有一个输入tensor需要梯度，输出就需要梯度
            requires_grad = any(getattr(t, 'requires_grad', False) for t in tensor_args)
            # 允许通过kwargs显式设置requires_grad
            if 'requires_grad' in kwargs:
                requires_grad = kwargs['requires_grad']
        
        # 创建输出tensor
        from tensor_T import Tensor
        output = Tensor(
            raw_output,
            requires_grad=requires_grad,
            _grad_fn=cls if requires_grad else None,
            _children=tensor_args if requires_grad else []
        )
        
        # 传递模块引用
        if source_module is not None and hasattr(output, 'attach_module_reference'):
            output.attach_module_reference(source_module)
        
        # 如果需要梯度，设置计算图
        if requires_grad:
            output._ctx = ctx
                
            # 为每个需要梯度的输入张量添加对当前输出的引用
            for t in tensor_args:
                if getattr(t, 'requires_grad', False):
                    if not hasattr(t, '_output_refs'):
                        t._output_refs = []
                    t._output_refs.append(weakref.ref(output))
        
        return output

class DistAutogradContext:
    """分布式自动微分上下文
    
    用于管理分布式环境下的自动微分。
    """
    def __init__(self):
        """初始化分布式自动微分上下文"""
        self._worker_id = 0
        self._contexts = {}

class BackwardEngine:
    _instance = None
    _lock = threading.Lock()
    _grad_mode = GradMode.TRAINING
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init()
            return cls._instance
    
    def _init(self):
        self._dist_context = None
    
    def _compute_backward(self, func, *args, grad_outputs):
        from tensor_T import Tensor
        try:
            # 检查梯度是否有效
            for grad in (grad_outputs if isinstance(grad_outputs, (tuple, list)) else [grad_outputs]):
                if grad is not None:
                    grad_array = arrays.Array(grad.data.flatten())
                    isnan_result = arrays.isnan(grad_array)
                    isinf_result = arrays.isinf(grad_array)
                    if any(isnan_result.data) or any(isinf_result.data):
                        print("Warning: Invalid gradient in backward computation")
                        return tuple(Tensor(arrays.zeros_like(arg.data)) for arg in args)
            
            # 执行反向传播
            if isinstance(grad_outputs, (tuple, list)):
                grads = func.backward(*args, *grad_outputs)
            else:
                grads = func.backward(*args, grad_outputs)
            
            # 检查计算结果
            if grads is None:
                return tuple(None for _ in args)
            
            # 确保返回的是元组
            if not isinstance(grads, tuple):
                grads = (grads,)
            
            # 检查每个梯度
            valid_grads = []
            for grad, arg in zip(grads, args):
                if grad is None:
                    valid_grads.append(None)
                else:
                    grad_array = arrays.Array(grad.data.flatten())
                    isnan_result = arrays.isnan(grad_array)
                    isinf_result = arrays.isinf(grad_array)
                    if any(isnan_result.data) or any(isinf_result.data):
                        print("Warning: Invalid gradient after backward computation")
                        valid_grads.append(Tensor(arrays.zeros_like(arg.data)))
                    else:
                        valid_grads.append(grad)
            
            return tuple(valid_grads)
            
        except Exception as e:
            print(f"Error in backward computation: {str(e)}")
            return tuple(Tensor(arrays.zeros_like(arg.data)) for arg in args)
    
    def execute_backward(self, root, grad=None):
        """执行反向传播"""
        from tensor_T import Tensor
        if grad is None:
            root_array = arrays.Array(root.data)
            ones_like_array = arrays.ones_like(root_array)
            grad = Tensor(arrays.array(ones_like_array.data))
        
        # 初始化梯度字典，用于累积梯度
        all_grads = {}
        all_grads[id(root)] = grad
        
        # 构建拓扑排序
        visited = set()
        topo = []
        
        def build_topo(node):
            if id(node) in visited or node is None:
                return
            visited.add(id(node))
            if hasattr(node, '_children'):
                for child in node._children:
                    if child is not None:
                        build_topo(child)
            topo.append(node)
        
        build_topo(root)
        
        # 反向遍历拓扑排序
        for node in reversed(topo):
            if node is None or not hasattr(node, '_grad_fn') or node._grad_fn is None:
                continue
                
            # 获取当前节点的梯度
            node_grad = all_grads.get(id(node))
            if node_grad is None:
                continue
            
            # 调用backward计算梯度
            ctx = getattr(node, '_ctx', None)
            if ctx is None:
                continue
                
            try:
                # 调用backward函数
                grads = node._grad_fn.backward(ctx, node_grad)
                
                if grads is not None:
                    # 确保grads是元组
                    if not isinstance(grads, tuple):
                        grads = (grads,)
                    
                    # 将梯度分配给子节点（输入）
                    children = getattr(node, '_children', [])
                    for i, (child, grad) in enumerate(zip(children, grads)):
                        if child is None or grad is None:
                            continue
                            
                        # 确保grad是Tensor
                        if not isinstance(grad, Tensor):
                            grad = Tensor(grad)
                        
                        # 累积梯度到all_grads字典
                        child_id = id(child)
                        if child_id in all_grads:
                            all_grads[child_id] = all_grads[child_id] + grad
                        else:
                            all_grads[child_id] = grad
                            
            except Exception as e:
                print(f"反向传播错误: {e}")
                import traceback
                traceback.print_exc()
        
        # 将累积的梯度分配给需要梯度的张量
        for node in topo:
            if node is None or not getattr(node, 'requires_grad', False):
                continue
                
            node_id = id(node)
            if node_id in all_grads:
                grad = all_grads[node_id]
                if node.grad is None:
                    node.grad = grad
                else:
                    node.grad = node.grad + grad

# 创建全局引擎实例
_engine = BackwardEngine()

def backward(tensor, grad_tensor=None):
    """执行反向传播
    
    Args:
        tensor: 要计算梯度的张量
        grad_tensor: 初始梯度
    """
    _engine.execute_backward(tensor, grad_tensor)

def enable_grad():
    """启用梯度计算
    
    Returns:
        梯度模式守卫
    """
    return GradModeGuard(GradMode.TRAINING)

def no_grad():
    """禁用梯度计算
    
    Returns:
        梯度模式守卫
    """
    return GradModeGuard(GradMode.INFERENCE)

class GradModeGuard:
    """梯度模式守卫
    
    用于临时切换梯度模式，并在退出时恢复原来的模式。
    """
    __slots__ = ['prev_mode']
    
    def __init__(self, mode):
        """初始化梯度模式守卫
        
        Args:
            mode: 要切换到的梯度模式
        """
        self.prev_mode = _engine._grad_mode
        _engine._grad_mode = mode
    
    def __enter__(self):
        """进入上下文"""
        pass
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文，恢复原来的梯度模式"""
        _engine._grad_mode = self.prev_mode

class FunctionRegistry:
    """函数注册表
    
    用于注册和管理自定义函数。
    """
    _custom_funcs = {}
    
    @classmethod
    def register(cls, name, forward, backward):
        """注册自定义函数
        
        Args:
            name: 函数名称
            forward: 前向传播函数
            backward: 反向传播函数
        """
        class CustomFunction(Function):
            @staticmethod
            def forward(ctx, *args, **kwargs):
                return forward(ctx, *args, **kwargs)
            
            @staticmethod
            def backward(ctx, *grad_outputs):
                return backward(ctx, *grad_outputs)
        
        CustomFunction.__name__ = name
        cls._custom_funcs[name] = CustomFunction
    
    @classmethod
    def get(cls, name):
        """获取自定义函数
        
        Args:
            name: 函数名称
        
        Returns:
            注册的自定义函数
        """
        return cls._custom_funcs[name]

def checkpoint(func, *args):
    """梯度检查点
    
    用于减少内存使用，通过在前向传播时不保存中间结果，
    而是在反向传播时重新计算。
    
    Args:
        func: 要检查点的函数
        *args: 函数的参数
    
    Returns:
        函数的输出
    """
    class CheckpointFunction(Function):
        @staticmethod
        def forward(ctx, func, *args):
            ctx.save_for_backward(func, *args)
            with no_grad():
                return func(*args)
        
        @staticmethod
        def backward(ctx, *grad_outputs):
            func, *args = ctx.saved_tensors
            return (None,) + _engine._compute_backward(func, *args, *grad_outputs)
    
    return CheckpointFunction.apply(func, *args)

# === Debug control functions ===
def set_debug_mode(enabled=True):
    """设置调试模式
    
    Args:
        enabled: 是否启用调试模式
    """
    global DEBUG_AUTOGRAD
    DEBUG_AUTOGRAD = enabled
    _engine._debug = enabled

def get_debug_mode():
    """获取调试模式状态
    
    Returns:
        bool: 调试模式是否启用
    """
    return DEBUG_AUTOGRAD

class MatMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        from tensor_T import Tensor
        ctx.save_for_backward(a, b)  # 先保存 Tensor
        
        def extract_data_for_matmul(tensor):
            if hasattr(tensor, 'data'):
                data = tensor.data
                if hasattr(data, '_flat_data') and hasattr(data, 'shape'):
                    import arrays
                    try:
                        # 特殊情况：检查是否是空数组
                        if 0 in data.shape or len(data._flat_data) == 0:
                            print(f"🚨 发现空数组: shape={data.shape}, _flat_data长度={len(data._flat_data)}")
                            # 创建正确形状的零数组
                            zeros = arrays.zeros(data.shape)
                            return zeros.data
                        
                        arrays_obj = arrays.Array(data.tolist())
                        return arrays_obj.data
                    except:
                        flat_data = data._flat_data
                        shape = data.shape
                        
                        # 处理空数组情况
                        if 0 in shape or len(flat_data) == 0:
                            print(f"🚨 处理异常时发现空数组: shape={shape}, flat_data长度={len(flat_data)}")
                            import arrays
                            zeros = arrays.zeros(shape)
                            return zeros.data
                        
                        import arrays
                        as_compatible = arrays.asarray_numpy_compatible(flat_data, dtype='float')
                        return as_compatible.data.reshape(shape)
                else:
                    # 其他类型的data，直接返回
                    return data
            else:
                return tensor
        
        a_data = extract_data_for_matmul(a)
        b_data = extract_data_for_matmul(b)
        
        # 添加数据提取的调试信息
        
        if hasattr(a_data, 'shape') and hasattr(b_data, 'shape'):
            # 检查是否有空数组参与运算
            if 0 in a_data.shape or 0 in b_data.shape:
                print(f"🚨 检测到空数组! a_data.shape: {a_data.shape}, b_data.shape: {b_data.shape}")
                # 对于空数组的matmul，返回适当形状的零数组
                if len(a_data.shape) == 2 and len(b_data.shape) == 2:
                    result_shape = (a_data.shape[0], b_data.shape[1])
                    print(f"🚨 返回零数组，形状: {result_shape}")
                    import arrays
                    zeros = arrays.zeros(result_shape)
                    print(f"🚨 zeros.data: {zeros.data}")
                    return Tensor(zeros.data, requires_grad=a.requires_grad or b.requires_grad)
                else:
                    # 其他维度情况，返回空的结果
                    print(f"🚨 返回空结果 (0,0)")
                    import arrays
                    zeros = arrays.zeros((0, 0))
                    return Tensor(zeros.data, requires_grad=a.requires_grad or b.requires_grad)
        
        # 执行矩阵乘法
        try:
            
            #result = strong_mat.smart_matmul(a_data, b_data)
            result = strong_matmul.perfect_matmul(a_data, b_data)
            

            
            # 提取结果数据
            if hasattr(result, 'data'):
                result_data = result.data
                print(f"   result.data类型: {type(result_data)}")
                print(f"   result.data形状: {getattr(result_data, 'shape', 'No shape')}")
            else:
                result_data = result
                print(f"   直接使用result作为result_data")
                
            return Tensor(result_data, requires_grad=a.requires_grad or b.requires_grad)
        except Exception as e:
            print(f"🚨 MatMul错误: {e}")
            print(f"🚨 错误类型: {type(e)}")
            import traceback
            print(f"🚨 完整异常信息:")
            traceback.print_exc()
            print(f"🚨 A shape: {getattr(a_data, 'shape', 'No shape')}")
            print(f"🚨 B shape: {getattr(b_data, 'shape', 'No shape')}")
            print(f"🚨 A 数据类型: {type(a_data)}")
            print(f"🚨 B 数据类型: {type(b_data)}")
            if hasattr(a_data, '__len__') and len(str(a_data)) < 300:
                print(f"🚨 A 数据内容: {a_data}")
            if hasattr(b_data, '__len__') and len(str(b_data)) < 300:
                print(f"🚨 B 数据内容: {b_data}")
            # 返回一个合理的默认值
            import arrays
            zeros = arrays.zeros((1, 1))
            print(f"🚨 返回默认零张量: {zeros.data}")
            return Tensor(zeros.data, requires_grad=a.requires_grad or b.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        # 获取保存的输入
        a, b = ctx.saved_tensors
        
        # 导入arrays用于安全的转置操作
        import arrays
        
        def smart_transpose(data):
            """智能转置，处理不同维度的情况，完全使用arrays模块"""
            try:
                if hasattr(data, 'shape'):
                    if len(data.shape) == 2:
                        # 标准2D矩阵转置
                        transposed = arrays.transpose(arrays.Array(data))
                        if hasattr(transposed, 'data'):
                            result = transposed.data
                        else:
                            result = transposed
                        
                        # 确保结果有shape属性
                        if not hasattr(result, 'shape'):
                            # 使用arrays重建数组
                            result_array = arrays.asarray_numpy_compatible(result)
                            result = result_array.data.reshape(data.shape[1], data.shape[0])
                        
                        return result
                    elif len(data.shape) == 1:
                        # 1D向量，转换为列向量然后转置
                        reshaped = data.reshape(-1, 1)
                        transposed = arrays.transpose(arrays.Array(reshaped))
                        if hasattr(transposed, 'data'):
                            result = transposed.data
                        else:
                            result = transposed
                        
                        # 确保结果有shape属性
                        if not hasattr(result, 'shape'):
                            result_array = arrays.asarray_numpy_compatible(result)
                            result = result_array.data.reshape(1, -1)
                        
                        return result
                    else:
                        print(f"  不支持的维度数: {len(data.shape)}")
                        return data
                else:
                    print(f"  数据没有shape属性")
                    return data
            except Exception as e:
                print(f"  转置失败: {e}")
                # 返回原数据的转置版本（使用arrays）
                try:
                    if hasattr(data, 'shape') and len(data.shape) == 2:
                        result = arrays.transpose(arrays.Array(data))
                        if hasattr(result, 'data'):
                            return result.data
                        else:
                            result_array = arrays.asarray_numpy_compatible(result)
                            return result_array.data
                except:
                    pass
                return data
        
        def smart_matmul_fixed(x, y, operation_name=""):
            try:
                #result = strong_mat.smart_matmul(x, y)
                result = strong_matmul.perfect_matmul(x, y)
                
                # 提取结果数据
                if hasattr(result, 'data'):
                    return result.data
                else:
                    return result
                
            except Exception as e:
                print(f"  {operation_name}异常: {e}")
                print(f"  形状: {getattr(x, 'shape', 'No shape')} @ {getattr(y, 'shape', 'No shape')}")
                return None
        
        # === 关键修复：针对不同矩阵乘法类型的正确梯度计算，完全使用arrays ===
        
        # 获取形状信息
        a_shape = getattr(a.data, 'shape', ())
        b_shape = getattr(b.data, 'shape', ())
        grad_shape = getattr(grad_output.data, 'shape', ())
        
        grad_a = None
        grad_b = None
        
        # 情况1: 2D @ 2D
        if len(a_shape) == 2 and len(b_shape) == 2:
            b_t = smart_transpose(b.data)
            grad_a = smart_matmul_fixed(grad_output.data, b_t, "grad_a(2D@2D)")
            
            a_t = smart_transpose(a.data)
            grad_b = smart_matmul_fixed(a_t, grad_output.data, "grad_b(2D@2D)")
        
        # 情况2: 2D @ 1D (矩阵 × 向量 = 向量)
        elif len(a_shape) == 2 and len(b_shape) == 1:
            # grad_A = grad_output.reshape(-1, 1) @ B.reshape(1, -1)
            grad_out_reshaped = grad_output.data.reshape(-1, 1)  # (n,) -> (n, 1)
            b_reshaped = b.data.reshape(1, -1)  # (m,) -> (1, m)
            
            # 使用arrays模块进行矩阵乘法
            grad_out_array = arrays.Array(grad_out_reshaped)
            b_reshaped_array = arrays.Array(b_reshaped)
            grad_a_result = arrays.matmul(grad_out_array, b_reshaped_array)
            if hasattr(grad_a_result, 'data'):
                grad_a = grad_a_result.data
            else:
                grad_a_array = arrays.asarray_numpy_compatible(grad_a_result)
                grad_a = grad_a_array.data
            
            # grad_B = A.T @ grad_output
            a_t = smart_transpose(a.data)
            grad_b = smart_matmul_fixed(a_t, grad_output.data, "grad_b(2D@1D)")
        
        # 情况3: 1D @ 2D (向量 × 矩阵 = 向量)
        elif len(a_shape) == 1 and len(b_shape) == 2:
            # grad_A = grad_output @ B.T
            b_t = smart_transpose(b.data)
            grad_a = smart_matmul_fixed(grad_output.data, b_t, "grad_a(1D@2D)")
            
            # grad_B = A.reshape(-1, 1) @ grad_output.reshape(1, -1)
            a_reshaped = a.data.reshape(-1, 1)  # (n,) -> (n, 1)
            grad_out_reshaped = grad_output.data.reshape(1, -1)  # (m,) -> (1, m)
            
            # 使用arrays模块进行矩阵乘法
            a_reshaped_array = arrays.Array(a_reshaped)
            grad_out_array = arrays.Array(grad_out_reshaped)
            grad_b_result = arrays.matmul(a_reshaped_array, grad_out_array)
            if hasattr(grad_b_result, 'data'):
                grad_b = grad_b_result.data
            else:
                grad_b_array = arrays.asarray_numpy_compatible(grad_b_result)
                grad_b = grad_b_array.data
        
        # 情况4: 1D @ 1D (向量点积 = 标量)
        elif len(a_shape) == 1 and len(b_shape) == 1:
            # 对于点积，梯度就是对方向量乘以标量梯度
            grad_a = grad_output.data * b.data  # 标量 * 向量 = 向量
            grad_b = grad_output.data * a.data  # 标量 * 向量 = 向量
        
        # 确保梯度不为None且形状正确
        if grad_a is None:
            print("grad_a为None，使用零梯度")
            grad_a = arrays.zeros_like(arrays.Array(a.data)).data
        if grad_b is None:
            print("grad_b为None，使用零梯度")
            grad_b = arrays.zeros_like(arrays.Array(b.data)).data
        
        # 强制调整形状以匹配原张量
        if hasattr(grad_a, 'shape') and grad_a.shape != a_shape:
            print(f"🚨 强制调整grad_a形状: {grad_a.shape} -> {a_shape}")
            print(f"   原始grad_a类型: {type(grad_a)}")
            print(f"   原始grad_a前10个: {str(grad_a)[:100]}...")
            grad_a = arrays.zeros(a_shape).data
            print(f"   调整后grad_a类型: {type(grad_a)}")
            print(f"   调整后grad_a前10个: {str(grad_a)[:100]}...")
        if hasattr(grad_b, 'shape') and grad_b.shape != b_shape:
            print(f"🚨 强制调整grad_b形状: {grad_b.shape} -> {b_shape}")
            print(f"   原始grad_b类型: {type(grad_b)}")
            print(f"   原始grad_b前10个: {str(grad_b)[:100]}...")
            grad_b = arrays.zeros(b_shape).data
            print(f"   调整后grad_b类型: {type(grad_b)}")
            print(f"   调整后grad_b前10个: {str(grad_b)[:100]}...")
        
        # 添加返回前的调试信息
        print(f"🔍 MatMul.backward返回:")
        print(f"   grad_a类型: {type(grad_a)}, 形状: {getattr(grad_a, 'shape', 'No shape')}")
        print(f"   grad_b类型: {type(grad_b)}, 形状: {getattr(grad_b, 'shape', 'No shape')}")
        if hasattr(grad_a, '__len__') and len(str(grad_a)) < 200:
            print(f"   grad_a内容: {grad_a}")
        if hasattr(grad_b, '__len__') and len(str(grad_b)) < 200:
            print(f"   grad_b内容: {grad_b}")

        return grad_a, grad_b

# 在 Tensor 类中添加 matmul 方法
def matmul(self, other):
    """矩阵乘法"""
    from tensor_T import Tensor
    if not isinstance(other, Tensor):
        other = Tensor(other)
    return MatMul.apply(self, other)

__all__ = [
    'Function', 'backward', 'no_grad', 'enable_grad',
    'checkpoint', 'FunctionRegistry'
]
