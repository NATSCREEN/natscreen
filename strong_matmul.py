"""
Strong Matmul Library - 强大的矩阵乘法库
专门替代np.matmul，不使用任何第三方库
完全自主实现，支持各种维度的矩阵乘法操作
"""

import arrays  # 导入arrays模块以确保兼容性

def perfect_matmul(a, b):
    """
    强大的矩阵乘法函数，完全替代np.matmul
    
    特点：
    - 完全自主实现，不依赖第三方库
    - 支持1D、2D、多维数组的矩阵乘法
    - 智能维度处理和广播
    - 强大的错误处理和边界情况处理
    - 高效的算法实现
    
    Args:
        a: 第一个矩阵/向量（标量、列表、多维列表）
        b: 第二个矩阵/向量（标量、列表、多维列表）
        
    Returns:
        矩阵乘法结果，包装为arrays.Array对象以保持兼容性
    """
    
    # 1. 预处理和验证输入
    if a is None or b is None:
        raise ValueError("Input arrays cannot be None")
    
    # 2. 提取真实数据 - 处理arrays.Array对象和memoryview
    a_data = _extract_data(a)
    b_data = _extract_data(b)
    
    # 3. 标准化输入数据
    a_processed, a_shape = _process_input(a_data)
    b_processed, b_shape = _process_input(b_data)
    

    # 4. 验证维度兼容性
    _validate_matmul_compatibility(a_shape, b_shape)
    
    # 5. 根据维度选择合适的乘法策略
    try:
        result = _dispatch_matmul(a_processed, b_processed, a_shape, b_shape)
        

    except Exception as e:
        print(f"🚨 strong_matmul计算异常: {e}")
        print(f"🚨 异常类型: {type(e)}")
        print(f"   A形状: {a_shape}, B形状: {b_shape}")
        print(f"🚨 A 数据类型: {type(a_data)}")
        print(f"🚨 B 数据类型: {type(b_data)}")
        print(f"🚨 A 处理后类型: {type(a_processed)}")
        print(f"🚨 B 处理后类型: {type(b_processed)}")
        
        # 特殊处理：如果是矩阵向量乘法失败，尝试修复
        if "don't match vector length" in str(e) and len(a_shape) == 2 and len(b_shape) == 1:
            print(f"🔧 尝试修复矩阵向量乘法...")
            # 检查数据实际长度
            if isinstance(b_processed, list):
                actual_b_len = len(b_processed)
                print(f"🔧 B的实际长度: {actual_b_len}")
                print(f"🔧 A的列数: {a_shape[1]}")
                
                # 如果长度匹配，但形状不对，可能是数据提取问题
                if actual_b_len == a_shape[1]:
                    print(f"🔧 长度匹配，重新尝试计算...")
                    try:
                        result = _matrix_vector_multiply(a_processed, b_processed)
                        print(f"🔧 修复成功!")
                    except Exception as e2:
                        print(f"🔧 修复失败: {e2}")
                        raise e
                else:
                    raise e
            else:
                raise e
        else:
            raise e
    
    # 6. 将结果包装为arrays.Array对象以保持兼容性
    try:

        
        # 如果结果是标量
        if isinstance(result, (int, float, complex)):
            result_array = arrays.Array([result])
            if (len(a_shape) == 2 and len(b_shape) == 2 and 
                a_shape[0] >= 19 and a_shape[1] >= 1000 and b_shape[0] >= 1000):
                print(f"   包装标量为: {result_array.shape}")
            return result_array
        # 如果结果是列表，包装为arrays.Array
        elif isinstance(result, list):
            result_array = arrays.Array(result)
            return result_array
        else:
            # 如果已经是arrays.Array，直接返回
            if (len(a_shape) == 2 and len(b_shape) == 2 and 
                a_shape[0] >= 19 and a_shape[1] >= 1000 and b_shape[0] >= 1000):
                print(f"   直接返回已有Array: {result.shape if hasattr(result, 'shape') else 'No shape'}")
            return result
    except Exception as e:
        print(f"🚨 结果包装异常: {e}")
        print(f"   结果类型: {type(result)}")
        print(f"   原始A形状: {a_shape}, B形状: {b_shape}")
        import traceback
        traceback.print_exc()
        # 重新抛出异常而不是返回错误结果
        raise e

def _extract_data(obj):
    """提取对象的真实数据，处理arrays.Array对象、FinalArrayCompatible对象和memoryview"""
    # 检查是否是FinalArrayCompatible对象
    if hasattr(obj, '__class__') and 'FinalArrayCompatible' in str(obj.__class__):
        # 对于FinalArrayCompatible对象，使用_data属性
        if hasattr(obj, '_data'):
            return obj._data
        elif hasattr(obj, 'data'):
            return obj.data
    
    # 如果是arrays.Array对象，提取其data属性
    if hasattr(obj, 'data'):
        data = obj.data
        # 如果data是memoryview，转换为列表
        if isinstance(data, memoryview):
            return _memoryview_to_list(data, obj.shape if hasattr(obj, 'shape') else None)
        return data
    # 如果直接是memoryview，转换为列表
    elif isinstance(obj, memoryview):
        return _memoryview_to_list(obj)
    # 否则直接返回
    return obj

def _memoryview_to_list(mv, shape=None):
    """将memoryview转换为多维列表结构"""
    try:
        # 首先尝试转换为列表
        flat_list = list(mv)
        
        # 如果有shape信息，重构为多维数组
        if shape is not None and len(shape) > 1:
            return _reconstruct_from_flat(flat_list, shape)
        else:
            # 没有shape信息，返回扁平列表
            return flat_list
    except Exception as e:
        # 如果转换失败，尝试其他方法
        try:
            # 尝试numpy风格的转换
            flat_list = [float(x) for x in mv]
            if shape is not None and len(shape) > 1:
                return _reconstruct_from_flat(flat_list, shape)
            return flat_list
        except:
            # 最后的备选方案
            return list(mv.tolist()) if hasattr(mv, 'tolist') else [mv]

def _reconstruct_from_flat(flat_data, shape):
    """从扁平数据重构多维数组"""
    if len(shape) == 1:
        return flat_data[:shape[0]]
    
    def _build_recursive(data, dims, start_idx=0):
        if len(dims) == 1:
            return data[start_idx:start_idx + dims[0]], start_idx + dims[0]
        
        result = []
        current_idx = start_idx
        for _ in range(dims[0]):
            sub_array, current_idx = _build_recursive(data, dims[1:], current_idx)
            result.append(sub_array)
        return result, current_idx
    
    result, _ = _build_recursive(flat_data, shape)
    return result

def _process_input(array):
    """
    处理输入数组，返回标准化的数据和形状信息
    保持高维结构完整性，正确处理各种数据类型
    """
    # 检查是否是FinalArrayCompatible对象，如果是，获取其真实数据和形状
    if hasattr(array, '__class__') and 'FinalArrayCompatible' in str(array.__class__):
        data = array._data if hasattr(array, '_data') else array.data
        shape = array._shape if hasattr(array, '_shape') else array.shape
        
        # 调试信息
        if (hasattr(array, '_shape') and len(array._shape) == 2 and 
            array._shape[0] >= 19 and array._shape[1] >= 1000):
            print(f"🔍 _process_input 处理 FinalArrayCompatible:")
            print(f"   原始形状: {shape}")
            print(f"   数据类型: {type(data)}")
            if isinstance(data, list) and len(data) > 0:
                print(f"   数据行数: {len(data)}")
                if isinstance(data[0], list):
                    print(f"   第一行长度: {len(data[0])}")
        
        return data, shape
    
    # 处理标量
    if isinstance(array, (int, float, complex)):
        return [[array]], (1, 1)
    
    # 处理memoryview对象（这种情况应该在_extract_data中处理了，但作为备份）
    if isinstance(array, memoryview):
        array = list(array)
    
    # 获取形状信息
    shape = _get_array_shape_matmul(array)
    
    if not shape:
        # 标量情况
        return [[array]], (1, 1)
    elif len(shape) == 1:
        # 一维向量 - 直接返回，不要额外包装
        return array, shape
    else:
        # 多维数组，保持原有结构
        return array, shape

def _flatten_array_matmul(array):
    """专门为矩阵乘法设计的扁平化函数"""
    if isinstance(array, (int, float, complex)):
        return [array]
    
    result = []
    
    def _flatten_recursive(data):
        if isinstance(data, (list, tuple)):
            for item in data:
                _flatten_recursive(item)
        elif hasattr(data, '__iter__') and not isinstance(data, (str, bytes)):
            try:
                for item in data:
                    _flatten_recursive(item)
            except TypeError:
                result.append(float(data))
        else:
            result.append(float(data))
    
    _flatten_recursive(array)
    return result

def _get_array_shape_matmul(array):
    """获取数组形状，专门为矩阵乘法优化，正确处理多维数组"""
    if isinstance(array, (int, float, complex)):
        return ()
    
    # 如果不是列表或元组，尝试获取长度
    if not isinstance(array, (list, tuple)):
        if hasattr(array, '__len__'):
            try:
                return (len(array),)
            except:
                return ()
        return ()
    
    def _shape_recursive(data):
        if not isinstance(data, (list, tuple)):
            return []
        if not data:
            return [0]
        
        # 检查第一个元素
        first_element = data[0]
        first_shape = _shape_recursive(first_element)
        
        # 检查所有元素是否具有相同的形状
        all_same_shape = True
        for item in data[1:]:
            if _shape_recursive(item) != first_shape:
                all_same_shape = False
                break
        
        if all_same_shape:
            return [len(data)] + first_shape
        else:
            # 如果形状不一致，只返回最外层维度
            return [len(data)]
    
    shape = _shape_recursive(array)
    return tuple(shape)

def _validate_matmul_compatibility(shape_a, shape_b):
    """验证两个数组是否可以进行矩阵乘法"""
    if not shape_a or not shape_b:
        # 标量情况，总是兼容的
        return
    
    # 如果任一个是标量（形状为(1,1)），则跳过维度检查，因为是标量乘法
    if (len(shape_a) == 2 and shape_a == (1, 1)) or (len(shape_b) == 2 and shape_b == (1, 1)):
        return
    
    # 对于高维数组，我们主要检查最后两个维度的兼容性
    if len(shape_a) >= 2 and len(shape_b) >= 2:
        # 检查矩阵乘法的核心维度兼容性
        if shape_a[-1] != shape_b[-2]:
            # 但是对于批量矩阵乘法，可能有广播情况，先尝试宽松检查
            if not _can_broadcast_matmul(shape_a, shape_b):
                raise ValueError(f"Incompatible dimensions for matrix multiplication: "
                               f"shape_a={shape_a}, shape_b={shape_b}. "
                               f"Last dimension of a ({shape_a[-1]}) must match "
                               f"second-to-last dimension of b ({shape_b[-2]})")
    elif len(shape_a) == 1 and len(shape_b) >= 2:
        # a是1D，b是2D+
        if shape_a[0] != shape_b[-2]:
            raise ValueError(f"Incompatible dimensions: vector of length {shape_a[0]} "
                           f"cannot multiply matrix with {shape_b[-2]} columns")
    elif len(shape_a) >= 2 and len(shape_b) == 1:
        # a是2D+，b是1D
        if shape_a[-1] != shape_b[0]:
            raise ValueError(f"Incompatible dimensions: matrix with {shape_a[-1]} columns "
                           f"cannot multiply vector of length {shape_b[0]}")
    elif len(shape_a) == 1 and len(shape_b) == 1:
        # 两个都是1D，进行点积
        if shape_a[0] != shape_b[0]:
            raise ValueError(f"Incompatible dimensions for dot product: "
                           f"{shape_a[0]} vs {shape_b[0]}")

def _can_broadcast_matmul(shape_a, shape_b):
    """检查两个形状是否可以进行广播矩阵乘法"""
    # 简化的广播检查：如果维度数量相同，且前面的维度兼容
    if len(shape_a) == len(shape_b):
        # 检查除了最后两个维度之外的所有维度
        for i in range(len(shape_a) - 2):
            if shape_a[i] != shape_b[i] and shape_a[i] != 1 and shape_b[i] != 1:
                return False
        # 检查矩阵乘法维度
        return shape_a[-1] == shape_b[-2]
    return False

def _dispatch_matmul(a, b, shape_a, shape_b):
    """根据维度分派到相应的矩阵乘法实现，改进判断逻辑"""
    
    # 首先严格检查：如果任何一个是大型矩阵，绝对不进行标量乘法
    is_large_matrix_a = (shape_a and len(shape_a) >= 2 and 
                        (shape_a[0] > 1 or shape_a[1] > 1) and
                        (shape_a[0] * shape_a[1] > 1))
    
    is_large_matrix_b = (shape_b and len(shape_b) >= 2 and 
                        (shape_b[0] > 1 or shape_b[1] > 1) and
                        (shape_b[0] * shape_b[1] > 1))
    
    # 如果任一是大型矩阵，直接跳到适当的矩阵乘法
    if is_large_matrix_a or is_large_matrix_b:
        # 跳过标量检查，直接进行矩阵乘法
        pass
    else:
        # 只有当两个都不是大型矩阵时，才检查标量情况
        # 标量情况 - 更严格的判断，排除大型矩阵
        is_scalar_a = (not shape_a or 
                       (len(shape_a) == 2 and shape_a == (1, 1) and isinstance(a, list) and len(a) == 1 and len(a[0]) == 1) or
                       (len(shape_a) == 1 and shape_a[0] == 1 and isinstance(a, list) and len(a) == 1) or
                       (isinstance(a, (int, float, complex))))
        
        is_scalar_b = (not shape_b or 
                       (len(shape_b) == 2 and shape_b == (1, 1) and isinstance(b, list) and len(b) == 1 and len(b[0]) == 1) or
                       (len(shape_b) == 1 and shape_b[0] == 1 and isinstance(b, list) and len(b) == 1) or
                       (isinstance(b, (int, float, complex))))
        
        # 只有当至少一个确实是标量时才进行标量乘法
        if is_scalar_a or is_scalar_b:
            return _scalar_multiply(a, b, shape_a, shape_b)
    
    # 1D x 1D = 点积
    if len(shape_a) == 1 and len(shape_b) == 1:
        return _dot_product(a, b)
    
    # 1D x 2D = 向量与矩阵相乘
    elif len(shape_a) == 1 and len(shape_b) == 2:
        return _vector_matrix_multiply(a, b)
    
    # 2D x 1D = 矩阵与向量相乘
    elif len(shape_a) == 2 and len(shape_b) == 1:
        return _matrix_vector_multiply(a, b)
    
    # 2D x 2D = 标准矩阵乘法
    elif len(shape_a) == 2 and len(shape_b) == 2:
        return _matrix_matrix_multiply(a, b)
    
    # 高维情况
    else:
        return _high_dimensional_matmul(a, b, shape_a, shape_b)

def _scalar_multiply(a, b, shape_a, shape_b):
    """处理标量乘法，改进数据类型处理"""
    # 提取标量值，更安全的方式
    scalar_a = _extract_scalar_value(a, shape_a)
    scalar_b = _extract_scalar_value(b, shape_b)
    
    # 判断哪个是标量，哪个是数组
    if scalar_a is not None and scalar_b is not None:
        # 两个都是标量
        return scalar_a * scalar_b
    elif scalar_a is not None:
        # a是标量，b是数组
        return _scalar_array_multiply(scalar_a, b)
    else:
        # a是数组，b是标量
        return _scalar_array_multiply(scalar_b, a)

def _extract_scalar_value(data, shape):
    """安全地提取标量值，避免对大型矩阵调用float()"""
    # 安全检查：如果是大型矩阵，立即返回None
    if shape and len(shape) >= 2:
        total_elements = 1
        for dim in shape:
            total_elements *= dim
        if total_elements > 1:
            # 这是一个大型矩阵，绝对不提取标量值
            return None
    
    # 检查是否是FinalArrayCompatible对象且是大型的
    if hasattr(data, '_shape') and hasattr(data, '_data'):
        if hasattr(data, '_shape') and len(data._shape) >= 2:
            total_elements = 1
            for dim in data._shape:
                total_elements *= dim
            if total_elements > 1:
                return None
    
    # 首先检查形状：只有真正的标量或(1,1)矩阵才继续
    if not shape or (len(shape) == 1 and shape[0] == 1) or (len(shape) == 2 and shape == (1, 1)):
        try:
            if isinstance(data, list):
                # 对于(1,1)矩阵
                if len(data) == 1 and isinstance(data[0], list) and len(data[0]) == 1:
                    # 确保内容是数值而不是FinalArrayCompatible对象
                    inner_val = data[0][0]
                    if isinstance(inner_val, (int, float, complex)):
                        return float(inner_val)
                    elif hasattr(inner_val, '__float__') and not hasattr(inner_val, '_shape') and not hasattr(inner_val, '_data'):
                        # 只有当对象有__float__方法且不是数组对象时才调用
                        return float(inner_val)
                # 对于长度为1的列表
                elif len(data) == 1:
                    inner_val = data[0]
                    if isinstance(inner_val, (int, float, complex)):
                        return float(inner_val)
                    elif hasattr(inner_val, '__float__') and not hasattr(inner_val, '_shape') and not hasattr(inner_val, '_data'):
                        return float(inner_val)
            elif isinstance(data, (int, float, complex)):
                return float(data)
            # 添加对FinalArrayCompatible对象的检查
            elif hasattr(data, '_shape') and hasattr(data, '_data'):
                # 这是一个FinalArrayCompatible对象，不要调用float()
                if data._shape == () or data._shape == (1,) or data._shape == (1, 1):
                    # 只有真正的标量形状才提取值
                    if data._shape == ():
                        return float(data._data) if isinstance(data._data, (int, float, complex)) else None
                    elif data._shape == (1,) and isinstance(data._data, list) and len(data._data) == 1:
                        return float(data._data[0]) if isinstance(data._data[0], (int, float, complex)) else None
                    elif data._shape == (1, 1) and isinstance(data._data, list) and len(data._data) == 1 and isinstance(data._data[0], list) and len(data._data[0]) == 1:
                        return float(data._data[0][0]) if isinstance(data._data[0][0], (int, float, complex)) else None
                # 对于大型矩阵，直接返回None
                return None
        except (ValueError, TypeError, IndexError, AttributeError):
            pass
    return None

def _scalar_array_multiply(scalar, array):
    """标量与数组相乘，改进数据类型处理，避免对大型矩阵调用float()"""
    def _multiply_recursive(data):
        if isinstance(data, list):
            return [_multiply_recursive(item) for item in data]
        elif isinstance(data, (int, float, complex)):
            return scalar * data
        elif isinstance(data, memoryview):
            # 如果遇到memoryview，先转换为float
            return scalar * float(data)
        elif hasattr(data, '_shape') and hasattr(data, '_data'):
            # 这是一个FinalArrayCompatible对象，不要调用float()
            # 直接返回错误信息，因为标量不应该与大型矩阵相乘
            raise ValueError(f"Cannot multiply scalar with FinalArrayCompatible array of shape {data._shape}")
        else:
            # 尝试转换为float进行乘法，但更加谨慎
            try:
                # 只有当对象确实应该是数值时才调用float()
                if hasattr(data, '__float__') and not hasattr(data, '_shape'):
                    return scalar * float(data)
                else:
                    # 如果转换失败，返回0或抛出错误
                    return 0.0
            except (ValueError, TypeError):
                # 如果转换失败，返回0或抛出错误
                return 0.0
    
    return _multiply_recursive(array)

def _dot_product(vec_a, vec_b):
    """计算两个向量的点积"""
    if len(vec_a) != len(vec_b):
        raise ValueError(f"Vector lengths don't match: {len(vec_a)} vs {len(vec_b)}")
    
    result = 0.0
    for i in range(len(vec_a)):
        result += vec_a[i] * vec_b[i]
    
    return result

def _vector_matrix_multiply(vector, matrix):
    """向量与矩阵相乘 (1D @ 2D)"""
    if len(vector) != len(matrix):
        raise ValueError(f"Vector length {len(vector)} doesn't match matrix rows {len(matrix)}")
    
    if not matrix or not matrix[0]:
        raise ValueError("Matrix cannot be empty")
    
    cols = len(matrix[0])
    result = []
    
    for j in range(cols):
        sum_val = 0.0
        for i in range(len(vector)):
            sum_val += vector[i] * matrix[i][j]
        result.append(sum_val)
    
    return result

def _matrix_vector_multiply(matrix, vector):
    """矩阵与向量相乘 (2D @ 1D)"""
    if not matrix or len(matrix[0]) != len(vector):
        raise ValueError(f"Matrix columns {len(matrix[0]) if matrix else 0} "
                        f"don't match vector length {len(vector)}")
    
    result = []
    for row in matrix:
        sum_val = 0.0
        for i in range(len(vector)):
            sum_val += row[i] * vector[i]
        result.append(sum_val)
    
    return result

def _matrix_matrix_multiply(matrix_a, matrix_b):
    """标准矩阵乘法 (2D @ 2D)"""
    if not matrix_a or not matrix_b or not matrix_a[0] or not matrix_b[0]:
        raise ValueError("Matrices cannot be empty")
    
    rows_a, cols_a = len(matrix_a), len(matrix_a[0])
    rows_b, cols_b = len(matrix_b), len(matrix_b[0])
    
    if cols_a != rows_b:
        raise ValueError(f"Cannot multiply {rows_a}x{cols_a} matrix with {rows_b}x{cols_b} matrix")
    
    # 添加大型矩阵乘法的调试信息
    if rows_a >= 19 and cols_a >= 500 and rows_b >= 500 and cols_b >= 500:
        print(f"🔍 _matrix_matrix_multiply处理大型矩阵: ({rows_a}, {cols_a}) @ ({rows_b}, {cols_b})")
        print(f"   预期结果形状: ({rows_a}, {cols_b})")
    
    # 初始化结果矩阵
    try:
        result = [[0.0 for _ in range(cols_b)] for _ in range(rows_a)]
        
        # 添加内存检查
        if rows_a >= 19 and cols_b >= 500:
            print(f"   结果矩阵初始化成功: {len(result)} x {len(result[0])}")
        
    except Exception as e:
        print(f"🚨 结果矩阵初始化失败: {e}")
        raise e
    
    # 执行矩阵乘法
    try:
        for i in range(rows_a):
            for j in range(cols_b):
                sum_val = 0.0
                for k in range(cols_a):
                    sum_val += matrix_a[i][k] * matrix_b[k][j]
                result[i][j] = sum_val
            

                
    except Exception as e:
        print(f"🚨 矩阵乘法计算失败: {e}")
        print(f"   失败位置: 行{i if 'i' in locals() else '?'}, 列{j if 'j' in locals() else '?'}, k{k if 'k' in locals() else '?'}")
        raise e
    

    
    return result

def _high_dimensional_matmul(a, b, shape_a, shape_b):
    """处理高维数组的矩阵乘法"""
    # 对于高维情况，我们需要正确处理批次维度和多头注意力维度
    # 通常在Transformer中，形状为 (batch_size, num_heads, seq_len, head_dim)
    
    # 如果两个都是高维，进行批量矩阵乘法
    if len(shape_a) > 2 and len(shape_b) > 2:
        return _batch_matmul_recursive(a, b, shape_a, shape_b)
    elif len(shape_a) > 2:
        # a是高维，b是2D，需要广播
        return _high_dim_2d_matmul(a, b, shape_a, shape_b)
    elif len(shape_b) > 2:
        # a是2D，b是高维，需要广播
        return _2d_high_dim_matmul(a, b, shape_a, shape_b)
    else:
        # 应该不会到达这里，但作为安全措施
        return _matrix_matrix_multiply(a, b)

def _batch_matmul_recursive(a, b, shape_a, shape_b):
    """递归处理批量矩阵乘法，保持高维结构"""
    # 如果两个都是2D矩阵，进行基础矩阵乘法
    if len(shape_a) == 2 and len(shape_b) == 2:
        return _matrix_matrix_multiply(a, b)
    
    # 如果两个都是相同的高维度
    if len(shape_a) == len(shape_b) and len(shape_a) > 2:
        result = []
        batch_size = min(len(a), len(b))
        
        for i in range(batch_size):
            # 递归处理每个批次/头部
            sub_result = _batch_matmul_recursive(
                a[i], b[i], 
                shape_a[1:], shape_b[1:]
            )
            result.append(sub_result)
        
        return result
    
    # 如果维度不同，尝试降维处理
    elif len(shape_a) > len(shape_b):
        # a是高维，b是低维
        result = []
        for i in range(len(a)):
            sub_result = _batch_matmul_recursive(
                a[i], b, 
                shape_a[1:], shape_b
            )
            result.append(sub_result)
        return result
    
    elif len(shape_b) > len(shape_a):
        # b是高维，a是低维
        result = []
        for i in range(len(b)):
            sub_result = _batch_matmul_recursive(
                a, b[i], 
                shape_a, shape_b[1:]
            )
            result.append(sub_result)
        return result
    
    else:
        # 其他情况，尝试2D矩阵乘法
        return _matrix_matrix_multiply(a, b)

def _high_dim_2d_matmul(a, b, shape_a, shape_b):
    """高维数组与2D矩阵相乘，保持高维结构"""
    if len(shape_a) == 3:
        # 3D @ 2D: (batch, seq, dim) @ (dim, out_dim) -> (batch, seq, out_dim)
        result = []
        for i in range(len(a)):
            sub_result = _matrix_matrix_multiply(a[i], b)
            result.append(sub_result)
        return result
    
    elif len(shape_a) == 4:
        # 4D @ 2D: (batch, heads, seq, dim) @ (dim, out_dim) -> (batch, heads, seq, out_dim)
        result = []
        for i in range(len(a)):  # batch维度
            batch_result = []
            for j in range(len(a[i])):  # heads维度
                sub_result = _matrix_matrix_multiply(a[i][j], b)
                batch_result.append(sub_result)
            result.append(batch_result)
        return result
    
    else:
        # 更高维度，递归处理
        result = []
        for i in range(len(a)):
            sub_result = _high_dim_2d_matmul(a[i], b, shape_a[1:], shape_b)
            result.append(sub_result)
        return result

def _2d_high_dim_matmul(a, b, shape_a, shape_b):
    """2D矩阵与高维数组相乘，保持高维结构"""
    if len(shape_b) == 3:
        # 2D @ 3D: (dim1, dim2) @ (batch, dim2, out_dim) -> (batch, dim1, out_dim)
        result = []
        for i in range(len(b)):
            sub_result = _matrix_matrix_multiply(a, b[i])
            result.append(sub_result)
        return result
    
    elif len(shape_b) == 4:
        # 2D @ 4D: (dim1, dim2) @ (batch, heads, dim2, out_dim) -> (batch, heads, dim1, out_dim)
        result = []
        for i in range(len(b)):  # batch维度
            batch_result = []
            for j in range(len(b[i])):  # heads维度
                sub_result = _matrix_matrix_multiply(a, b[i][j])
                batch_result.append(sub_result)
            result.append(batch_result)
        return result
    
    else:
        # 更高维度，递归处理
        result = []
        for i in range(len(b)):
            sub_result = _2d_high_dim_matmul(a, b[i], shape_a, shape_b[1:])
            result.append(sub_result)
        return result

# 便利函数
def matrix_multiply(a, b):
    """简化的矩阵乘法函数名"""
    return perfect_matmul(a, b)

def dot_product(a, b):
    """计算点积的便利函数"""
    return perfect_matmul(a, b)

def safe_matmul(a, b, default_value=0.0):
    """安全的矩阵乘法，失败时返回默认值"""
    try:
        return perfect_matmul(a, b)
    except Exception as e:
        print(f"Warning: matmul failed ({e}), returning default value")
        return default_value

def batch_matmul(matrices_a, matrices_b):
    """批量矩阵乘法"""
    if len(matrices_a) != len(matrices_b):
        raise ValueError("Batch sizes must match")
    
    results = []
    for i in range(len(matrices_a)):
        result = perfect_matmul(matrices_a[i], matrices_b[i])
        results.append(result)
    
    return results

# 测试函数
def test_strong_matmul():
    """测试函数，验证strong_matmul的正确性"""
    print("Testing strong_matmul.py...")
    
    # 测试1：向量点积
    vec1 = [1, 2, 3]
    vec2 = [4, 5, 6]
    result1 = perfect_matmul(vec1, vec2)
    expected1 = 32  # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assert abs(result1 - expected1) < 1e-10, f"Test 1 failed: {result1} != {expected1}"
    
    # 测试2：矩阵与向量相乘
    matrix = [[1, 2], [3, 4]]
    vector = [5, 6]
    result2 = perfect_matmul(matrix, vector)
    expected2 = [17, 39]  # [1*5+2*6, 3*5+4*6] = [17, 39]
    for r, e in zip(result2, expected2):
        assert abs(r - e) < 1e-10, f"Test 2 failed: {result2} != {expected2}"
    
    # 测试3：向量与矩阵相乘
    result3 = perfect_matmul(vector, matrix)
    expected3 = [23, 34]  # [5*1+6*3, 5*2+6*4] = [23, 34]
    for r, e in zip(result3, expected3):
        assert abs(r - e) < 1e-10, f"Test 3 failed: {result3} != {expected3}"
    
    # 测试4：标准矩阵乘法
    matrix_a = [[1, 2], [3, 4]]
    matrix_b = [[5, 6], [7, 8]]
    result4 = perfect_matmul(matrix_a, matrix_b)
    expected4 = [[19, 22], [43, 50]]  # [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
    for i in range(len(result4)):
        for j in range(len(result4[i])):
            assert abs(result4[i][j] - expected4[i][j]) < 1e-10, \
                f"Test 4 failed: {result4} != {expected4}"
    
    # 测试5：标量乘法
    result5 = perfect_matmul(2, matrix_a)
    expected5 = [[2, 4], [6, 8]]
    for i in range(len(result5)):
        for j in range(len(result5[i])):
            assert abs(result5[i][j] - expected5[i][j]) < 1e-10, \
                f"Test 5 failed: {result5} != {expected5}"
    
    # 测试6：单元素矩阵
    single = [[5]]
    result6 = perfect_matmul(single, single)
    expected6 = [[25]]
    assert abs(result6[0][0] - expected6[0][0]) < 1e-10, \
        f"Test 6 failed: {result6} != {expected6}"
    
    print("All tests passed! ✅")

# 直接替换np.matmul的函数
def replace_np_matmul(a, b):
    """
    直接替换np.matmul调用的函数
    """
    return perfect_matmul(a, b)

def _create_compatible_array(data):
    """创建与operations_T.py兼容的数组对象"""
    try:
        return arrays.Array(data)
    except:
        # 如果arrays.Array不可用，创建一个简单的包装类
        class CompatibleArray:
            def __init__(self, data):
                self.data = data
        return CompatibleArray(data)

if __name__ == "__main__":
    test_strong_matmul() 