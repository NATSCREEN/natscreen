import sys
import os
from datetime import datetime
from pure_data_loader import prepare_pure_training_dataset  # 使用100%纯Python数据加载器
from Transformer import MolecularTransformer
from operations_T import sigmoid
from autograd_T import no_grad
from optimizer_T import Adam
from new_focal_loss import FocalLoss
from tensor_T import Tensor  # 确保使用自定义Tensor

class TeeOutput:
    """同时输出到终端和文件的类"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log_file = open(filename, 'w', encoding='utf-8')
        
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # 立即写入文件
        
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
        
    def close(self):
        self.log_file.close()

def setup_logging():
    """设置日志输出到文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"training_log_latest.txt"
    
    # 重定向stdout到同时输出到终端和文件
    tee = TeeOutput(log_filename)
    sys.stdout = tee
    
    print(f"🎯 训练日志将保存到: {log_filename}")
    print(f"📅 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    return tee, log_filename

def initialize_network_and_optimizer(optimal_parameters, activation='gelu'):
    """Initialize network and optimizer with optimal parameters"""
    print(f"🎯 创建网络 - 激活函数: {activation.upper()}")
    network = MolecularTransformer(
        input_features=2048,#2048
        output_features=1, 
        embedding_size=128,#512
        layer_count=optimal_parameters['transformer_depth'],
        head_count=optimal_parameters['attention_heads'], 
        hidden_size=optimal_parameters['hidden_dimension'], 
        dropout_rate=0.1,
        activation=activation  # 新增激活函数参数
    )
    optimizer = Adam(network.parameters(), lr=optimal_parameters['learning_rate'])
    return network, optimizer

def conduct_individual_training(network, data_handler, optimizer, network_index, model_version, unique_id):
    """Train individual network"""
    print(f"=== 开始训练网络 {unique_id+1} ===")
    
    criterion = FocalLoss(alpha=0.25, gamma=2.0, high_pred_penalty=5.0, reduction='mean')  # 降低惩罚系数
    
    for epoch in range(1):  # 训练2个epoch
        epoch_losses = []
        batch_count = 0
        high_pred_false_count = 0
        very_high_pred_false_count = 0 
        extreme_high_pred_false_count = 0
        all_predictions = []
        
        # 训练每个批次
        for batch_idx, (features, labels) in enumerate(data_handler):
            batch_count += 1
            
            # 确保输入数据是我们的自定义Tensor
            if not isinstance(features, Tensor):
                features = Tensor(features, requires_grad=False)
            if not isinstance(labels, Tensor):
                labels = Tensor(labels, requires_grad=False)
            
            # 前向传播
            outputs = network(features)  
            
            # 确保标签维度正确
            if labels.data.ndim > 1:
                labels = Tensor(labels.data.squeeze(), requires_grad=False)
            
            # 计算损失
            loss = criterion(outputs.squeeze(), labels)
            
            # 异常损失检测
            loss_value = loss.data.item() if hasattr(loss.data, 'item') else float(loss.data)
            if loss_value > 5.0: 
                print(f"⚠️ 异常高损失 {loss_value:.4f}，跳过参数更新")
                continue
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss_value)
            
            # 计算预测准确性统计
            with no_grad():
                predictions = sigmoid(outputs.squeeze())
                all_predictions.extend(predictions.data.flatten().tolist())
                
                pred_data = predictions.data.flatten()
                label_data = labels.data.flatten()
                
                for pred, label in zip(pred_data, label_data):
                    if pred > 0.9 and label < 0.5:
                        high_pred_false_count += 1
                    if pred > 0.95 and label < 0.5:
                        very_high_pred_false_count += 1
                    if pred > 0.98 and label < 0.5:
                        extreme_high_pred_false_count += 1
            
            # 每10个批次打印损失
            if batch_count % 10 == 0:
                print(f"    批次 {batch_count}, 损失: {loss_value:.4f}")
        
        # Epoch总结
        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
        print(f"  Epoch [{epoch+1}], 平均损失: {avg_loss:.4f}, "
              f"High Pred False: {high_pred_false_count}, "
              f"Very High Pred False: {very_high_pred_false_count}, "
              f"Extreme High Pred False: {extreme_high_pred_false_count}")
        
        # 打印预测值范围
        if all_predictions:
            min_pred = min(all_predictions)
            max_pred = max(all_predictions)
            print(f"  [DEBUG] Epoch {epoch+1} 预测值范围: [{min_pred:.4f}, {max_pred:.4f}]")
    
    # 保存模型
    model_filename = f'model_{unique_id+1}.dict'
    USE_PICKLE_FORMAT = True  # 使用pickle格式保存
    print(f"[SAVE] 保存格式: {'pickle(快速)' if USE_PICKLE_FORMAT else 'model_serializer(纯Python)'}")
    
    if USE_PICKLE_FORMAT:
        print(f"[SAVE] 使用pickle格式保存模型到: {model_filename}")
        import pickle
        save_data = {
            'model_parameters': network.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': 2
        }
        with open(model_filename, 'wb') as f:
            pickle.dump(save_data, f)
    else:
        print(f"[SAVE] 使用model_serializer格式保存模型到: {model_filename}")
        from utils2 import store_model
        store_model(network, optimizer, 2, model_filename, use_pickle=False)
    
    print(f"✅ 网络 {unique_id+1} 训练完成，模型已保存到 {model_filename}")
    return network

def training():
    """Main function for network training - Sequential version"""
    # 设置日志输出
    tee, log_filename = setup_logging()
    
    # 设置固定随机种子，确保结果可重现
    import pure_random
    FIXED_SEED = 42  # 使用固定种子
    pure_random.seed(FIXED_SEED)
    print(f"🎲 使用固定随机种子: {FIXED_SEED} (确保结果可重现)")
    print("=" * 60)
    
    try:
        from command_line_parser import CommandLineProcessor  
        
        config_parser = CommandLineProcessor(description='Molecular property prediction')
        config_parser.add_argument('--num_networks', type=int, default=1, 
                                 help='Quantity of networks to train')
        config_parser.add_argument('--activation', type=str, default='gelu', choices=['relu', 'gelu'],
                                 help='选择激活函数: relu 或 gelu (默认: gelu)')
        parameters = config_parser.parse_args()

        print(f'🚀 开始顺序训练 {parameters.num_networks} 个网络')
        print(f'⚙️  激活函数设置: {parameters.activation.upper()}')
        
        if parameters.activation.lower() == 'gelu':
            print("✨ GELU激活函数通常在Transformer架构中表现更佳")
        else:
            print("⚡ ReLU激活函数计算速度更快，资源消耗更少")

        # 使用100%纯Python数据加载器
        print("正在准备数据集...")
        #batch_size=10 #32
        data_handler = prepare_pure_training_dataset('training_dataset.csv', fingerprint_type='Morgan', 
                                                   batch_size=32, shuffle=False, balance_data=True)
        print("✅ 数据集准备完成")
        
        # 优化参数配置
        optimal_parameters = {
            'learning_rate': 0.001,
            'transformer_depth': 2,   #改成4（原来6）
            'attention_heads': 2, 
            'hidden_dimension': 64   #改成512（原来2048）
        }

        # 初始化和训练网络
        print("正在初始化网络...")
        trained_networks = []
        
        for network_idx in range(parameters.num_networks):
            print(f'\n--- 初始化网络 {network_idx+1} ---')
            network, optimizer = initialize_network_and_optimizer(optimal_parameters, parameters.activation)
            
            # 训练此网络
            trained_network = conduct_individual_training(
                network, data_handler, optimizer, 0, 2, network_idx
            )
            trained_networks.append(trained_network)

        print(f"\n🎉 所有训练完成！成功训练了 {len(trained_networks)} 个网络")
        print(f"模型文件: {[f'model_{i+1}.dict' for i in range(len(trained_networks))]}")
        print(f"🔧 使用的激活函数: {parameters.activation.upper()}")
        
        print("=" * 60)
        print(f"📄 训练日志已保存到: {log_filename}")
        print(f"🕐 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"❌ 训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # 恢复原始stdout并关闭日志文件
        sys.stdout = tee.terminal
        tee.close()
        print(f"✅ 日志已保存到文件: {log_filename}")

if __name__ == "__main__":
    training()
