import os
import sys
import torch

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

# 加载模型
model = MiniMindForCausalLM(MiniMindConfig(
    hidden_size=768,
    num_hidden_layers=16,
    use_moe=False
))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('../out/full_sft_768.pth', map_location=device), strict=True)
model.eval()  # 设置为评估模式

# 根据配置文件创建正确的示例输入
batch_size = 1
seq_length = 8192  # 使用较短的序列长度进行追踪，实际max_seq_len是8192
vocab_size = 6400  # 从配置文件获取

# 创建示例输入 - 语言模型需要token IDs (整数张量)
example_input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), dtype=torch.long)

print(f"输入维度: {example_input_ids.shape}")
print(f"词汇表大小: {vocab_size}")
print(f"序列长度: {seq_length}")

# 创建一个包装器函数，只返回 logits
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, input_ids):
        # 只返回 logits，避免复杂的输出对象
        output = self.model(input_ids)
        return output.logits if hasattr(output, 'logits') else output

# 创建包装器
wrapped_model = ModelWrapper(model)
wrapped_model.eval()

try:
    # 测试包装后的模型
    with torch.no_grad():
        output = wrapped_model(example_input_ids)
        print(f"包装模型输出维度: {output.shape}")
        
        # 追踪包装后的模型
        print("开始追踪包装后的模型...")
        traced_model = torch.jit.trace(wrapped_model, example_input_ids)
        
        # 保存TorchScript模型
        traced_model.save('model_traced.pt')
        print("追踪模型保存完成: model_traced.pt")
        
        # 验证追踪后的模型
        print("验证追踪后的模型...")
        traced_output = traced_model(example_input_ids)
        print(f"追踪模型输出维度: {traced_output.shape}")
        print(f"输出差异: {torch.max(torch.abs(output - traced_output)).item():.6f}")
        
except Exception as e:
    print(f"包装模型追踪失败: {e}")

# 尝试 script 方法 - 但通常会失败，因为模型太复杂
try:
    print("\n尝试脚本化包装后的模型...")
    scripted_model = torch.jit.script(wrapped_model)
    scripted_model.save('model_script.pt')
    print("脚本化模型保存完成: model_script.pt")
except Exception as e:
    print(f"脚本化失败: {e}")
    print("这是预期的，因为模型包含复杂的操作")

# 额外尝试：禁用 flash attention
print("\n尝试禁用 flash attention 后追踪...")
try:
    # 临时禁用 flash attention
    original_flash = model.model.layers[0].self_attn.flash if hasattr(model.model.layers[0].self_attn, 'flash') else None
    
    # 遍历所有层，禁用 flash attention
    for layer in model.model.layers:
        if hasattr(layer.self_attn, 'flash'):
            layer.self_attn.flash = False
    
    # 重新创建包装器
    wrapped_model_no_flash = ModelWrapper(model)
    wrapped_model_no_flash.eval()
    
    with torch.no_grad():
        traced_model_no_flash = torch.jit.trace(wrapped_model_no_flash, example_input_ids)
        traced_model_no_flash.save('model_traced_no_flash.pt')
        print("禁用 flash attention 的追踪模型保存完成: model_traced_no_flash.pt")
        
        # 验证
        traced_output_no_flash = traced_model_no_flash(example_input_ids)
        print(f"无 flash attention 模型输出维度: {traced_output_no_flash.shape}")
    
    # 恢复 flash attention 设置
    if original_flash is not None:
        for layer in model.model.layers:
            if hasattr(layer.self_attn, 'flash'):
                layer.self_attn.flash = original_flash
                
except Exception as e:
    print(f"禁用 flash attention 后仍然失败: {e}")

print("转换尝试完成！")