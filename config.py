# 配置文件 - 管理模型文件路径
import os

# 模型文件路径配置
# 请根据你的实际文件结构修改这些路径

# 方案1: 如果模型文件在上级目录的DnCNN_paddle-main文件夹中
MODEL_PATH = os.path.join(os.path.dirname(__file__), "DnCNN_paddle-main")

# 方案2: 如果模型文件在绝对路径中（请替换为你的实际路径）
# MODEL_PATH = "D:/Study/LAN/DnCNN_paddle-main"

# 方案3: 如果模型文件在当前目录的子文件夹中
# MODEL_PATH = os.path.join(os.path.dirname(__file__), "DnCNN_paddle-main")

# 权重文件路径
WEIGHT_PATH = os.path.join(MODEL_PATH, "logs", "net.pdparams")

# 检查路径是否存在
def check_paths():
    """检查配置的路径是否存在"""
    if not os.path.exists(MODEL_PATH):
        print(f"警告: 模型路径不存在: {MODEL_PATH}")
        return False
    
    if not os.path.exists(WEIGHT_PATH):
        print(f"警告: 权重文件不存在: {WEIGHT_PATH}")
        return False
    
    print(f"模型路径: {MODEL_PATH}")
    print(f"权重路径: {WEIGHT_PATH}")
    return True

if __name__ == "__main__":
    check_paths() 