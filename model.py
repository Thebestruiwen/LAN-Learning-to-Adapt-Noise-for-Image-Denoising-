import importlib
import sys
from pathlib import Path
import os
import torch
import paddle
dir_name = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本文件的绝对路径目录
sys.path.append(os.path.join(dir_name,'Uformer-main/utils'))
sys.path.append(os.path.join(dir_name,'Uformer-main'))# 将数据集目录添加到系统路径+
from UFmodel import Uformer, UNet
import utils
from collections import OrderedDict


def get_model_Restormer():
    path = 'Restormer.basicsr.models.archs.restormer_arch'
    module_spec = importlib.util.spec_from_file_location(path, str(Path().joinpath(*path.split('.'))) + '.py')
    print("path", path)
    print("path____", str(Path().joinpath(*path.split('.'))) + '.py')

    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    model = module.Restormer(LayerNorm_type='BiasFree').cuda()
    checkpoint = torch.load("./checkpoint/real_denoising.pth")["params"]
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model


def get_model_dncnn(
        model_path="./checkpoint/model.pdiparams",
        channels=3,
        num_of_layers=17,
        device='gpu'
):
    """加载 DnCNN 模型 (PaddlePaddle 实现)"""
    # 设置正确的模块路径
    dncnn_base_path = "DnCNN_paddle-main"
    module_name = "models"  # 假设模型定义在 model.py 文件中

    # 构建模块的完整路径
    module_path = os.path.join(dncnn_base_path, f"{module_name}.py")

    # 动态导入模块
    module_spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    # 创建模型实例
    model = module.DnCNN(channels=channels, num_of_layers=num_of_layers)

    # 设备设置
    if paddle.device.is_compiled_with_cuda() and 'gpu' in device:
        paddle.set_device('gpu:0')
    else:
        paddle.set_device('cpu')

    # 加载预训练权重
    if model_path:
        state_dict = paddle.load(model_path)
        model.set_state_dict(state_dict)

    # 设置为评估模式
    model.eval()

    return model


def get_model_Uformer(
        img_size=128,
        embed_dim=32,
        checkpoint_path="/home/yanyutai/Uformer-main/logs/model_best.pth",
        device="cuda"
):
    """加载Uformer模型并检查权重加载状态"""
    # 动态导入Uformer类
    model_restoration = Uformer(img_size=img_size, embed_dim=embed_dim, win_size=8, token_projection='linear',
                                token_mlp='leff',
                                depths=[1, 2, 8, 8, 2, 8, 8, 2, 1], modulator=True, dd_in=3)
    utils.load_checkpoint(model_restoration, checkpoint_path)
    model_restoration.eval()
    return model_restoration




