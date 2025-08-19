import scipy.io as sio
import numpy as np
from PIL import Image
import os
import argparse


def _find_mat_files(root_dir):
    """递归查找所有.mat文件"""
    mat_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.mat'):
                full_path = os.path.join(dirpath, filename)
                mat_files.append(full_path)
    return mat_files

def mat_to_png(mat_path, output_dir="./real_image_noise_dataset", variables=None, verbose=True):
    """
    将 .mat 文件中的图像数据保存为 PNG 格式

    参数:
        mat_path (str): .mat 文件路径
        output_dir (str): 输出目录（默认为 .mat 文件所在目录）
        variables (list): 要转换的变量名列表（默认为所有图像变量）
        verbose (bool): 是否显示详细信息
    """
    # 确保文件存在
    if not os.path.isfile(mat_path):
        print(f"错误: 文件不存在 - {mat_path}")
        return

    # 创建输出目录
    if output_dir is None:
        output_dir = os.path.dirname(mat_path)
    os.makedirs(output_dir, exist_ok=True)

    # 读取 .mat 文件
    try:
        mat_data = sio.loadmat(mat_path)
        if verbose:
            print(f"成功加载: {mat_path}")
            print("文件包含的变量:", list(mat_data.keys()))
    except Exception as e:
        print(f"读取 .mat 文件失败: {e}")
        return

    # 确定要转换的变量
    image_vars = []
    if variables:
        # 使用用户指定的变量
        image_vars = [var for var in variables if var in mat_data]
    else:
        # 自动检测图像变量
        for var_name, var_data in mat_data.items():
            # 排除 MATLAB 系统变量（以 __ 开头）
            if var_name.startswith('__'):
                continue

            # 检查是否为图像数据
            if isinstance(var_data, np.ndarray):
                # 检查维度：2D(灰度) 或 3D(彩色) 或 4D(批量)
                if var_data.ndim in (2, 3, 4):
                    # 检查数据类型是否适合图像
                    if var_data.dtype in (np.uint8, np.uint16, np.float32, np.float64):
                        image_vars.append(var_name)

    if not image_vars:
        print("警告: 未找到可识别的图像变量")
        return

    # 处理每个图像变量
    for var_name in image_vars:
        image_data = mat_data[var_name]

        # 处理不同维度的图像数据
        if image_data.ndim == 2:  # 灰度图像
            images = [image_data]
        elif image_data.ndim == 3:  # 彩色或多帧图像
            if image_data.shape[2] == 3:  # 彩色图像 (H, W, 3)
                images = [image_data]
            else:  # 多帧图像 (H, W, N)
                images = [image_data[:, :, i] for i in range(image_data.shape[2])]
        elif image_data.ndim == 4:  # 批量彩色图像 (N, H, W, C)
            images = [image_data[i, :, :, :] for i in range(image_data.shape[0])]
        else:
            print(f"跳过不支持的维度: {var_name} ({image_data.shape})")
            continue

        # 保存每张图像
        for i, img in enumerate(images):
            # 确定文件名
            base_name = os.path.splitext(os.path.basename(mat_path))[0]
            output_path = output_dir+f"/{var_name}"
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            try:
                # 处理不同数据类型
                if img.dtype == np.uint8:
                    # 直接保存 uint8
                    pil_img = Image.fromarray(img)
                elif img.dtype == np.uint16:
                    # 缩放 uint16 到 uint8 (0-65535 -> 0-255)
                    scaled_img = (img / 256).astype(np.uint8)
                    pil_img = Image.fromarray(scaled_img)
                elif img.dtype in (np.float32, np.float64):
                    # 处理浮点图像数据
                    if img.max() <= 1.0:  # 假设是 0-1 范围
                        scaled_img = (img * 255).astype(np.uint8)
                    else:  # 假设是 0-255 范围
                        scaled_img = np.clip(img, 0, 255).astype(np.uint8)
                    pil_img = Image.fromarray(scaled_img)
                else:
                    print(f"跳过不支持的数据类型: {var_name} ({img.dtype})")
                    continue

                # 保存为 PNG

                pil_img.save(os.path.join(output_path, f"{base_name}_{i + 1}.png"))
                if verbose:
                    print(f"已保存: {output_path}/{base_name}_{i + 1}.png")

            except Exception as e:
                print(f"保存 {var_name} 失败: {e}")


if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='将 .mat 文件中的图像转换为 PNG 格式')
    parser.add_argument('--mat_dir', type=str, help='含有 .mat 文件的文件夹路径')
    parser.add_argument('-o', '--output', type=str, help='输出目录')
    parser.add_argument('-v', '--variables', nargs='+', help='指定要转换的变量名')
    parser.add_argument('-q', '--quiet', action='store_true', help='静默模式，不显示详细信息')
    args = parser.parse_args()

    mat_list=_find_mat_files(args.mat_dir)
    for mat in  mat_list:
        # 调用转换函数
        mat_to_png(
            mat_path=mat,
            variables=args.variables,
            verbose=not args.quiet
        )
