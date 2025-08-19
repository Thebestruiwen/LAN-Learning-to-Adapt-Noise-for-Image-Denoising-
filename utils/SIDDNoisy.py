import os
import numpy as np
import cv2
import scipy.io as sio
from tqdm import tqdm


def calculate_average_noise(gt_folder, noisy_folder, output_mat='average_noise.mat'):
    """
    计算GT和NOISY图像之间的平均噪声

    参数:
    gt_folder: 存放干净图像的文件夹路径
    noisy_folder: 存放噪声图像的文件夹路径
    output_mat: 输出的MAT文件名
    """
    # 获取图像文件列表
    gt_files = sorted(
        [f for f in os.listdir(gt_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))])
    noisy_files = sorted(
        [f for f in os.listdir(noisy_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))])

    # 确保文件一一对应
    if gt_files != noisy_files:
        # 找出共同的文件名
        common_files = sorted(set(gt_files) & set(noisy_files))
        if not common_files:
            raise ValueError("未找到匹配的图像文件")
        print(f"警告: 部分文件不匹配，使用 {len(common_files)} 对图像")
    else:
        common_files = gt_files

    # 初始化累加器和计数器
    total_noise = None
    valid_count = 0

    # 遍历所有图像对
    for filename in tqdm(common_files, desc="处理图像"):
        # 读取图像

        gt_img = cv2.imread(os.path.join(gt_folder, filename))
        noisy_img = cv2.imread(os.path.join(noisy_folder, filename))
        gt_img=gt_img[0:255,0:255,:]
        noisy_img = noisy_img[0:255, 0:255, :]
        # 检查图像是否成功读取
        if gt_img is None or noisy_img is None:
            print(f"警告: 无法读取 {filename}，跳过")
            continue

        # 转换为浮点数并归一化
        gt_img = gt_img.astype(np.float32) / 255.0
        noisy_img = noisy_img.astype(np.float32) / 255.0

        # 检查图像尺寸是否匹配
        if gt_img.shape != noisy_img.shape:
            print(f"警告: {filename} 尺寸不匹配 ({gt_img.shape} vs {noisy_img.shape})，跳过")
            continue

        # 计算噪声
        noise = noisy_img - gt_img

        # 初始化累加器
        if total_noise is None:
            total_noise = np.zeros_like(noise)

        # 累加噪声
        total_noise += noise
        valid_count += 1

    if valid_count == 0:
        raise RuntimeError("没有有效的图像对进行处理")

    # 计算平均噪声
    average_noise = total_noise / valid_count

    # 保存为MAT文件
    sio.savemat(output_mat, {'average_noise': average_noise})

    # 打印统计信息
    print(f"\n处理完成: 共处理 {valid_count} 对图像")
    print(f"平均噪声统计:")
    if len(average_noise.shape) == 3:  # 彩色图像
        for i, channel in enumerate(['B', 'G', 'R']):
            ch_noise = average_noise[:, :, i]
            print(f"  {channel}通道 - 均值: {np.mean(ch_noise):.6f}, 标准差: {np.std(ch_noise):.6f}")
    else:  # 灰度图像
        print(f"  均值: {np.mean(average_noise):.6f}, 标准差: {np.std(average_noise):.6f}")

    print(f"结果已保存到: {output_mat}")
    return average_noise


# 使用示例
if __name__ == "__main__":
    # 设置文件夹路径
    gt_folder = "/home/yanyutai/trainData/SRGB/SIDD_Medium_Srgb/Data/GT"  # 替换为你的GT文件夹路径
    noisy_folder = "/home/yanyutai/trainData/SRGB/SIDD_Medium_Srgb/Data/NOISY"  # 替换为你的NOISY文件夹路径
    output_file = "SIDD_average_noise.mat"  # 输出文件名

    # 计算平均噪声
    calculate_average_noise(gt_folder, noisy_folder, output_file)