import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import scipy.io as sio
import torchvision.transforms as transforms
from pathlib import Path


class MatDataset(Dataset):
    def __init__(self, root_dir, crop_size=256):
        """
        初始化MAT文件数据集（适配Real Image Noise Dataset）
        Args:
            root_dir: 数据集根目录路径
            crop_size: 裁剪尺寸，默认256
        """
        self.root_dir = Path(root_dir)
        self.crop_size = crop_size
        self.mat_files = self._find_mat_files(self.root_dir)
        
        # 数据预处理转换
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        # 打印数据集统计信息
        print(f"Found {len(self.mat_files)} .mat files in {root_dir}")
        self._print_dataset_stats()

    def _find_mat_files(self, root_dir):
        """递归查找所有.mat文件"""
        mat_files = []
        for mat_file in root_dir.rglob("*.mat"):
            mat_files.append(str(mat_file))
        return sorted(mat_files)  # 确保顺序一致

    def _print_dataset_stats(self):
        """打印数据集统计信息（按设备/ISO分组）"""
        from collections import defaultdict
        stats = defaultdict(lambda: defaultdict(int))

        for path in self.mat_files:
            # 提取设备名和ISO设置
            parts = Path(path).parts
            if len(parts) >= 3:
                device = parts[-3]  # 如 "Nikon_D800"
                iso = parts[-2]  # 如 "ISO_3200"
                stats[device][iso] += 1

        print("\nDataset statistics:")
        for device, isos in stats.items():
            print(f"  {device}:")
            for iso, count in isos.items():
                print(f"    {iso}: {count} files")

    def __len__(self):
        return len(self.mat_files)

    def __getitem__(self, idx):
        mat_path = self.mat_files[idx]

        try:
            # 加载.mat文件
            data = sio.loadmat(mat_path)

            # 提取噪声图像和均值图像
            noisy_img = data['img_noisy']  # HxWxC, uint8
            mean_img = data['img_mean']  # HxWxC, uint8
            
            # 可选：提取掩码和协方差矩阵（用于高级应用）
            mask = data.get('img_mask', None)  # HxW, logical
            cov_matrix = data.get('img_cov', None)  # HxWx6, single

            # 转换为RGB PIL图像
            noisy_img = Image.fromarray(noisy_img.astype('uint8'), 'RGB')
            mean_img = Image.fromarray(mean_img.astype('uint8'), 'RGB')

            # 应用转换并归一化到[0,1]
            noisy_img = self.transform(noisy_img)  # 已经是[0,1]范围
            mean_img = self.transform(mean_img)    # 已经是[0,1]范围

            # 裁剪到指定尺寸
            noisy_img = noisy_img[:, :self.crop_size, :self.crop_size]
            mean_img = mean_img[:, :self.crop_size, :self.crop_size]

            return noisy_img, mean_img

        except Exception as e:
            print(f"Error loading {mat_path}: {str(e)}")
            # 返回空数据
            return torch.zeros(3, self.crop_size, self.crop_size), torch.zeros(3, self.crop_size, self.crop_size)


# 使用示例
if __name__ == "__main__":
    # 使用示例
    dataset = MatDataset(root_dir="./real_image_noise_dataset", crop_size=256)
    print(f"Total samples: {len(dataset)}")

    # 测试加载第一个样本
    noisy, mean = dataset[0]
    print(f"Noisy image shape: {noisy.shape}")
    print(f"Mean image shape: {mean.shape}")
    print(f"Value range: [{noisy.min():.3f}, {noisy.max():.3f}]")
