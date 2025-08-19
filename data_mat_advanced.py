import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import scipy.io as sio
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np


class MatDatasetAdvanced(Dataset):
    def __init__(self, root_dir, crop_size=256, use_covariance=False, use_mask=False):
        """
        高级MAT文件数据集（适配Real Image Noise Dataset）
        Args:
            root_dir: 数据集根目录路径
            crop_size: 裁剪尺寸，默认256
            use_covariance: 是否使用协方差矩阵信息
            use_mask: 是否使用掩码信息
        """
        self.root_dir = Path(root_dir)
        self.crop_size = crop_size
        self.use_covariance = use_covariance
        self.use_mask = use_mask
        self.mat_files = self._find_mat_files(self.root_dir)
        
        # 数据预处理转换
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        # 打印数据集统计信息
        print(f"Found {len(self.mat_files)} .mat files in {root_dir}")
        print(f"Using covariance: {use_covariance}")
        print(f"Using mask: {use_mask}")
        self._print_dataset_stats()

    def _find_mat_files(self, root_dir):
        """递归查找所有.mat文件"""
        mat_files = []
        for mat_file in root_dir.rglob("*.mat"):
            mat_files.append(str(mat_file))
        return sorted(mat_files)

    def _print_dataset_stats(self):
        """打印数据集统计信息（按设备/ISO分组）"""
        from collections import defaultdict
        stats = defaultdict(lambda: defaultdict(int))

        for path in self.mat_files:
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

    def _process_covariance(self, cov_matrix):
        """处理协方差矩阵，转换为可用的特征"""
        if cov_matrix is None:
            return None
        
        # 协方差矩阵形状: HxWx6 (R*R, G*G, B*B, R*G, R*B, G*B)
        # 转换为张量并归一化
        cov_tensor = torch.from_numpy(cov_matrix.astype(np.float32))
        
        # 归一化到[0,1]范围
        cov_tensor = cov_tensor / (255.0 * 255.0)  # 因为原始范围是0-255²
        
        return cov_tensor

    def _process_mask(self, mask):
        """处理掩码"""
        if mask is None:
            return None
        
        # 转换为张量
        mask_tensor = torch.from_numpy(mask.astype(np.float32))
        return mask_tensor

    def __len__(self):
        return len(self.mat_files)

    def __getitem__(self, idx):
        mat_path = self.mat_files[idx]

        try:
            # 加载.mat文件
            data = sio.loadmat(mat_path)

            # 提取基本数据
            noisy_img = data['img_noisy']  # HxWxC, uint8
            mean_img = data['img_mean']  # HxWxC, uint8
            
            # 提取可选数据
            mask = data.get('img_mask', None) if self.use_mask else None
            cov_matrix = data.get('img_cov', None) if self.use_covariance else None

            # 转换为RGB PIL图像
            noisy_img = Image.fromarray(noisy_img.astype('uint8'), 'RGB')
            mean_img = Image.fromarray(mean_img.astype('uint8'), 'RGB')

            # 应用转换并归一化到[0,1]
            noisy_img = self.transform(noisy_img)
            mean_img = self.transform(mean_img)

            # 裁剪到指定尺寸
            noisy_img = noisy_img[:, :self.crop_size, :self.crop_size]
            mean_img = mean_img[:, :self.crop_size, :self.crop_size]

            # 处理协方差矩阵
            if self.use_covariance and cov_matrix is not None:
                cov_tensor = self._process_covariance(cov_matrix)
                cov_tensor = cov_tensor[:, :self.crop_size, :self.crop_size]
            else:
                cov_tensor = None

            # 处理掩码
            if self.use_mask and mask is not None:
                mask_tensor = self._process_mask(mask)
                mask_tensor = mask_tensor[:self.crop_size, :self.crop_size]
            else:
                mask_tensor = None

            # 返回数据
            result = {'noisy': noisy_img, 'mean': mean_img}
            if cov_tensor is not None:
                result['covariance'] = cov_tensor
            if mask_tensor is not None:
                result['mask'] = mask_tensor
            
            return result

        except Exception as e:
            print(f"Error loading {mat_path}: {str(e)}")
            # 返回空数据
            result = {
                'noisy': torch.zeros(3, self.crop_size, self.crop_size),
                'mean': torch.zeros(3, self.crop_size, self.crop_size)
            }
            if self.use_covariance:
                result['covariance'] = torch.zeros(6, self.crop_size, self.crop_size)
            if self.use_mask:
                result['mask'] = torch.zeros(self.crop_size, self.crop_size)
            return result


# 兼容性包装器，保持与原始Dataset接口一致
class MatDataset(Dataset):
    def __init__(self, root_dir, crop_size=256):
        """
        兼容性包装器，保持与data.py相同的接口
        """
        self.dataset = MatDatasetAdvanced(root_dir, crop_size, use_covariance=False, use_mask=False)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        return data['noisy'], data['mean']


# 使用示例
if __name__ == "__main__":
    # 基本使用（与data.py兼容）
    print("=== Basic Usage ===")
    dataset = MatDataset(root_dir="./real_image_noise_dataset", crop_size=256)
    noisy, mean = dataset[0]
    print(f"Noisy image shape: {noisy.shape}")
    print(f"Mean image shape: {mean.shape}")
    print(f"Value range: [{noisy.min():.3f}, {noisy.max():.3f}]")
    
    # 高级使用（包含协方差矩阵和掩码）
    print("\n=== Advanced Usage ===")
    dataset_adv = MatDatasetAdvanced(
        root_dir="./real_image_noise_dataset", 
        crop_size=256,
        use_covariance=True,
        use_mask=True
    )
    data = dataset_adv[0]
    print(f"Keys: {list(data.keys())}")
    if 'covariance' in data:
        print(f"Covariance shape: {data['covariance'].shape}")
    if 'mask' in data:
        print(f"Mask shape: {data['mask'].shape}") 