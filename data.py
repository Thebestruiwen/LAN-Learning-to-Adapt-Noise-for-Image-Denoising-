import torch
from pathlib import Path
import torchvision


#用于直接从数据集接受图片输入
class Dataset(torch.utils.data.Dataset):
    def __init__(self, lq_dir, gt_dir, crop_size=256):
        self.lq_dir = Path(lq_dir)
        self.gt_dir = Path(gt_dir)
        self.crop_size = crop_size

        # 支持多种图像格式
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff','*.JPG']

        # 获取所有匹配的图像文件路径
        self.lq_paths = []
        for ext in image_extensions:
            self.lq_paths.extend(sorted(self.lq_dir.glob(ext)))

        self.gt_paths = []
        for ext in image_extensions:
            self.gt_paths.extend(sorted(self.gt_dir.glob(ext)))

        # 确保文件数量匹配
        assert len(self.lq_paths) == len(self.gt_paths), \
            f"低质量图像({len(self.lq_paths)})和高质量图像({len(self.gt_paths)})数量不匹配"

        # 确保文件名一一对应
        lq_names = [p.stem for p in self.lq_paths]
        gt_names = [p.stem for p in self.gt_paths]
        assert set(lq_names) == set(gt_names), "文件名不匹配"

        # 按文件名排序确保对应关系
        self.lq_paths = sorted(self.lq_paths, key=lambda p: p.stem)
        self.gt_paths = sorted(self.gt_paths, key=lambda p: p.stem)
    
    def __len__(self):
        return len(self.lq_paths)
    
    def __getitem__(self, idx):
        lq_name = self.lq_paths[idx].stem
        gt_name = self.gt_paths[idx].stem
        assert lq_name == gt_name
        lq = torchvision.io.read_image(str(self.lq_paths[idx]))/255.0
        gt = torchvision.io.read_image(str(self.gt_paths[idx]))/255.0
            # 检查图像尺寸是否足够进行裁剪
        #固定裁剪区域的时候用
        # min_height = min(lq.shape[1], gt.shape[1])
        # min_width = min(lq.shape[2], gt.shape[2])
        #
        #     # 设置裁剪尺寸
        # crop_size = 256
        #
        # x = 1749
        # y = 1031
        #
        #     # 执行裁剪
        # if min_height >= crop_size and min_width >= crop_size:
        #         lq = lq[:, y:y + crop_size, x:x + crop_size]
        #         gt = gt[:, y:y + crop_size, x:x + crop_size]
        # else:
        #         # 如果图像尺寸小于裁剪尺寸，使用中心裁剪
        #         transform = torchvision.transforms.CenterCrop(crop_size)
        #         lq = transform(lq)
        #         gt = transform(gt)
        lq = lq[:, :self.crop_size, :self.crop_size]
        gt = gt[:, :self.crop_size, :self.crop_size]
        return lq, gt