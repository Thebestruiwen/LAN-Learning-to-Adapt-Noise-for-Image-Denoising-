import subprocess
import torchvision
from pathlib import Path
from tqdm import tqdm

subprocess.run(["git", "clone", "https://github.com/csjunxu/PolyU-Real-World-Noisy-Images-Dataset.git"])

patch_size = 512
root_dir = Path("PolyU-Real-World-Noisy-Images-Dataset/OriginalImages")
output_dir = Path("polyu")

gt_dir = output_dir / "gt"
lq_dir = output_dir / "lq"
for lq_path, gt_path in tqdm(list(zip(sorted(root_dir.glob("*Real.JPG")), sorted(root_dir.glob("*mean.JPG"))))):
    lq_name = "_".join(lq_path.stem.split("_")[:-1])
    gt_name = "_".join(gt_path.stem.split("_")[:-1])
    assert lq_name == gt_name
    lq = torchvision.io.read_image(str(lq_path))/255.0
    gt = torchvision.io.read_image(str(gt_path))/255.0
    lq_patches = lq.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size).permute(1, 2, 0, 3, 4).reshape(-1, 3, patch_size, patch_size)
    gt_patches = gt.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size).permute(1, 2, 0, 3, 4).reshape(-1, 3, patch_size, patch_size)
    for i, (lq_patch, gt_patch) in enumerate(zip(lq_patches, gt_patches)):
        lq_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)
        torchvision.utils.save_image(lq_patch, str(lq_dir / f"{lq_name}_{str(i).zfill(4)}.png"))
        torchvision.utils.save_image(gt_patch, str(gt_dir / f"{gt_name}_{str(i).zfill(4)}.png"))

subprocess.run(["rm", "-rf", "PolyU-Real-World-Noisy-Images-Dataset"])