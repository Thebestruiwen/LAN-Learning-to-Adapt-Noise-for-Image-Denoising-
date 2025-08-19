from pathlib import Path
import importlib

path = 'Restormer.Denoising.utils'
module_spec = importlib.util.spec_from_file_location(path, str(Path().joinpath(*path.split('.')))+'.py')
module = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(module)

def cal_psnr(a, b):
    a = a.squeeze(0) * 255
    b = b.squeeze(0) * 255
    a = a.permute(1, 2, 0).cpu().detach().numpy()
    b = b.permute(1, 2, 0).cpu().detach().numpy()
    return module.calculate_psnr(a, b)

def cal_ssim(a, b):
    a = a.squeeze(0) * 255
    b = b.squeeze(0) * 255
    a = a.permute(1, 2, 0).cpu().detach().numpy()
    b = b.permute(1, 2, 0).cpu().detach().numpy()
    return module.calculate_ssim(a, b)

def cal_batch_psnr_ssim(pred, gt):
    psnr = []
    ssim = []
    for i in range(pred.shape[0]):
        psnr.append(cal_psnr(pred[i:i+1], gt[i:i+1]))
        ssim.append(cal_ssim(pred[i:i+1], gt[i:i+1]))
    return psnr, ssim