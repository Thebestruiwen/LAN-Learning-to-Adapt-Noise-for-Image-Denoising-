import torch
from data import Dataset
from data_mat_advanced import MatDataset

from model import get_model_Restormer,get_model_dncnn,get_model_Uformer
from metric import cal_batch_psnr_ssim
import pandas as pd
from tqdm import tqdm
import argparse
from adapt import zsn2n, nbr2nbr
import numpy as np
import os
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import scipy.io as sio
from datetime import datetime
from resultHandle import  handle
# python main.py --model Restormer --dataset poly-U --method lan --self_loss {zsn2n, nbr2nbr} --num_samples 1 --num_GPU 1

parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, required=True, choices=["DnCNN", "Restormer","Uformer"])

parser.add_argument("--dataset", type=str, required=True, choices=["poly-U", "Nam","NamMAT","FIG3","FIG6"])

parser.add_argument("--method", type=str, required=True, choices=["finetune", "lan"])

parser.add_argument("--self_loss", type=str, required=True, choices=["nbr2nbr", "zsn2n"])

# 添加 num_samples 参数，默认值为 None（表示使用所有样本）
parser.add_argument("--num_samples", type=int, default=None,
                    help="Number of samples to test (default: all samples)")


parser.add_argument("--num_GPU", type=str, required=True,choices=["0", "1"])

args = parser.parse_args()

print("===============================")
print(f"使用的GPU序号：{args.num_GPU}")

# 设置使用哪张GPU（例如1号卡）
os.environ["CUDA_VISIBLE_DEVICES"] = args.num_GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建结果目录
now = datetime.now()
# 获取月份和日期并转为字符串
# month_str = str(now.month)  # 月份 (1-12)
# day_str = str(now.day)
# hour_str=str(now.hour)

results_dir = f"results/{args.method}_{args.model}_{args.self_loss}_{args.dataset}"

os.makedirs(results_dir, exist_ok=True)
print(f"所有结果将保存在: {os.path.abspath(results_dir)}")

if args.self_loss == "zsn2n":
    loss_func = zsn2n.loss_func
elif args.self_loss == "nbr2nbr":
    loss_func = nbr2nbr.loss_func
else:
    raise NotImplementedError

if args.model=="Restormer":
    model_generator = get_model_Restormer
elif args.model=="Uformer" :
    model_generator = get_model_Uformer
else:
    model_generator = get_model_dncnn
model = model_generator().to(device)
for param in model.parameters():
    param.requires_grad = args.method == "finetune"
print("trainable model parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

# 创建完整数据集
print(f"将使用{args.dataset}数据集进行训练")
if args.dataset == 'poly-U':
    full_dataset = Dataset("polyu/lq", "polyu/gt")
elif args.dataset == 'Nam':
    full_dataset = Dataset("real_image_noise_dataset/img_noisy", "real_image_noise_dataset/img_mean")
elif args.dataset == 'NamMAT':
    full_dataset = MatDataset("real_image_noise_dataset", crop_size=256)
elif args.dataset == 'FIG3':
    full_dataset = Dataset("real_image_noise_dataset/NamFIG3/img_noisy", "real_image_noise_dataset/NamFIG3/img_mean")
else:
    full_dataset = Dataset("results/adaptIMG/lq", "results/adaptIMG/gt")

# 如果指定了 num_samples，则创建子集
if args.num_samples is not None:
    # 确保不超过数据集的实际大小
    num_samples = min(args.num_samples, len(full_dataset))
    # 创建子集索引
    indices = list(range(num_samples))
    dataset = torch.utils.data.Subset(full_dataset, indices)
    print(f"使用前 {num_samples} 张图片进行测试 (共 {len(full_dataset)} 张)")
else:
    dataset = full_dataset
    print(f"使用所有 {len(full_dataset)} 张图片进行测试")

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
lr = 5e-4 if args.method == "lan" else 5e-6


class Lan(torch.nn.Module):
    def __init__(self, shape):
        super(Lan, self).__init__()
        self.phi = torch.nn.parameter.Parameter(torch.zeros(shape), requires_grad=True)

    def forward(self, x):
        return x + torch.tanh(self.phi)


logs_key = ["psnr", "ssim"]
total_logs = {key: [] for key in logs_key}
inner_loop = 30
p_bar = tqdm(dataloader, ncols=100, desc=f"{args.method}_{args.self_loss}")

# 初始化空的DataFrame
df = pd.DataFrame()

# 样本计数器
sample_idx = 0

for lq, gt in p_bar:
    sample_idx += 1
    lq = lq.to(device)
    gt = gt.to(device)
    lan = Lan(lq.shape).to(device) if args.method == "lan" else torch.nn.Identity()
    tmp_batch_size = lq.shape[0]
    model = model_generator().to(device)
    for param in model.parameters():
        param.requires_grad = args.method == "finetune"

    params = list(lan.parameters()) if args.method == "lan" else list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    logs = {key: [] for key in logs_key}

    for i in range(inner_loop):
        optimizer.zero_grad()
        adapted_lq = lan(lq)
        with torch.no_grad():
            pred = model(adapted_lq).clip(0, 1)
        loss = loss_func(adapted_lq, model, i, inner_loop)
        loss.backward()
        optimizer.step()
        psnr, ssim = cal_batch_psnr_ssim(pred, gt)
        for key in logs_key:
            logs[key].append(locals()[key])
    else:
        with torch.no_grad():
            adapted_lq = lan(lq) # 获取适应后的噪声图像
            final_pred = model(adapted_lq).clip(0, 1)
            psnr, ssim = cal_batch_psnr_ssim(final_pred, gt)
        for key in logs_key:
            logs[key].append(locals()[key])

    # 为当前样本创建目录
    sample_dir = os.path.join(results_dir, f"sample_{sample_idx}")
    os.makedirs(sample_dir, exist_ok=True)

    # 保存输入图像（低质量）
    vutils.save_image(lq.cpu(), os.path.join(sample_dir, "input.png"))

    # 保存为.mat文件 (H x W x C 格式)
    input_np = lq.squeeze(0).permute(1, 2, 0).cpu().numpy()  # 移除批次维度并调整通道顺序
    sio.savemat(os.path.join(sample_dir, "input.mat"), {"input": input_np})

    #保存适应完噪音之后的图像
    adapted_img = adapted_lq.detach().cpu()
    vutils.save_image(adapted_img, os.path.join(sample_dir, "adapted_noise.png"))

    # 保存为.mat文件 (H x W x C 格式)
    adapted_np = adapted_img.squeeze(0).permute(1, 2, 0).numpy()
    sio.savemat(os.path.join(sample_dir, "adapted_noise.mat"),
                {"adapted_noise": adapted_np})

    # 保存最终预测结果
    vutils.save_image(final_pred.cpu(), os.path.join(sample_dir, "output.png"))
    output_np = final_pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
    sio.savemat(os.path.join(sample_dir, "output.mat"), {"output": output_np})

    # 保存真实图像（高质量）
    vutils.save_image(gt.cpu(), os.path.join(sample_dir, "ground_truth.png"))
    gt_np = gt.squeeze(0).permute(1, 2, 0).cpu().numpy()
    sio.savemat(os.path.join(sample_dir, "ground_truth.mat"), {"ground_truth": gt_np})


    for key in logs_key:
        total_logs[key].extend(np.array(logs[key]).transpose())
    p_bar.set_postfix(
        PSNR=f"{np.array(total_logs['psnr']).mean(0)[0]:.2f}->{np.array(total_logs['psnr']).mean(0)[-1]:.2f}",
        SSIM=f"{np.array(total_logs['ssim']).mean(0)[0]:.3f}->{np.array(total_logs['ssim']).mean(0)[-1]:.3f}"
    )

    # 更新DataFrame
    df_dict = {
        "idx": [i for i in range(len(total_logs['psnr'])) for _ in range(inner_loop + 1)],
        "loop": [i for i in range(inner_loop + 1)] * len(total_logs['psnr']),
    }
    for key in logs_key:
        df_dict[key] = [value for value_list in total_logs[key] for value in value_list]
    df = pd.DataFrame(df_dict)
    df.to_csv(os.path.join(results_dir, f"{args.model}_{args.method}_{args.self_loss}.csv"), index=False)

    # 打印当前batch的统计结果
    if not df.empty:
        print(df.groupby('loop').mean()[['psnr', 'ssim']])
    else:
        print("Warning: No data processed!")

# 保存全局平均曲线
avg_psnr = np.array(total_logs['psnr']).mean(axis=0)
avg_ssim = np.array(total_logs['ssim']).mean(axis=0)


print(f"所有结果已保存到: {os.path.abspath(results_dir)}")
print(f"测试了 {sample_idx} 张图片")
try:
    handle(results_dir,f"result_{args.method}_{args.self_loss}.csv",inner_loop)
except Exception as e:
    print(f"输出平均值失败，原因：{e}")