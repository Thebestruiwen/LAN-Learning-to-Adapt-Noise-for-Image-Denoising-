import pandas as pd
import argparse

#python resultHandle.py --inputDir /home/yanyutai/miniconda3/envs/Lan/LAN-master/results/Restormer_zsn2n_Nam --inputFile result_lan_zsn2n.csv

def handle(inputDir,inputFile,targetItem):

    # 读取CSV文件（替换为你的实际文件路径）
    file_path = f'{inputDir}/{inputFile}'  # 修改为你的CSV文件路径
    df = pd.read_csv(file_path)  # 如果文件没有列名，添加: header=None, names=['idx','loop','psnr','ssim']

    # 定义目标迭代次数
    if targetItem == 20:
        target_iterations = [5, 10, 15, 20]
    else:
        target_iterations = [5, 10, 15, 20,25,30]

    result_df = (
        df[df['loop'].isin(target_iterations)]
        .groupby('loop')
        .agg(
            avg_psnr=('psnr', 'mean'),
            avg_ssim=('ssim', 'mean')
        )
        .reset_index()
    )

    # 第二步：添加字符串列，格式为 "avg_psnr/avg_ssim"
    # 注意：这里需要将数字转换为字符串，然后拼接
    # 同时，我们可以控制小数位数，比如保留两位小数
    result_df['psnr_ssim'] = result_df.apply(
        lambda row: f"{row['avg_psnr']:.4f}/{row['avg_ssim']:.4f}",
        axis=1
    )


    print(result_df)
    # 保存结果到CSV文件
    output_file = f'{inputDir}/average_results.csv'
    result_df.to_csv(output_file, index=False, float_format='%.4f')

    print(f"结果已保存到: {output_file}")
    print("\n结果预览:")
    print(result_df)
