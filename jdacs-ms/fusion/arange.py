# -*- coding: utf-8 -*-
# @Time    : 2020/06/02 10:32
# @Author  : Xu HongBin
# @Email   : 2775751197@qq.com or 17770026885@163.com
# @github  : https://github.com/ToughStoneX
# @blog    : https://blog.csdn.net/hongbin_xu
# @File    : create_dtu_eval
# @Software: PyCharm

import os
from pathlib import Path


output_dir = Path("../outputs")
if not output_dir.exists():
    print('output_dir does not exist.'.format(output_dir))
    exit(-1)

fusibile_out_list = [
    "../outputs/fused_0.4_0.25",
]

dtu_eval_list = [
    "../outputs/mvsnet_0.4_0.25"
]

testlist = "../lists/dtu/test.txt"
with open(testlist) as f:
    scans = f.readlines()
    scans = [line.rstrip() for line in scans]

for i, fusibile_out_folder in enumerate(fusibile_out_list):
    for scan in scans:
        scan_folder = os.path.join(fusibile_out_folder, scan)
        consis_folders = [f for f in os.listdir(scan_folder) if f.startswith('consistencyCheck-')]
        consis_folders.sort()
        consis_folder = consis_folders[-1]
        source_ply = os.path.join(fusibile_out_folder, scan, consis_folder, 'final3d_model.ply')
        scan_idx = int(scan[4:])
        save_folder = dtu_eval_list[i]
        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)
        target_ply = os.path.join(save_folder, 'mvsnet{:03d}_l3.ply'.format(scan_idx))

        # cmd = 'cp ' + source_ply + ' ' + target_ply
        cmd = 'mv ' + source_ply + ' ' + target_ply
        print(cmd)
        os.system(cmd)
