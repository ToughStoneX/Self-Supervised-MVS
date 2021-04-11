# -*- coding: utf-8 -*-
# @Time    : 2020/05/30 14:08
# @Author  : Xu HongBin
# @Email   : 2775751197@qq.com or 17770026885@163.com
# @github  : https://github.com/ToughStoneX
# @blog    : https://blog.csdn.net/hongbin_xu
# @File    : cleaner
# @Software: PyCharm

import os
from pathlib import Path

print('cleaning temporary directories.')
pwd = os.getcwd()
names = os.listdir(pwd)
for name in names:
    if 'log' in name:
        temp_path = os.path.join(pwd, name)
        print('removing {}'.format(temp_path))
        os.system('rm -rf {}'.format(temp_path))
print('done.')
