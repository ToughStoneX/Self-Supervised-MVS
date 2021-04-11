# -*- coding: utf-8 -*-
# @Time    : 2020/05/21 22:34
# @Author  : Xu HongBin
# @Email   : 2775751197@qq.com or 17770026885@163.com
# @github  : https://github.com/ToughStoneX
# @blog    : https://blog.csdn.net/hongbin_xu
# @File    : __init__.py
# @Software: PyCharm

import importlib


# find the dataset definition by name, for example dtu_yao (dtu_yao.py)
def find_dataset_def(dataset_name):
    module_name = 'datasets.{}'.format(dataset_name)
    module = importlib.import_module(module_name)
    return getattr(module, "MVSDataset")