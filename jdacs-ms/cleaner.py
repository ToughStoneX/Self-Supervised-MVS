# -*- coding: utf-8 -*-
# @Time    : 2020/04/16 21:25
# @Author  : Xu HongBin
# @Email   : 2775751197@qq.com or 17770026885@163.com
# @github  : https://github.com/ToughStoneX
# @blog    : https://blog.csdn.net/hongbin_xu
# @File    : cleaner
# @Software: PyCharm

from pathlib import Path


def rm_dir(dir_path, silent=True):
    p = Path(dir_path).resolve()
    if (not p.is_file()) and (not p.is_dir()) :
        print('It is not path for file nor directory :',p)
        return

    paths = list(p.iterdir())
    if (len(paths) == 0) and p.is_dir() :
        p.rmdir()
        if not silent : print('removed empty dir :',p)

    else :
        for path in paths :
            if path.is_file() :
                path.unlink()
                if not silent : print('removed file :',path)
            else:
                rm_dir(path)
        p.rmdir()
        if not silent : print('removed empty dir :',p)


def clean():
    ckpt_dir = Path('./checkpoints')
    log_dir = Path('./logs')
    summary_dir = Path('./summary')

    rm_dir(ckpt_dir)
    rm_dir(log_dir)
    rm_dir(summary_dir)

    print('{} removed!'.format(ckpt_dir))
    print('{} removed!'.format(log_dir))
    print('{} removed!'.format(summary_dir))
    print('[*] Cleaning finished!')


if __name__ == '__main__':
    clean()