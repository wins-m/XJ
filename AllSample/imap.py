"""
tqdm多进程显示数据处理进度的方法
https://blog.csdn.net/qq_31385803/article/details/122352725

"""
import time
from multiprocessing import Pool, RLock, freeze_support
from tqdm import tqdm
import os


def my_process(process_name):
    # tqdm中的position参数需要设定呦！！！
    pro_bar = tqdm(range(50), ncols=80,
                   desc=f"Process—{process_name} pid:{str(os.getpid())}",
                   delay=0.01, position=process_name, ascii=False)
    for file in pro_bar:
        time.sleep(0.2)
    pro_bar.close()


if __name__ == '__main__':
    print(f'父进程 {os.getpid()}')
    freeze_support()
    pro_num = 3
    # 多行显示，需要设定tqdm中全局lock
    p = Pool(pro_num, initializer=tqdm.set_lock, initargs=(RLock(),))
    for idx in range(pro_num):
        p.apply_async(image_process, kwds={"process_name": idx})

    p.close()
    p.join()
