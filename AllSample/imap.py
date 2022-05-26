"""
https://blog.csdn.net/HUSTHY/article/details/103087691

"""
import time
from multiprocessing import Pool
from tqdm import tqdm


def main(project, patent, wmd_model, function):
    t1 = time.time()
    params = []
    for index in range(len(project)):
        params.append((index, project, patent, wmd_model))
    with Pool(12) as p:
        res = list(tqdm(p.imap(function, params), total=len(params), desc='多进程计算相似度，得出匹配结果：'))
    p.close()
    p.join()
    t2 = time.time()
    print('耗时：', (t2 - t1))
