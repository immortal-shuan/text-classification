# import math
# import datetime
# import multiprocessing as mp
#
# def fun(x):
#     m = 1
#     for i in range(x):
#         m = m ** 2
#
# # start_t = datetime.datetime.now()
#
# if __name__ == '__main__':
#     num_cores = int(mp.cpu_count())
#     print("本地计算机有: " + str(num_cores) + " 核心")
#     pool = mp.Pool(4)
#     pool.apply_async(fun, (100,))
#     pool.apply_async(fun, (200,))
#     pool.apply_async(fun, (300,))
#     pool.apply_async(fun, (400,))
#     pool.close()
#     pool.join()


# import multiprocessing
# import os
# import time
# import numpy
#
# def task(args):
#     print("PID =", os.getpid(), ", args =", args)
#     return os.getpid(), args
#
# task("test")
# if __name__ == '__main__':
#     pool = multiprocessing.Pool(processes=4)
#
#     result = pool.map(task, [1,2,3,4,5,6,7,8])



# import random
# # 举例
# your_list = [1, 2, 3]
# b = random.sample(your_list, 2)
# 随机选取列表中一个元素
# random.choice(your_list)

from multiprocessing import Process

def f(name):
    print('hello', name)

if __name__ == '__main__':
    p = Process(target=f, args=('bob',))
    p.start()
    p.join()
