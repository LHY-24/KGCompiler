import GPUtil
import time

# 获取所有GPU的信息
GPUs = GPUtil.getGPUs()

while True:
    time.sleep(1) 
    # 打印每个GPU的信息
    for gpu in GPUs:
      print('GPU ID: ', gpu.id)
      print('GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB'.format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))