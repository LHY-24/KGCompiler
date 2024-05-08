import torch

def add(a, b):
    return a + b

def sub(a, b):
    return a - b

opt_add = torch.compile(add)
opt_sub = torch.compile(sub)


# **changed**  import TimeCounter.py

# from TimeCounter import TimeCounter


import time

class AverageTimeClock:
    count = 0
    start = time.perf_counter()
    def __init__(self, name: str) -> None:
        # self.start = time.perf_counter()
        self.end = 0
        self.name = name
    def __enter__(self):
        # self.start = time.perf_counter()
        pass
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.perf_counter()
        time_s = (self.end - AverageTimeClock.start) / (AverageTimeClock.count + 1)
        print("[%s]-%d: %fms"%(self.name, AverageTimeClock.count, time_s * 1000))
        AverageTimeClock.count += 1

class TimeClock:
    count = 0
    def __init__(self, name: str) -> None:
        self.start = 0
        self.end = 0
        self.name = name
    def __enter__(self):
        self.start = time.perf_counter()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.perf_counter()
        time_s = self.end - self.start
        print("[%s]-%d: %fms"%(self.name, TimeClock.count, time_s * 1000))
        TimeClock.count += 1

if __name__ == "__main__":
    for _ in range(10):
        # with TimeCounter.profile_time("time counter"):
        # with AverageTimeClock("average time clock"):
        with TimeClock("time clock"):
            x = torch.randn([32, 32])
            y = torch.randn([32, 32])

            z_1 = opt_add(x, y)
            z_2 = opt_sub(x, y)
            # z_1 = opt_add(x, y)
            # z_2 = opt_sub(x, y)
            # z_1 = opt_add(x, y)
            # z_2 = opt_sub(x, y)
            # z_1 = opt_add(x, y)
            # z_2 = opt_sub(x, y)
            # print(z_1, z_2, sep = "\n"+"*"*20+"\n") 