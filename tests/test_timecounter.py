import sys
import os
sys.path.append("/home/hongyu2021/KG-Compilation/utils")
from TimeCounter import TimeCounter

for _ in range(10):
    with TimeCounter.profile_time("counter1", "/home/hongyu2021/KG-Compilation/tests/output"):
        print("123")