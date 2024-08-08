#!/home/hongyu2021/anaconda3/envs/smore/bin/python

import subprocess 
import time
from datetime import datetime
import argparse
import os

def append_to_file(s, f_path, end="\n"):
    with open(f_path, "a") as f:
        print(s, file=f, end=end)
    
    
parser = argparse.ArgumentParser()
parser.add_argument("-o", "--out", type=str, required=True, help="output file path")
parser.add_argument("-p", "--pid", type=str, required=True, help="pid of parent process")
args = parser.parse_args()

output_path = args.out
# add csv head
with open(output_path, "w") as f:
    print("time, pid, mem_occ", file=f)

while(True):
    # print pid and mem occ
    # 查找父进程的所有子进程
    pids = [args.pid]
    try:
        pgrep_output = subprocess.check_output(['pgrep', '-P', args.pid])
        raw_pids = pgrep_output.decode()
        for pid in raw_pids.split("\n"):
            if pid:
                pids.append(pid)
    except:
        pass
    # total_mem_occ = 0
    # print(pids)
    # 遍历所有的进程
    for i, p in enumerate(pids):
        # print col of time
        if i == 0:
            now_str = datetime.now().strftime("%H:%M:%S") 
            append_to_file(now_str, output_path, ", ")
        else:
            append_to_file("", output_path, ", ")
        
        # print col of mem_occ
        # 调用nvidia-smi命令获取子进程显存占用情况 
        nvidia_output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,nounits,noheader']) 
        raw_memory_used = nvidia_output.decode()
        lines = raw_memory_used.splitlines()
        flag = True
        for line in lines:
            if p in line:
                append_to_file(line, output_path)
                flag = False
                break
        if flag:
            append_to_file(p + ", 0", output_path)
            
        
        
    # print total
    # append_to_file(", total, " + str(total_mem_occ), output_path)
    time.sleep(2.5)