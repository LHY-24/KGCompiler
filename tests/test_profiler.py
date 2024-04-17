import os
import numpy as np
import torch
from torchvision.models import resnet18
import time


if __name__ == '__main__':
    model = resnet18(pretrained=True)
    device = torch.device('cuda')
    model.eval()
    model.to(device)
    dump_input = torch.ones(1,3,224,224).to(device)

    # Warn-up
    for _ in range(5):
        start = time.time()
        outputs = model(dump_input)
        torch.cuda.synchronize()
        end = time.time()
        print('Time:{}ms'.format((end-start)*1000))

    with torch.profiler.profile(
        activities=[
          torch.profiler.ProfilerActivity.CPU,
          torch.profiler.ProfilerActivity.CUDA,
        ], 
        schedule=torch.profiler.schedule(
          wait=1,
          warmup=1,
          active=2,
          repeat=1
        ),
        record_shapes=True, profile_memory=True, with_stack=True, on_trace_ready=torch.profiler.tensorboard_trace_handler("./res/")
      ) as p:
        outputs = model(dump_input)
    print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
