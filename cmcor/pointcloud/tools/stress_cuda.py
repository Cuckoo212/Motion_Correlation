"""ROS/SAM-free CUDA launch stress reproducer; run under GDB."""
import argparse
import time
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--iterations', type=int, default=1000000)
args = parser.parse_args()
x = torch.randn(200, 256, device='cuda')
y = torch.randn_like(x)
started = time.monotonic()
with torch.inference_mode():
    for i in range(args.iterations):
        z = torch.cat((x, y), dim=0)
        z = torch.nn.functional.layer_norm(z, (256,))
        if i % 10000 == 0:
            torch.cuda.synchronize()
            print(f'iteration={i} elapsed={time.monotonic()-started:.2f}s', flush=True)
torch.cuda.synchronize()
print('PASS', flush=True)
