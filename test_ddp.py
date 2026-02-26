import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

def spmd_main():
    # These are the parameters used to initialize the process group
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
    rank = int(env_dict['RANK'])
    local_rank = int(env_dict['LOCAL_RANK'])
    local_world_size = int(env_dict['LOCAL_WORLD_SIZE'])

    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    acc = torch.accelerator.current_accelerator()
    vendor_backend = torch.distributed.get_default_backend_for_device(acc)
    torch.accelerator.set_device_index(rank)
    torch.distributed.init_process_group(backend=vendor_backend)

    demo_basic(rank)

    # Tear down the process group
    torch.distributed.destroy_process_group()


def demo_basic(rank):
    print(
        f"[{os.getpid()}] rank = {torch.distributed.get_rank()}, "
        + f"world_size = {torch.distributed.get_world_size()}"
    )

    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)
    loss_fn(outputs, labels).backward()
    optimizer.step()


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


if __name__ == "__main__":
    spmd_main()

