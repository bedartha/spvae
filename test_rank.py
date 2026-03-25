import os
import torch.distributed as dist

def main():
    # 1. Initialize the process group. 
    # Because we will use torchrun, we don't need to pass rank or world_size here;
    # init_process_group will read them automatically from the environment variables.
    dist.init_process_group(backend='nccl')
    
    # 2. Now you can safely get your global and local ranks
    global_rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    
    print(f"Success! I am global rank {global_rank} out of {world_size}. My local rank on this node is {local_rank}.")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
