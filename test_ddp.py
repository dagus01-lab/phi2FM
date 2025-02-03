import torch.distributed as dist
import torch

def main():
    dist.init_process_group(backend='nccl', init_method='env://')
    rank = dist.get_rank()
    print(f"Hello from process {rank}!")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
