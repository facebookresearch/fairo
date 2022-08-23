import torch
import torch.nn as nn
import torch.utils.data as tds
import submitit

def add_args(parser):
    parser.add_argument(
        "--distributed", action="store_true", default=False, help="distributed training"
    )
    parser.add_argument("--local_rank", type=int, default=0, help="")
    parser.add_argument(
        "--dist-init", type=str, default="", help="distributed training"
    )


def init(args):
    if args.submitit:
        job_env = submitit.JobEnvironment()
        args.local_rank = job_env.local_rank
        args.rank = job_env.global_rank
        args.world_size = job_env.num_tasks
        args.batchsize = args.batchsize // args.world_size
        torch.distributed.init_process_group(
            backend="nccl",
            init_method=args.dist_init,
            rank=job_env.global_rank,
            world_size=job_env.num_tasks,
        )

def split_data(args, train_data, val_data):
    assert args.batchsize % args.world_size == 0
    args.batchsize = args.batchsize // args.world_size
    train_data = train_data[args.batchsize * args.rank : args.batchsize * (args.rank + 1)]
    val_data = val_data[args.batchsize * args.rank : args.batchsize * (args.rank + 1)]
    return train_data, val_data

def wrap_model(args, model):
    if args.distributed:
        model = model.to(args.device)
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
    else:

        class DummyWrapper(nn.Module):
            def __init__(self, mod):
                super(DummyWrapper, self).__init__()
                self.module = mod

            def forward(self, *args, **kwargs):
                return self.module(*args, **kwargs)

            def save(self, *args, **kwargs):
                return self.module.save(*args, **kwargs)
            
            def load(self, *args, **kwargs):
                return self.module.load(*args, **kwargs)
            
            def _build(self, *args, **kwargs):
                return self.module._build(*args, **kwargs)

        model = DummyWrapper(model)
        model = model.to(args.device)
    return model

def wrap_dataset(args, dataset, collater=None, sampler=None, test=False):
    if args.distributed:
        job_env = submitit.JobEnvironment()
        sampler = tds.distributed.DistributedSampler(
            dataset, num_replicas=job_env.num_tasks, rank=job_env.global_rank
        )
    batchsize = args.batchsize
    dataloader = tds.DataLoader(
        dataset, collate_fn=collater, batch_size=batchsize, shuffle=(sampler is None), sampler=sampler, drop_last=True, num_workers=args.num_workers
    )
    return dataloader, sampler