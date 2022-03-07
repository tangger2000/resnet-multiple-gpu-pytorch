"""
使用DistributedDataParallel
进行ResNet分布式训练
"""
import pandas as pd
import torch
from torch import nn, optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from resnet import ResNet
from distribute_tools import setup, cleanup, run_distribute
from torch.nn.parallel import DistributedDataParallel



def main_worker(rank, world_size):
    """
    这里定义每个线程需要执行的任务
    :param rank: 线程号,mp.spawn会自动传进来
    :param world_size: 节点数*每个节点控制的GPU数量,这个有两个节点,每个节点控制一张卡, 所以 2*1=2
    :return:
    """
    print(f"Running basic DistributedDataParallel example on rank {rank}.")
    setup(rank, world_size)

    batchsz = 128
    normalize_op = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    to_tensor_op = transforms.ToTensor()

    cifar_train = datasets.CIFAR10(r'./data',
                                   train=True,
                                   transform=transforms.Compose([
                                       to_tensor_op,
                                       normalize_op
                                       ]),
                                   download=True)
    # sampler是用于分发数据,如果是两张卡,数据机会对半分
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        cifar_train,
        num_replicas=world_size,
        rank=rank
    )
    # 要添加sampler用于分发数据,添加以后,如果有2张卡,step就会变成原来的1/2
    cifar_train = DataLoader(dataset=cifar_train,
                             batch_size=batchsz,
                             shuffle=False,
                             num_workers=0,
                             pin_memory=True,
                             sampler=train_sampler
                             )


    cifar_test = datasets.CIFAR10(r'./data',
                                  train=False,
                                  transform=transforms.Compose([
                                      to_tensor_op,
                                      normalize_op]),
                                  download=True)
    cifar_test = DataLoader(cifar_test,
                            batch_size=batchsz,
                            shuffle=True)


    model = ResNet().to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])

    criteon = nn.CrossEntropyLoss().to(rank)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 收集训练信息
    res_data = {"epoch": [], "loss": [], "acc": []}
    for epoch in range(100):
        train_sampler.set_epoch(epoch)
        model.train()
        for batchidx, (x, label) in enumerate(cifar_train):
            x, label = x.to(rank), label.to(rank)
            logits = model(x)
            loss = criteon(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batchidx % 50 == 0:
                print(f"rank: {rank}, epoch: {epoch}, step: {batchidx}, loss: {loss.item()}")
        # 在线程0中进行验证
        if rank == 0:
            model.eval()
            with torch.no_grad():
                total_correct = 0
                total_num = 0
                for x, label in cifar_test:
                    x, label = x.to(rank), label.to(rank)
                    logits = model(x)
                    pred = logits.argmax(dim=1)
                    correct = torch.eq(pred, label).float().sum().item()
                    total_correct += correct
                    total_num += x.size(0)
                acc = total_correct / total_num
                print(f"epoch: {epoch}, acc: {acc}")
            res_data["epoch"].append(epoch)
            res_data["loss"].append(loss.item())
            res_data["acc"].append(acc)
            df = pd.DataFrame(res_data)
            df.to_csv("pytorch_res.csv", index=False)
    cleanup()


if __name__ == '__main__':
    import time

    start_time = time.time()
    run_distribute(main_worker, 2)
    finish_time = time.time()
    print("total time cost: {} s".format(finish_time-start_time))