import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
from time import time
import torch.distributed as dist

device = "cuda"
n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

dist_url = "env://"
world_size = torch.cuda.device_count()

start = time()


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    gpu = int(os.environ["LOCAL_RANK"])

    dist.init_process_group(
        backend="nccl",
        init_method=dist_url,
        world_size=world_size,
        rank=rank,
    )

    torch.cuda.set_device(gpu)
    print("| distributed init (rank {}): {}".format(rank, dist_url), flush=True)
    dist.barrier()
    setup_for_distributed(rank == 0)


init_distributed_mode()


train_ds, test_ds = [
    torchvision.datasets.MNIST(
        "/sprint1/datasets/MNIST/",
        train=is_train,
        download=True,
        # torchvision.datasets.MNIST('/files/', train=True, download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    )
    for is_train in [True, False]
]

train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
test_sampler = torch.utils.data.distributed.DistributedSampler(test_ds)

train_loader = torch.utils.data.DataLoader(
    train_ds,
    sampler=train_sampler,
    batch_size=batch_size_train,
)

test_loader = torch.utils.data.DataLoader(
    test_ds,
    sampler=test_sampler,
    batch_size=batch_size_test,
)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]


def train(epoch):
    network.train()
    train_loader.sampler.set_epoch(epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset))
            )


def test():
    network.eval()
    test_loss = 0
    correct = 0
    all_preds = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            all_preds += len(pred)
    test_loss /= all_preds
    test_losses.append(test_loss)
    print(
        "\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            all_preds,
            100.0 * correct / all_preds,
        )
    )


model = network.to(device)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])


test()
for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()

print("Time taken: ", time() - start)
