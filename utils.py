import torch
import torchvision
import torchvision.transforms as transforms

def acc(y_pred, y):
    return torch.sum(y_pred.argmax(1) == y).item()/len(y)

def eye_like(tensor):
    return torch.eye(*tensor.size(), out=torch.empty_like(tensor))

def load_fashion_mnist(batch_size: int = 128) -> dict:
    transform = transforms.Compose(
        [
         transforms.ToTensor(), 
         transforms.Lambda(lambda x: torch.flatten(x))
        ]
    )

    trainset = torchvision.datasets.FashionMNIST(
        root="../data", download=True, train=True, transform=transform
    )
    testset = torchvision.datasets.FashionMNIST(
        root="../data", download=True, train=False, transform=transform
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )

    return {
        "trainloader": trainloader,
        "testloader": testloader,
        "num_classes": 10,
        "dim": 784,
    }

