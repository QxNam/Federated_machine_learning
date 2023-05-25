import flwr as fl
from typing import OrderedDict, Dict, Tuple, Optional, Callable, List
import numpy as np

from centralized import train, Net, device, Dataloader, transformtransform_image

from flwr.common import (
    Parameters,
    Scalar,
    Weights,
    parameters_to_weights,
    weights_to_parameters,
)
import torch
import torchvision

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=1)
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}



def client_init(name_client: str) -> FlowerClient:
    net = Net().to(device)
    
    train_loader, test_loader = Dataloader(path_data=f'dataset/{name_client.upper()}-ROI-Mammography', \
        transform=transformtransform_image(), batch_size=32)

    return FlowerClient(net, train_loader, test_loader)