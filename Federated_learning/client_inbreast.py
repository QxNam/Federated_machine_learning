from client import *
NAME_DATA = 'INBREAST'
dataset = Dataset(NAME_DATA, transform=transform)
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2)
train_dataloader, test_dataloader = dataloader(train_dataset, test_dataset, batch_size=BATCH_SIZE)
model = GoogLeNet(num_classes=2).to(device)

class FlowerClient(fl.client.NumPyClient):
    
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(model, train_dataloader, epochs=1)
        return self.get_parameters(config={}), len(train_dataloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(model, test_dataloader)
        return loss, len(test_dataloader.dataset), {"accuracy": accuracy}

fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient(),
)