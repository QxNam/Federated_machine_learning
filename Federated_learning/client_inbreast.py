from client import *
import csv

NAME_DATA = 'INBREAST'
dataset = Dataset(NAME_DATA, transform=transform)
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2)
train_dataloader, test_dataloader = dataloader(train_dataset, test_dataset, batch_size=BATCH_SIZE)
model = GoogLeNet(num_classes=2).to(device)
data = {'train': 0, 'val': 0}

class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        data_train = train(model, train_dataloader, epochs=50)
        data['train'] = data_train
        return self.get_parameters(config={}), len(train_dataloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy, data_test = test(model, test_dataloader)
        data['val'] = data_test
        return loss, len(test_dataloader.dataset), {"accuracy": accuracy}

fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient(),
)

with open(f'output/inbreast_train.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header row
    writer.writerow(['id', 'loss', 'accuracy'])
    # Write the data rows
    for i in range(len(data['train']['id'])):
        writer.writerow([data['train']['id'][i], data['train']['loss'][i], data['train']['accuracy'][i]])

with open(f'output/inbreast_test.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header row
    writer.writerow(['id', 'loss', 'accuracy'])
    # Write the data rows
    for i in range(len(data['val']['id'])):
        writer.writerow([data['val']['id'][i], data['val']['loss'][i], data['val']['accuracy'][i]])

print(c.SUCCESS('Saved data to csv!'))