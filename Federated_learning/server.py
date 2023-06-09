import flwr as fl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import color
c = color.clr()

data = {'avg_loss': 0, 'avg_accuracy': 0, 'loss': 0, 'accuracy': 0, 'num_samples': 0, 'num_failures': 0, 'parameters': ''}

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self):
        super().__init__()
        self.losses = []
        self.accuracies = []
        self.parameters = []
        
    def compute_metrics(self, results):
        print(results)
        # # Compute loss and accuracy based on results from clients
        # # You need to implement this based on your specific use case
        # # Iterate over the results to calculate the metrics
        # total_loss = 0.0
        # total_accuracy = 0.0
        # num_results = len(results)

        # for result in results:
        #     loss = result["loss"]
        #     accuracy = result["accuracy"]
        #     total_loss += loss
        #     total_accuracy += accuracy

        # avg_loss = total_loss / num_results
        # avg_accuracy = total_accuracy / num_results
        # return avg_loss, avg_accuracy
    
    def aggregate_fit(
        self, rnd, results, failures
    ):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        
        
        self.compute_metrics(results)
        # self.losses.append(loss)
        # self.accuracies.append(accuracy)

        # Store model parameters
        self.parameters.append(aggregated_weights)
        # Save aggregated_weights
        print(c.SUCCESS(f"Saving round {rnd} aggregated_weights..."))
        np.savez(f"round-{rnd}-weights.npz", *aggregated_weights)
        return aggregated_weights
        


if __name__ == "__main__":
    strategy = SaveModelStrategy()
    
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=1) ,
        strategy = strategy
    )

    f = open("output/server.txt", "w")
    f.write(str(data))
    f.close()