import json
import argparse
from experiment import ExperimentCollection, Experiment
from feynman_dataset import get_feynman_dataloader
from occamnet import OccamNet
import matplotlib.pyplot as plt
import numpy as np
import torch

# Assuming the JSON file is in the root directory of Colab
json_file_path = "/content/parameters.json"

def plot_occamnet_learning_curves(experiment_collection):
    num_experiments = len(experiment_collection.experiments)
    fig, axes = plt.subplots(num_experiments + 2, 1, figsize=(12, 6*(num_experiments + 2)), sharex=True)
    fig.suptitle('Learning Curves for OccamNet Experiments', fontsize=16)

    # Individual experiment plots
    for i, experiment in enumerate(experiment_collection.experiments):
        ax = axes[i]
        for run, loss_list in enumerate(experiment.losses):
            epochs = [loss[0] for loss in loss_list]
            mean_g_values = [loss[1] for loss in loss_list]
            ax.plot(epochs, mean_g_values, label=f'Run {run+1}')
        ax.set_title(f'Experiment {i+1}: {experiment.name}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean G(f(x))')
        ax.legend()

    # Overlaid learning curves
    ax_overlay = axes[-2]
    for i, experiment in enumerate(experiment_collection.experiments):
        for run, loss_list in enumerate(experiment.losses):
            epochs = [loss[0] for loss in loss_list]
            mean_g_values = [loss[1] for loss in loss_list]
            ax_overlay.plot(epochs, mean_g_values, label=f'Exp {i+1}, Run {run+1}')
    
    ax_overlay.set_title('Overlaid Learning Curves')
    ax_overlay.set_xlabel('Epoch')
    ax_overlay.set_ylabel('Mean G(f(x))')
    ax_overlay.legend()

    # Average final performance
    ax_avg = axes[-1]
    x = range(num_experiments)
    avg_final_g = [np.mean([loss_list[-1][1] for loss_list in experiment.losses]) 
                   for experiment in experiment_collection.experiments]
    ax_avg.bar(x, avg_final_g, width=0.6)
    ax_avg.set_title('Average Final Mean G(f(x)) per Experiment')
    ax_avg.set_xlabel('Experiment')
    ax_avg.set_ylabel('Average Final Mean G(f(x))')
    ax_avg.set_xticks(x)
    ax_avg.set_xticklabels([f'Exp {i+1}' for i in x])

    plt.tight_layout()
    plt.savefig('learning_curves.png')
    plt.show()

def create_model_and_dataset(experiment_params):
    # Create the Feynman dataset
    train_loader = get_feynman_dataloader(batch_size=experiment_params.get('batch_size', 32))

    # Create the OccamNet model
    model = OccamNet(
        bases=experiment_params.get('bases', ['add', 'mul', 'div', 'sub', 'sin', 'cos', 'exp', 'log']),
        number_of_inputs=5,  # Feynman dataset can have up to 5 inputs
        number_of_outputs=1,
        depth=experiment_params.get('depth', 3),
        temperature=experiment_params.get('temperature', 1.0),
        device=experiment_params.get('device', "cuda" if torch.cuda.is_available() else "cpu")
    )

    return model, train_loader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--collection_name', type=str, default='feynman_collection')
    args = parser.parse_args()
    
    experiment_collection = ExperimentCollection(args.collection_name)
    
    with open(json_file_path, 'r') as f:
        parameters = json.load(f)
        for experiment_params in parameters['collection']:
            model, dataset = create_model_and_dataset(experiment_params)
            experiment = Experiment(model=model, dataset=dataset, **experiment_params)
            experiment_collection.push(experiment)
    
    # Run all experiments
    experiment_collection.run()

    # After all experiments are complete, plot the learning curves
    plot_occamnet_learning_curves(experiment_collection)
