import json
import argparse
from experiment import ExperimentCollection, Experiment
from feynman_dataset import get_feynman_dataloader
from occamnet import OccamNet

# Assuming the JSON file is in the root directory of Colab
json_file_path = "/content/parameters.json"

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
    
    experiment_collection.run()
