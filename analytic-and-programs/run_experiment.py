
import json
import argparse
from experiment import ExperimentCollection, Experiment

# Assuming the JSON file is in the root directory of Colab
json_file_path = "/content/parameters.json"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--collection_name', type=str, default='basic_collection')
    args = parser.parse_args()

    experiment_collection = ExperimentCollection(args.collection_name)
    
    with open(json_file_path, 'r') as f:
        parameters = json.load(f)
        for experiment_params in parameters['collection']:
            experiment_collection.push(Experiment(**experiment_params))

    experiment_collection.run()
