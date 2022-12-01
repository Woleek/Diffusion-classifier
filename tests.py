import torch
import csv
import os

import utils_lightning

def predict_labels(model, i):
    """Predicts labels for training dataset. Model is loaded from saved state. Predicted labels are computed and saved to predictions.csv file.

    Args:
        model (LitModel): recurrent neural network model (GRU)
    """
    
    model.load_state_dict(torch.load(f'results/GRU{i}.pt'))
    model.eval()

    dataset, dataloader = utils_lightning.load_test_data()
    n_tests = len(dataset)

    predictions = []
    for index in range(n_tests):
        sample = dataset.__getitem__(index)[0].reshape(1, 300, 2)
        prediction = str(int(torch.argmax(model(sample))))
        predictions.append(prediction)
        
    with open(f'results/predictions{i}.csv', mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['index', 'class'])
        for idx, prediction in enumerate(predictions):
            writer.writerow([idx, prediction])