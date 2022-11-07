import torch
import csv
import os

import utils_lightning

def predict_labels(model):
        
    model.load_state_dict(torch.load('GRU.pt'))
    model.eval()

    dataset, dataloader = utils_lightning.load_test_data()
    n_tests = len(dataset)

    predictions = []
    print("Started predicting\n")
    for index in range(n_tests):
        sample = dataset.__getitem__(index)[0].reshape(1, 300, 2)
        prediction = str(int(torch.argmax(model(sample))))
        predictions.append(prediction)
    
    print(predictions)
    
    i = 0
    while os.path.exists(f"predictions{i}.csv"):
        i += 1
        
    with open(f'predictions{i}.csv', mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['index', 'class'])
        for idx, prediction in enumerate(predictions):
            writer.writerow([idx, prediction])