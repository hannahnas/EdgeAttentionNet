import torch
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data_folder = '/home/hannah/Documents/Thesis/data/preprocessed_structured3D'
    # data_folder = '/project/pjsminha/data/preprocessed_structured3D'
    split = 'val'
    inds = np.loadtxt(f'{data_folder}/{split}_inds.txt', dtype=str)

    filtered = []
    percentages = []
    
    for i in inds:
        edges_path = f'{data_folder}/{split}/gray_edges/{i}_gray_edges.npy'
        edges = np.load(edges_path)[2:254, 2:254]
        if np.all((edges == 0)):
            continue
        percentage = edges.sum()/edges.size
        if percentage < 0.0050:
            continue

        filtered.append(i)
        percentages.append(percentage)
    
    print(np.mean(percentages))

    np.savetxt(f'{data_folder}/filtered_{split}_inds.txt', filtered, fmt='%s')