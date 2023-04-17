import pickle
import os

import numpy as np

dataset = "nbody-n4"
#method = "dropout_0.5"
#method = "vi"
method = "none"
print(dataset)
print(method)
file_path = os.path.join(os.path.expanduser("~"), 
                         "maplecg_nfs_public/bpignn/results", dataset, method, "all_traj.pkl")
pickle_file = open(file_path, "rb")                         
trajs = pickle.load(pickle_file)

rel_res = []
abs_res = []

for i in range(len(trajs)):
    traj = trajs[i]
    all_pred_pos = traj["pred_pos"]
    actual_pos = traj["actual_pos"]
    actual_norm = np.linalg.norm(actual_pos)
    total_time = actual_pos.shape[0]
    rel_mses = []
    abs_mses = []
    
    for j in range(all_pred_pos.shape[0]):
        pred_pos = all_pred_pos[j]
        diff = pred_pos - actual_pos
        displace = np.linalg.norm(diff)
        rel_mse = displace / actual_norm
        abs_mse = displace / total_time
        rel_mses.append(rel_mse)
        abs_mses.append(abs_mse)
        
    rel_mses = np.array(rel_mses)
    abs_mses = np.array(abs_mses)
    avg_rel_mse = np.mean(rel_mses)
    avg_abs_mse = np.mean(abs_mses)
    rel_res.append(avg_rel_mse)
    abs_res.append(avg_abs_mse)
    
print("relative_mse: ")
print( rel_res)

print("absolute_mse: ")
print( abs_res)