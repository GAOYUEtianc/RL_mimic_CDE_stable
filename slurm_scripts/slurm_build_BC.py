import os
import itertools

learning_rates = ['1e-5', '1e-4', '1e-3']
num_nodes = [64, 128, 256]
weight_decays = [0, 0.15, 0.25]
optim_type = ['adam','sgd']

test_array = list(itertools.product(learning_rates, num_nodes, weight_decays, optim_type))

with open('slurm_scripts/BC_exp_params.txt', 'w') as f:
    for lr, nodes, wd, optim in test_array:
        folder_name = f"BC_l{lr}_n{nodes}_w{wd}_{optim}"
        storage_path = os.path.join("data", "behaviour_clone", folder_name)
        line = f"--storage_folder {storage_path} --learning_rate {lr} --num_nodes {nodes} --weight_decay {wd} --optimizer_type {optim}\n"
        f.write(line)