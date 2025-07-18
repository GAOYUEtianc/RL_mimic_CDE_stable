import os
import itertools

latent_dims = [4,8,16,32,64,128,256]
rand_seeds = [25, 32, 1234, 2020, 53]
include_demos = [False] # You can change it to [True, False] if you want to include cases with demographic features
corr_coeff_params = [0, 1]
stabilizations = ['no_stabilization', 'gradient_clipping', 'l2_regularization', 'implicit_adam']
early_stoppings = [True, False]
        
test_array = list(itertools.product(latent_dims, rand_seeds, include_demos, corr_coeff_params, stabilizations, early_stoppings))
with open('slurm_scripts/cde_exp_params.txt', 'w') as f:
    for hidden_size, random_seed, context_input, corr_coeff_param, stabilization, early_stopping in test_array:
        name = 'cde_sz{0}_rand{1}_corr{2}_context{3}_{4}_earlystop{5}_sepsis_training'.format(
            hidden_size, random_seed, corr_coeff_param, context_input, stabilization, early_stopping)
        f.writelines('--autoencoder CDE --domain sepsis -o folder_name {0} -o hidden_size {1} -o random_seed {2} -o context_input {3} -o corr_coeff_param {4} -o stabilization {5} -o earlystop {6}\n'.format(
            name, hidden_size, random_seed, context_input, corr_coeff_param, stabilization, early_stopping))
