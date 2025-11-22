
# RL_mimic_CDE_stable

This repository provides the implementation for the paper **Stable CDE Autoencoders with Acuity Regularization for Offline Reinforcement Learning in Sepsis Treatment** 
(https://arxiv.org/abs/2506.15019), accepted to **IEEE Transactions on Artificial Intelligence**.
If you use this code in your project, please cite 
> @misc{gao2025stablecdeautoencodersacuity,
      title={Stable CDE Autoencoders with Acuity Regularization for Offline Reinforcement Learning in Sepsis Treatment}, 
      author={Yue Gao},
      year={2025},
      eprint={2506.15019},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.15019}, 
}

A stable training framework 
**with early stopping mechanism and multiple stabilization techniques** was proposed for learning clinically meaningful state representations from irregular ICU time series using Neural Controlled Differential Equations (CDEs), and demonstrate improved offline reinforcement learning performance on the MIMIC-III sepsis treatment cohort.

---
## 📖 Citation & Acknowledgments

This repository builds upon prior work from:

```bibtex
@inproceedings{killian2020empirical,
  title={An Empirical Study of Representation Learning for Reinforcement Learning in Healthcare},
  author={Killian, Taylor W and Zhang, Haoran and Subramanian, Jayakumar and Fatemi, Mehdi and Ghassemi, Marzyeh},
  booktitle={Machine Learning for Health},
  pages={139--160},
  year={2020},
  organization={PMLR}
}
```
with [repo](https://github.com/MLforHealth/rl_representations/tree/main).

Compared to the prior work, this repo adds multiple stabilization techniques (gradient clipping, stiffness regularization, implicit adam solver) and an early stopping mechanism.

## 🔧 Project Structure

RL_mimic_CDE_stable/

|── configs/ # Experiment and model configuration files

|── scripts/ # Training and evaluation scripts

|── slurm_scripts/ # SLURM job submission templates (for HPC)

|── environment.yml # Conda environment file

|── requirements.txt # Python dependencies

|── README.md

|── process_mimic_data # preprocess MIMIC-III data

---

## 🧪 Replicating the Experiments

Below is a step by step manual to re-implement the experiment pipeline.

### 0️⃣ Environment Setup

Recommend using Python 3.8+ in a virtual environment. To install dependencies:

```bash
# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

---

### 1️⃣ Data Processing

The [**MIMIC-III v1.4** ICU dataset](https://physionet.org/content/mimiciii/1.4/) (Johnson, A., Pollard, T., & Mark, R. (2016). MIMIC-III Clinical Database (version 1.4). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/C2XW26) was used to extract a cohort of septic patients based on Sepsis-3 criteria. The data extraction and cohort definition process builds on the open-source pipeline from [microsoft/mimic_sepsis](https://github.com/microsoft/mimic_sepsis/tree/main), with the following key modifications:

- ✅ **Google BigQuery integration**: We replaced the original Postgres-based data access with queries using **Google Cloud BigQuery**, making it easier to use cloud-hosted MIMIC datasets.
- ✅ **Bug fixes and minor revisions**: Several small issues in the original scripts were fixed to ensure compatibility and clean execution across environments.

The adapted code is located in the `process_mimic_data/` folder. 
```bash
cd process_mimic_data
```

It follows a two-step process:

1. Extracts relevant bigquery tables from MIMIC-III into intermediate CSVs.
    ```bash
    python preprocess.py
    ```
    Running will be taking > 1 hour
2. Constructs the final sepsis cohort using Sepsis-3 rules and saves the patient trajectories.
    ```bash
    python sepsis_cohort.py
    ```
    Running will be taking ~2 hour

    > ⚠️ **Note**: To run these scripts, you must have access to MIMIC-III and ensure the required reference files are available (`Reflabs.tsv`, `Refvitals.tsv`, `sample_and_hold.csv` under `process_mimic_data/ReferenceFiles`).

    The 3 resulting csv files will be saved in `data/sepsis_mimiciii/` 
3. Go back to root dir
    ```bash
    cd ..
    ```
    - Then Run ```scripts/compute_acuity_scores.py``` to compute additional acuity scores with the raw features.
    - Run ```scripts/split_sepsis_cohort.py``` to compose a training, and validation split on cohorts, and organize the patient data into trajectory formats.

---

### 2️⃣ Behaviour Cloning 
A behaviour policy will be used in both dBCQ training (as an action elimination baseline) and WIS evaluation. Behaviour cloning is trained using a 2 layer fully connected neural network.

1. Run ```scripts/create_buffers.py``` to create replay buffers. The saving location is setup in ```configs/config_behavCloning.yaml``` in train_buffer and val_buffer fields.

2. Run ```python slurm_scripts/slurm_build_BC.py``` to create args lines into ```slurm_scripts/slurm_bc_exp```.
3. Run ```bash slurm_scripts/slurm_bc_exp ```, this will automatically run ```scripts/train_behavCloning_with_command_line_args.py``` with each args line inside ```slurm_scripts/slurm_bc_exp```. As a result, behaviour cloning policy folders will be saved under ```data/behaviour_clone/```.
4. After getting all behaviour cloning policies under ```data/behaviour_clone/```, you can use a notebook to compare their validation losses to pick the best performing behaviour cloning policy, and save its path into ```configs/common.yaml``` inside behav_policy_file field to be used in later dBCQ training.

Note that as an alternative, you can also run ```scripts/train_behavCloning_with_config_file.py``` to train behaviour cloning policies, however, that will need you to update ```configs/config_behavCloning.yaml``` to include the parameters each time you run.

---

### 3️⃣ Learning CDE State Representations
1. Run ```python slurm_scripts/slurm_build_cde.py```, this will generate ```slurm_scripts/cde_exp_params.txt``` including arguments used to train cde autoencoders.
2. To train state representations, you can run ```bash slurm_scripts/slurm_cde_exp```, this will directly run the batch of experiments with different args. However, note that for different train-test split, different behaviour cloning policy, or different hidden size, the most proper early stopping criteria thresholds might be different. They can be adjusted in ```scripts/experiment.py```:
    ```bash
    self.epsilon_1 = 0.05  # near-optimal val loss tolerance for early stopping criteria
    self.epsilon_2 = 0.08  # plateau variation tolerance for early stopping criteria
    self.plateau_epochs = 20
    self.check_early_stop_epochs = 60 # Number of epochs to check for early stopping criteria
    self.corr_threshold = 0.6  # threshold for correlation loss to be considered significant    
    ```
3. The state representation files including evaluation results and losses will be saved under ```test/``` folder.
---
### 4️⃣ Offline RL Policy Learning and Evaluation
RL policies are learned using the discretized form of [Batch Constrained Q-learning](https://github.com/sfujim/BCQ) from [Fujimoto, et al (2019)](https://arxiv.org/abs/1910.01708). These policies are learned as the final part of the previous step. 
If you want to train autoencoders and RL policies separately, you can go to `scripts/train_model` and comment out `experiment.train_dBCQ_policy(params['pol_learning_rate'])` and add it back later.
The policies are evaluated intermittently using a form of Weighted Importance Sampling, which is a commonly used evaluation in offline RL. 

---
## 👤 Authors

**Yue Gao**  
[GitHub Profile](https://github.com/GAOYUEtianc)  
If you have any questions, please don't hesitate to reach out via [email](mailto:gao12@ualberta.ca)

---

## 📄 License

This project is licensed under the [MIT License]((https://opensource.org/licenses/MIT)).
