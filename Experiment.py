import yaml
import numpy as np
from tqdm import tqdm
from utils.utils import save_array_to_npy
from dataset.sim_1d_no_x import Sim1d_noX
from dataset.high_dimension import High_dim
from model.rkhs.Trainer import RKHS_Trainer


def load_hyperparameters(filepath):
    """
    Load hyperparameters from a YAML file.

    Arg:
        - filepath (str): The path to the YAML file containing hyperparameters.

    Returns:
        - dict: A dictionary of hyperparameters.
    """
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)


def generate_datasets(cfg, seed):
    """
    Generate treatment, target, training, and testing datasets based on the configuration.

    Args:
        - cfg (object): Configuration object containing dataset and sample details.
        - seed (int): Random seed for reproducibility.

    Returns:
        - treatment (np.array): Array of treatment effects.
        - tar (np.array): Array of target values (ground truth).
        - train_dataset (tuple): Training dataset.
        - test_dataset (tuple): Testing dataset.
    """
    if cfg.dataset == 'sim1d_no_x':
        treatment, tar = Sim1d_noX.generate_test_effect(-1, 2, 100)
        train_dataset = Sim1d_noX(seed, cfg.sample).generatate_sim(
            W_miss=False, Z_miss=False)
        test_dataset = Sim1d_noX.generate_test(1000, seed + 1)
    elif cfg.dataset == 'high':
        treatment, tar = High_dim.generate_test_effect(
            0, 1, 100, 'quardratic', 10, 10, 100)
        high = High_dim(seed, cfg.sample, dim_z=10, dim_w=10, dim_x=100)
        train_dataset = high.generatate_high(False)
        test_dataset = high.generate_test(1000, seed + 1, False)
    return treatment, tar, train_dataset, test_dataset


def fit_and_test_model(cfg, treatment, train_dataset, test_dataset, hyperparams):
    """
    Fit the RKHS model using cross-validation and evaluate it.

    Args:
        - cfg (object): Configuration object containing dataset and model details.
        - treatment (np.array): Array of treatment effects.
        - train_dataset (tuple): Training dataset.
        - test_dataset (tuple): Testing dataset.
        - hyperparams (dict): Dictionary of hyperparameters for model training.

    Returns:
        - ATE_h (float): Estimated Average Treatment Effect using h-test.
        - ATE_q (float): Estimated Average Treatment Effect using q-test.
        - ATE_dr (float): Estimated Average Treatment Effect using doubly robust test.
    """
    rkhs_train = RKHS_Trainer(train_dataset, **hyperparams)
    rkhs_train.fit_h_cv()
    ATE_h = rkhs_train._htest(treatment, test_dataset)

    if cfg.dataset == 'sim1d_no_x':
        rkhs_train.fit_q_cv(type='kde')
    elif cfg.dataset == 'high':
        rkhs_train.fit_q_cv(type='cnf', cnf=cfg.cnf)

    ATE_q = rkhs_train._qtest(treatment, test_dataset)
    ATE_dr = rkhs_train._drtest(treatment, test_dataset)

    return ATE_h, ATE_q, ATE_dr


def RKHS_Experiment(cfg):
    """
    Run the RKHS experiment over multiple seeds and save the results.

    Args:
        - cfg (object): Configuration object containing experiment settings.

    Returns:
        - ATE_h_list (list): List of ATE estimates using h-test across all seeds.
        - ATE_q_list (list): List of ATE estimates using q-test across all seeds.
        - ATE_dr_list (list): List of ATE estimates using doubly robust test across all seeds.
        - tar (np.array): Array of target values (ground truth).
    """
    seeds = np.random.randint(1000, 10000, size=20)

    ATE_h_list, ATE_q_list, ATE_dr_list = [], [], []
    hyperparams = load_hyperparameters(cfg.rkhs)

    for seed in tqdm(seeds):
        treatment, tar, train_dataset, test_dataset = generate_datasets(
            cfg, seed)
        ATE_h, ATE_q, ATE_dr = fit_and_test_model(
            cfg, treatment, train_dataset, test_dataset, hyperparams)

        ATE_h_list.append(ATE_h)
        ATE_q_list.append(ATE_q)
        ATE_dr_list.append(ATE_dr)

    save_array_to_npy(tar, "Ground_Truth", cfg)
    save_array_to_npy(np.array(ATE_h_list), "ATE_h", cfg)
    save_array_to_npy(np.array(ATE_q_list), "ATE_q", cfg)
    save_array_to_npy(np.array(ATE_dr_list), "ATE_dr", cfg)

    return ATE_h_list, ATE_q_list, ATE_dr_list, tar
