import argparse


def config():
    parser = argparse.ArgumentParser(
        description='Proximal causal inference for continuous treatment')

    parser.add_argument('--dataset', type=str, default="sim1d_no_x",  choices=["sim1d_no_x", "high"],
                        help='Specify the dataset to use. Options are "sim1d_no_x" for 1D simulated data without covariates and "high" for high-dimensional data.')
    parser.add_argument('--sample', type=int, default=1000,
                        help='Number of samples to generate or use from the dataset.')
    parser.add_argument('--random_seed', type=int, default=123456,
                        help='Random seed for reproducibility of experiments.')
    parser.add_argument('--model', type=str, default="rkhs",
                        help='Model type to use for analysis.')
    parser.add_argument('--rkhs', type=str, default="configs/rkhs.yaml",
                        help='Path to the configuration file for the RKHS model.')
    parser.add_argument('--cnf', type=str, default="configs/cnf.yaml",
                        help='Path to the configuration file for the conditional normalizing flow (CNF) model.')
    parser.add_argument('--ATE', type=str, default="ATE",
                        help='Path to the outcomes for estimating the Average Treatment Effect (ATE).')
    args = parser.parse_args()

    return args
