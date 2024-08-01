from config import config
from Experiment import RKHS_Experiment
from utils.utils import set_random_seed, causal_mse, compute_mean_and_variance


def main():
    # Load configuration
    cfg = config()

    # Set the random seed for reproducibility
    set_random_seed(cfg.random_seed)

    # Run RKHS experiment and obtain results
    ATE_h_list, ATE_q_list, ATE_dr_list, tar = RKHS_Experiment(cfg)

    # Calculate mean squared errors (MSE) for each method
    methods = {'ATE_h': ATE_h_list, 'ATE_q': ATE_q_list, 'ATE_dr': ATE_dr_list}
    results = {}

    for key, pre_list in methods.items():
        mse = causal_mse(pre_list, tar)
        mean, var = compute_mean_and_variance(mse)
        results[key] = (mean, var)

    # Print results
    for key, (mean, var) in results.items():
        print(f"{key}: {mean:.4f} +/- {var:.4f}")


if __name__ == "__main__":
    main()
