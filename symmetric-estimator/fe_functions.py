import numpy as np
from estimators import uni_df_t, uni_pmf
from estimators import bi_df_t, bi_pmf
from estimators import sym_est_df_t_v1, sym_est_pmf_v1

def unidirectional_fe(pulling_data, nblocks, ntrajs_per_block,
                      timeseries_indices,
                      which_data,
                      nbootstraps=0):
    """
    Unidirectional free energy estimator

    Parameters:
    pulling_data : dict
        data from steered MD (does not have to be)
    nblocks : int
        number of blocks
    ntrajs_per_block : int
        number of trajectories per block
    timeseries_indices : 1d array
        indices of timeseries
    which_data : str
        "F" or "R"  
    nbootstraps : int
        number of bootstraps

    Returns:
    free_energies : dict
        free energies
    """
    if which_data not in ["F", "R"]:
        raise ValueError("Unknown which_data")

    #if ntrajs_per_block % 2 != 0:
    #    raise ValueError("Number of trajs per block must be even")

    if which_data == "F":
        total_ntrajs_in_data = pulling_data["wF_t"].shape[0]
    elif which_data == "R":
        total_ntrajs_in_data = pulling_data["wR_t"].shape[0]

    total_ntrajs_requested = nblocks * ntrajs_per_block
    if total_ntrajs_requested  > total_ntrajs_in_data:
        raise ValueError("Number of trajs requested is too large")

    free_energies = {}
    free_energies["nblocks"] = nblocks
    free_energies["ntrajs_per_block"] = ntrajs_per_block
    free_energies["timeseries_indices"] = timeseries_indices

    free_energies["ks"] = pulling_data["ks"]
    free_energies["dt"] = pulling_data["dt"]

    if which_data == "F":
        w_t = pulling_data["wF_t"][: total_ntrajs_requested, timeseries_indices]
        lambdas = pulling_data["lambda_F"][timeseries_indices]
    elif which_data == "R":
        w_t = pulling_data["wR_t"][: total_ntrajs_requested, timeseries_indices]
        lambdas = pulling_data["lambda_R"][timeseries_indices]

    free_energies["lambdas"] = lambdas

    free_energies["main_estimates"] = {}
    for block in range(nblocks):
        left_bound = block * ntrajs_per_block
        right_bound = (block + 1) * ntrajs_per_block
        free_energies["main_estimates"]["block_%d"%block] = uni_df_t(w_t[left_bound : right_bound])

    for bootstrap in range(nbootstraps):
        free_energies["bootstrap_%d"%bootstrap] = {}
        traj_indices = np.random.choice(total_ntrajs_requested, size=total_ntrajs_requested, replace=True)
        w_t_bootstrap = w_t[traj_indices]

        for block in range(nblocks):
            left_bound = block * ntrajs_per_block
            right_bound = (block + 1) * ntrajs_per_block
            free_energies["bootstrap_%d" % bootstrap]["block_%d" % block] = uni_df_t(
                w_t_bootstrap[left_bound : right_bound])

    return free_energies

def bidirectional_fe(pulling_data, nblocks, ntrajs_per_block,
                    timeseries_indices,
                    nbootstraps=0):
    """
    Bidirectional free energy estimator

    Parameters:
    pulling_data : dict
        data from steered MD (does not have to be)
    nblocks : int
        number of blocks
    ntrajs_per_block : int
        number of trajectories per block
    timeseries_indices : 1d array
        indices of timeseries
    nbootstraps : int
        number of bootstraps

    Returns:
    free_energies : dict
        free energies
    """
    if pulling_data["wF_t"].shape[0] != pulling_data["wR_t"].shape[0]:
        raise ValueError("Forward and reverse must have the same number of trajectories")

    #if ntrajs_per_block % 2 != 0:
    #    raise ValueError("Number of trajs per block must be even")

    total_ntrajs_in_data = (pulling_data["wF_t"].shape[0] + pulling_data["wR_t"].shape[0]) // 2

    total_ntrajs_requested = nblocks * ntrajs_per_block

    if total_ntrajs_requested > total_ntrajs_in_data:
        raise ValueError("Number of trajs requested is too large")

    free_energies = {}
    free_energies["nblocks"] = nblocks
    free_energies["ntrajs_per_block"] = ntrajs_per_block
    free_energies["timeseries_indices"] = timeseries_indices

    free_energies["ks"] = pulling_data["ks"]
    free_energies["dt"] = pulling_data["dt"]
    free_energies["lambdas"] = pulling_data["lambda_F"][timeseries_indices]

    wF_t = pulling_data["wF_t"][: total_ntrajs_requested // 2, timeseries_indices]
    wR_t = pulling_data["wR_t"][: total_ntrajs_requested // 2, timeseries_indices]

    free_energies["main_estimates"] = {}
    for block in range(nblocks):
        left_bound = block * (ntrajs_per_block // 2)
        right_bound = (block + 1) * (ntrajs_per_block // 2)
        free_energies["main_estimates"]["block_%d" % block] = bi_df_t(wF_t[left_bound : right_bound],
                                                                      wR_t[left_bound : right_bound])

        for bootstrap in range(nbootstraps):
            free_energies["bootstrap_%d" % bootstrap] = {}
            traj_indices = np.random.choice(total_ntrajs_requested // 2,
                                            size=total_ntrajs_requested // 2, replace=True)
            wF_t_bootstrap = wF_t[traj_indices]
            wR_t_bootstrap = wR_t[traj_indices]

            for block in range(nblocks):
                left_bound = block * (ntrajs_per_block // 2)
                right_bound = (block + 1) * (ntrajs_per_block // 2)
                free_energies["bootstrap_%d" % bootstrap]["block_%d" % block] = bi_df_t(
                    wF_t_bootstrap[left_bound : right_bound],
                    wR_t_bootstrap[left_bound : right_bound])

    return free_energies

def symmetric_fe(pulling_data, nblocks, ntrajs_per_block,
                      timeseries_indices,
                      nbootstraps=0):
    """
    Symmetric free energy estimator

    Parameters:
    pulling_data : dict
        data from steered MD (does not have to be)
    nblocks : int
        number of blocks
    ntrajs_per_block : int
        number of trajectories per block
    timeseries_indices : 1d array
        indices of timeseries
    nbootstraps : int
        number of bootstraps

    Returns:
    free_energies : dict
        free energies
    """
    total_ntrajs_in_data = pulling_data["wF_t"].shape[0]

    total_ntrajs_requested = nblocks * ntrajs_per_block
    if total_ntrajs_requested > total_ntrajs_in_data:
        raise ValueError("Number of trajs requested is too large")

    free_energies = {}
    free_energies["nblocks"] = nblocks
    free_energies["ntrajs_per_block"] = ntrajs_per_block
    free_energies["timeseries_indices"] = timeseries_indices

    free_energies["ks"] = pulling_data["ks"]
    free_energies["dt"] = pulling_data["dt"]

    w_t = pulling_data["wF_t"][: total_ntrajs_requested, timeseries_indices]
    lambdas = pulling_data["lambda_F"][timeseries_indices]
    free_energies["lambdas"] = lambdas

    free_energies["main_estimates"] = {}
    for block in range(nblocks):
        left_bound = block * ntrajs_per_block
        right_bound = (block + 1) * ntrajs_per_block
        free_energies["main_estimates"]["block_%d" % block] = sym_est_df_t_v1(w_t[left_bound: right_bound])

    for bootstrap in range(nbootstraps):
        free_energies["bootstrap_%d"%bootstrap] = {}
        traj_indices = np.random.choice(total_ntrajs_requested, size=total_ntrajs_requested, replace=True)
        w_t_bootstrap = w_t[traj_indices]

        for block in range(nblocks):
            left_bound = block * ntrajs_per_block
            right_bound = (block + 1) * ntrajs_per_block
            free_energies["bootstrap_%d" % bootstrap]["block_%d" % block] = sym_est_df_t_v1(
                w_t_bootstrap[left_bound : right_bound])

    return free_energies