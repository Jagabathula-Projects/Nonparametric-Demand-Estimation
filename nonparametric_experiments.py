import sys
from mixed_logit_estimators import FrankWolfeMixedLogitBLPEstimator
# from collections import defaultdict
from constants import *
import pickle
from numpy.random import RandomState
import logging
from IPython import embed
import pandas as pd
from scipy.linalg import eigh, svd, sqrtm
from copy import deepcopy
import itertools


# setup logger environment
logger = logging.getLogger('Simulations')
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

prng = RandomState(1244)

def compute_variables_from_expt_file(file_name, no_purchase_const, xi_bar_in_constraint, num_feats):
    data_df = pd.read_csv(file_path_data + file_name+'.csv')
    observed_ms = data_df['shares'].values.reshape((num_markets, -1))
    nopurch_shares = 1 - np.sum(observed_ms, 1, keepdims=True)
    observed_ms = np.hstack((nopurch_shares, observed_ms))
    prices_offered = np.hstack((NOPURCHSHOCKS, data_df['prices'].values.reshape((num_markets, -1))))
    prod_feats_offered = []
    N = num_markets * num_prods
    add_constant_instr = int(no_purchase_const > 0)
    num_total_instr = len(data_df.columns) - 4 + add_constant_instr # 4 corresponds to the first 4 columns like market_id, product_id,  etc.
    more_instruments = np.ones((N, num_total_instr))
    num_zs = more_instruments.shape[1] - (num_feats - 1 + add_constant_instr)
    for instr_id in range(num_zs):
        more_instruments[:, instr_id] = data_df['demand_instruments{0}'.format(instr_id)]
    for exo_feat in range(num_feats - 1):
        x_offered = np.hstack((NOPURCHSHOCKS, data_df['x_{0}'.format(exo_feat)].values.reshape((num_markets, -1))))
        prod_feats_offered.append(x_offered)
        more_instruments[:, num_zs + add_constant_instr + exo_feat] = data_df['x_{0}'.format(exo_feat)]

    if add_constant_instr > 0:
        constant_feature = np.ones_like(observed_ms)
        constant_feature[:, 0] = 0
        prod_feats_offered.insert(0, constant_feature)

    W_more = np.linalg.inv(np.dot(more_instruments.T, more_instruments / N))  # make same as pyBLP
    more_instruments /= N
    prod_feats_offered.append(prices_offered)

    return observed_ms, prod_feats_offered, W_more, more_instruments


def run_nonparametric_estimator():
    n = num_prods
    run_times = dict()
    membership = np.ones((num_markets, n + 1), dtype=np.int)
    # logger.info('Starting experiment {0} for T={1} and J={2}'.format(exp_iter, num_markets, num_prods))
    N = num_markets * n
    add_constant_instr = int(no_purch_const > 0)
    observed_ms, prod_feats_offered, W, instruments = compute_variables_from_expt_file(
        file_name, no_purch_const, XI_ZERO_MEAN, num_feats)
    eigvals, eigenvectors = eigh(W)
    mean_correct_matrix = np.eye(N) - np.ones((N, N)) / N
    if XI_ZERO_MEAN:
        svd_matrix = np.linalg.multi_dot(
            [mean_correct_matrix.T, instruments, eigenvectors, np.sqrt(np.diag(eigvals)), eigenvectors.T])
    else:
        svd_matrix = np.linalg.multi_dot([instruments, eigenvectors, np.sqrt(np.diag(eigvals)), eigenvectors.T])
    U, svd_sigmas, Vh = svd(svd_matrix)
    gmm_matrix = np.dot(svd_matrix, svd_matrix.T)
    num_non_zero_eigvals = instruments.shape[1] - (add_constant_instr * XI_ZERO_MEAN)
    zero_eigenvectors = U[:, num_non_zero_eigvals:]

    init_mix_props = init_coefs = init_demand_shocks = None

    if ALGO_VARIANT.startswith('IgnoreUF'):
        fw_blp_est = FrankWolfeMixedLogitBLPEstimator(LIKELIHOOD, 'corrective', 'ignore')
        expt_id_iter = 'Results_no_xi_'+file_name
        
        fw_blp_est.fit_to_choice_data(membership, prod_feats_offered, observed_ms, gmm_matrix, instruments, W, num_iterations_IgnoreUF, 0,
                                      expt_id_iter, init_coefs, init_demand_shocks, init_mix_props)

    if ALGO_VARIANT.startswith('AltDesc'):
        print('Starting run for G0 = {0} xi_per = {1}'.format(g0, xi_per))

        fw_blp_est = FrankWolfeMixedLogitBLPEstimator(LIKELIHOOD, 'corrective_{0}'.format(NUM_BETA_UPDATES),
                                                        'exact', gmm_obj_ub=g0, norm_ub_type=NORM_TYPE,
                                                        num_xi_updates=NUM_XI_UPDATES, fixed_coef_ub=xi_per,
                                                        do_away_steps=False)
        id_term = 'swap'

        fw_blp_est.non_zero_eigenvalues = svd_sigmas[:num_non_zero_eigvals]
        fw_blp_est.non_zero_eigenvectors = U[:, :num_non_zero_eigvals].T
        fw_blp_est.zero_eigenvectors = U[:, num_non_zero_eigvals:].T

        expt_id_iter = 'Results_'+file_name+'_G0={0}_xi_per={1}_'.format(g0, xi_per)
        expt_id_iter += id_term

        fw_blp_est.fit_to_choice_data(membership, prod_feats_offered, observed_ms, gmm_matrix, instruments, W, num_iterations_AltDesc, 0,
                                        expt_id_iter, init_coefs, init_demand_shocks, init_mix_props)



def compute_metrics():         
    train_shares, train_prod_feats_offered, W_more, instruments = compute_variables_from_expt_file(
        file_name, no_purch_const, XI_ZERO_MEAN, num_feats)  
    membership = np.ones((num_markets, num_prods + 1), dtype=np.int)  
    FEATURE_INDEX = -1
    N = num_markets*num_prods

    if ALGO_VARIANT.startswith('IgnoreUF'):
        exp_file = file_path_results + f'Results_no_xi_{file_name}_BLPestimator_variant=corrective_subprob_type=ignore_away_xi_steps=False_iter=0_R={num_FW_iterations}.stats'

    if ALGO_VARIANT.startswith('AltDesc'):
        exp_file = file_path_results+f'Results_{file_name}_G0={g0}_xi_per={xi_per}__swap_BLPestimator_variant=corrective_1_subprob_type=exact_away_xi_steps=False_iter=0_R={num_FW_iterations}.stats'
        
    mix_props_iter, coefs_iter, gmm_obj_iter, kldiv_iter, demand_shocks_iter = pickle.load(open(exp_file, 'rb'))
        
    # reuse estimator object from before
    zetas_iter = np.ravel(demand_shocks_iter[:,1:])
    # zetas_iter_center = zetas_iter - np.mean(zetas_iter)
    zetas_instr = np.dot(zetas_iter, instruments)
    gmm_obj_iter_computed = np.linalg.multi_dot([zetas_instr, W_more, zetas_instr])

    xi_per_computed = np.ravel(demand_shocks_iter[:,1:]) - np.linalg.multi_dot([instruments, instruments.T, np.ravel(demand_shocks_iter[:,1:])])
    xi_per_constraint_computed = np.dot(xi_per_computed.T, xi_per_computed) / N


    dummy_est = FrankWolfeMixedLogitBLPEstimator(LIKELIHOOD, 'corrective')
    dummy_est.mix_props = mix_props_iter
    dummy_est.coefs_ = coefs_iter[:, np.newaxis, :]
    dummy_est.demand_shocks = demand_shocks_iter

    #train price elasticities
    price_elas_train = dummy_est.compute_price_elasticities(membership, train_prod_feats_offered, FEATURE_INDEX)
    own_price_elas_train = price_elas_train[range(N), list(range(num_prods))*num_markets]
    train_pred_shares = dummy_est.predict_proba(membership, train_prod_feats_offered)

    ############### test market share prediction at 1% price change
    test_prod_feats_offered_1 = deepcopy(train_prod_feats_offered)
    test_prices_offered_1 = test_prod_feats_offered_1[-1]
    # increase price of randomly chosen product in each market
    test_prices_offered_1[range(num_markets), prng.randint(1, num_prods + 1, size=num_markets)] *= (
            1 + PRICE_CHANGE_PERCENT_1)

    test_pred_shares_1 = dummy_est.predict_proba(membership, test_prod_feats_offered_1)

    ############### test market share prediction at 5% price change
    test_prod_feats_offered_2 = deepcopy(train_prod_feats_offered)
    test_prices_offered_2 = test_prod_feats_offered_2[-1]
    # increase price of randomly chosen product in each market
    test_prices_offered_2[range(num_markets), prng.randint(1, num_prods + 1, size=num_markets)] *= (
            1 + PRICE_CHANGE_PERCENT_2)

    test_pred_shares_2 = dummy_est.predict_proba(membership, test_prod_feats_offered_2)
    
    ############### test market share prediction at 10% price change
    test_prod_feats_offered_3 = deepcopy(train_prod_feats_offered)
    test_prices_offered_3 = test_prod_feats_offered_3[-1]
    # increase price of randomly chosen product in each market
    test_prices_offered_3[range(num_markets), prng.randint(1, num_prods + 1, size=num_markets)] *= (
            1 + PRICE_CHANGE_PERCENT_3)

    test_pred_shares_3 = dummy_est.predict_proba(membership, test_prod_feats_offered_3)

    return price_elas_train, own_price_elas_train, train_pred_shares, test_pred_shares_1, test_pred_shares_2, test_pred_shares_3
    


if __name__ == "__main__":        
    run_nonparametric_estimator()
    price_elas_train, own_price_elas_train, train_pred_shares, test_pred_shares_1, test_pred_shares_2, test_pred_shares_3 = compute_metrics()


