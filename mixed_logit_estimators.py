from scipy.optimize import minimize, fminbound, minimize_scalar, lsq_linear
import multiprocessing as mp
from IPython import embed
from functools import partial
import logging
from constants import *
from numpy.random import RandomState
import pickle
from scipy.stats import multivariate_normal

# setup logger environment
logger = logging.getLogger('MixedLogitEstimators')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class LearnDistr(object):
    """
    Generic class representing different mixture distribution estimators
    """
    def __init__(self, loss, base_class, product_features=None, covariance=None):
        self.loss = loss
        self.base_class = base_class
        if product_features is not None:
            self.product_features = np.copy(product_features)
            self.num_cont_features = product_features.shape[1]
            self.num_products = product_features.shape[0]
        if covariance is not None:
            self.covariance = covariance

        # estimation time
        self.estimation_time = -1

    def _negative_log_likelihood(self, xk, n_counts):
        return -np.mean(n_counts*np.where(n_counts > 0, np.log(xk), 0))

    def _gradient_likelihood(self, xk, n_counts):
        # return (-1.*n_counts)/np.where(xk > 0, xk, 1)
        # normalizing the gradient makes it harder to find descent direction, so leave it unnormalized
        return (-1.*n_counts)/np.where(n_counts > 0, xk, 1)

    def compute_optimization_objective(self, xk, n_counts, aux_info=None):
        if self.loss == 'quadratic':
            if aux_info is not None:
                return np.sum((xk*aux_info - n_counts)**2)/2
            else:
                return np.sum((xk - n_counts)**2)/2
        elif self.loss == 'likelihood':
            return self._negative_log_likelihood(xk, n_counts)

    def compute_objective_gradient(self, xk, n_counts, aux_info=None):
        if self.loss == 'quadratic':
            if aux_info is not None:
                return (aux_info*xk - n_counts)*aux_info
            else:
                return xk - n_counts
        elif self.loss == 'likelihood':
            return self._gradient_likelihood(xk, n_counts)

    def compute_regularized_objective(self, xk, n_counts, alpha, aux_info=None):
        obj = self.compute_optimization_objective(xk[:-1], n_counts, aux_info)
        return obj + alpha*xk[-1]

    def compute_regularized_objective_gradient(self, xk, n_counts, alpha, aux_info=None):
        grad_obj = self.compute_objective_gradient(xk[:-1], n_counts, aux_info)
        return np.append(grad_obj, alpha)

    def _predict_MNL_proba(self, X_agg, MNL_param_vec, prod_feats_to_use=None):
        assert np.all(np.isfinite(MNL_param_vec)), 'Some MNL coefs have infinite value!'
        if prod_feats_to_use is None:
            prod_utils = np.dot(self.product_features, MNL_param_vec)
        else:
            prod_utils = np.dot(prod_feats_to_use, MNL_param_vec)
        prod_utils -= np.max(prod_utils)
        prod_wt = np.exp(prod_utils)
        # compute unnormalized product weights
        probs = X_agg*prod_wt
        row_sums = np.sum(probs, 1)
        non_empty_assorts = (row_sums != 0)
        row_sums[~non_empty_assorts] = 1
        # compute choice probabilities
        probs = probs/row_sums[:, np.newaxis]
        assert np.all(np.around(np.sum(probs[non_empty_assorts], 1) - np.ones(np.sum(non_empty_assorts)), 7) == 0)
        # assert (np.all(np.around(np.sum(probs, 1) - np.ones(X_agg.shape[0]), 7) == 0))
        return probs

    def _predict_Bernoulli_proba(self, X_obs, params, loga=False):
        if loga:
            return np.sum(np.log(params)*X_obs + np.log(1-params)*(1-X_obs), axis=1)
        else:
            return np.prod(params*X_obs + (1-params)*(1-X_obs), axis=1)

    def _predict_Gaussian_proba(self, X_obs, params, loga=False):
        if loga:
            return multivariate_normal.logpdf(X_obs, mean=params, cov=self.covariance)
        else:
            return multivariate_normal.pdf(X_obs, mean=params, cov=self.covariance)

    def _predict_Gaussian_cdf(self, X_obs, params):
        return multivariate_normal.cdf(X_obs, mean=params, cov=self.covariance)

    @staticmethod
    def evaluate_choice_predictions(n_obs_agg, n_pred_agg, metric, prediction_indicators=None):
        if metric == 'Chisquare':
            all_preds = (n_obs_agg - n_pred_agg)**2/n_pred_agg
        elif metric == 'RMSE':
            all_preds = (n_obs_agg-n_pred_agg)**2
        elif metric == 'MAPE':
            all_preds = np.abs((n_obs_agg - n_pred_agg)/n_obs_agg)
        elif metric == 'NLL':
            all_preds = -n_obs_agg*np.log(n_pred_agg)
            
        if prediction_indicators is not None:
            error_preds = all_preds[prediction_indicators]
        else:
            error_preds = all_preds

        if metric == 'RMSE':
            return np.sqrt(np.nanmean(error_preds))
        else:
            return np.nanmean(error_preds)

    def penalty_function(self, x0):
        if self.reg_type == 'l2':
            return .5*np.linalg.norm(x0)**2
        elif self.reg_type == 'l1':
            return np.linalg.norm(x0, 1)

    def gradient_penalty_function(self, x0):
        if self.reg_type == 'l2':
            return x0
        elif self.reg_type == 'l1':
            return np.sign(x0)


class FrankWolfeMixedLogitBLPEstimator(LearnDistr):
    """
    """

    def __init__(self, loss, fwVariant, subprob_type='exact', learning_rate=1.0, num_subprob_init=NUM_BFGS_INIT, gmm_obj_ub=1.0, norm_ub_type='l2', num_xi_updates=0, fixed_coef_ub=1.0, do_away_steps=False):
        """
        """
        # initialize params for base estimator class
        LearnDistr.__init__(self, loss, 'MNL')
        # determine which frank-wolfe variant to run
        self.fwVariant = fwVariant

        # the learning rate
        self.learning_rate = learning_rate

        # number of random points to run BFGS from in subproblems
        self.num_subprob_init = num_subprob_init

        # method used to solve subproblem in each iteration
        self.subprob_type = subprob_type

        # MNL coefficients for different classes (K x d)
        self.coefs_ = None
        # proportions of different classes
        self.mix_props = None
        # \zeta_{jt} terms in BLP utility
        self.demand_shocks = None
        # sequence of demand shocks recovered
        self.all_demand_shocks = None
        # convex combination weights that self.demand_shocks satisfies w.r.t conv(self.all_demand_shocks)
        self.demand_shocks_weights = None
        # market shares by component
        self.component_market_shares = None
        # number of iterations
        self.num_iterations = 0

        self.gmm_obj_ub = gmm_obj_ub
        self.norm_ub_type = norm_ub_type
        self.fixed_coef_ub = fixed_coef_ub

        # 3 x JT matrix with each row an eigenvector
        self.non_zero_eigenvectors = None
        # (JT - 3) x JT matrix
        self.zero_eigenvectors = None
        # square roots of the non-zero eigenvalues
        self.non_zero_eigenvalues = None
        # number of xi updates in each iteration
        self.num_xi_updates = num_xi_updates
        # whether to do away FW steps when updating xi
        self.do_away_steps = do_away_steps

        # zero padding for no-purchase option
        self.nopurch_shocks = None
        

    def _predict_MNL_proba_blp(self, membership, prod_feats, coefs, demand_shocks):
        prod_utils = np.zeros_like(membership, dtype=np.float)
        coefs_2d = np.atleast_2d(coefs)
        # embed()
        for feat_d, prod_feats_d in enumerate(prod_feats):
            prod_utils += coefs_2d[:, [feat_d]] * prod_feats_d
        prod_utils += demand_shocks
        prod_utils -= np.max(prod_utils, axis=1)[:, np.newaxis]
        prod_wt = np.exp(prod_utils)
        probs = membership * prod_wt
        row_sums = np.sum(probs, 1)
        non_empty_assorts = (row_sums != 0)
        row_sums[~non_empty_assorts] = 1
        probs = probs / row_sums[:, np.newaxis]
        assert np.all(np.around(np.sum(probs[non_empty_assorts], 1) - np.ones(np.sum(non_empty_assorts))) == 0)
        return probs

    def set_component_market_shares(self, X_obs, F_obs):
        num_offersets, num_prods = X_obs.shape
        num_support = self.mix_props.shape[0]
        curr_probs = np.zeros_like(X_obs, dtype=np.float)
        self.component_market_shares = np.zeros((num_support, num_offersets, num_prods))
        for k in range(num_support):
            self.component_market_shares[k] = self._predict_MNL_proba_blp(X_obs, F_obs, self.coefs_[k], self.demand_shocks)
            curr_probs += self.mix_props[k]*self.component_market_shares[k]

        return curr_probs

    # predict choice probabilities of chosen products under current mixture parameters
    def predict_choice_proba(self, mixture_weights, mixture_params, membership, prod_feats_offered, demand_shocks):
        probs = np.zeros_like(membership, dtype=np.float)
        # mix_props is now T x K, that is we allow for different proportions in different markets
        mix_props_2d = np.atleast_2d(mixture_weights)
        num_support = mix_props_2d.shape[1]
        num_offersets, num_prods = membership.shape
        component_market_shares = np.zeros((num_support, num_offersets, num_prods))
        for k in range(num_support):
            component_market_shares[k] = self._predict_MNL_proba_blp(membership, prod_feats_offered, mixture_params[k], demand_shocks)
            probs += mix_props_2d[:, [k]] * component_market_shares[k]

        return probs, component_market_shares

    # external function for consistent usage
    def predict_proba(self, membership, prod_feats_offered):
        choice_probs = np.zeros_like(membership, dtype=np.float)
        # mix_props is now T x K, that is we allow for different proportions in different markets
        # embed()
        mix_props_2d = np.atleast_2d(self.mix_props)
        # embed()
        for k in range(mix_props_2d.shape[1]):
            choice_probs += mix_props_2d[:, [k]] * self._predict_MNL_proba_blp(membership, prod_feats_offered, self.coefs_[k], self.demand_shocks)

        return choice_probs

    def compute_own_price_elasticities(self, membership, features, feature_index=-1):
        prices = features[feature_index]
        # mix_props is now T x K, that is we allow for different proportions in different markets
        mix_props_2d = np.atleast_2d(self.mix_props)
        curr_probs, component_market_shares = self.predict_choice_proba(mix_props_2d, self.coefs_, membership, features, self.demand_shocks)
        skjt = component_market_shares[:, :, 1:]
        # mix_props_tensor = self.mix_props[:, np.newaxis, np.newaxis]
        mix_props_tensor = mix_props_2d.T[:, :, np.newaxis]
        est_price_betas = mix_props_tensor*self.coefs_[:, :, [feature_index]]
        dshare_price = np.sum(est_price_betas*skjt*(1-skjt), axis=0)
        est_prod_shares = np.sum(mix_props_tensor*skjt, 0)
        est_own_price_elast = (prices[:, 1:]/est_prod_shares)*dshare_price
        # est_own_price_elast = (prices[:, 1:]/observed_ms[:, 1:])*dshare_price
        logger.debug('Estimated own-price elasticities: %s', est_own_price_elast)
        return np.ravel(est_own_price_elast)


    def compute_price_elasticities(self, membership, features, feature_index=-1):
        # stack JxJ matrices on top of each other like PyBLP
        mix_props_2d = np.atleast_2d(self.mix_props)
        price_elasticities_by_market = np.zeros((num_prods*num_markets, num_prods))
        prices = features[feature_index][:, 1:]
        curr_probs, component_market_shares = self.predict_choice_proba(mix_props_2d, self.coefs_, membership, features, self.demand_shocks)
        skjt = component_market_shares[:, :, 1:]
        est_prod_shares = curr_probs[:, 1:]
        mix_props_tensor = mix_props_2d.T[:, :, np.newaxis]
        est_price_betas = mix_props_tensor*self.coefs_[:, :, [feature_index]]
        # est_price_betas = self.mix_props*self.coefs_[:, feature_index]
        dshare_price_own = np.sum(est_price_betas * skjt * (1 - skjt), axis=0)
        for prod in range(num_prods): # populate the columns of price_elasticities_by_market
            # compute d(sjt)/d(plt) for all j \neq l, where prod corresponds to l
            dshare_price_other = np.sum(-est_price_betas * skjt * skjt[:, :, [prod]], axis=0)
            price_elas_prod = (prices[:, [prod]]/est_prod_shares)*dshare_price_other
            price_elas_prod[:, prod] = (prices[:, prod]/est_prod_shares[:, prod])*dshare_price_own[:, prod]
            price_elasticities_by_market[:, prod] = np.ravel(price_elas_prod)

        return price_elasticities_by_market


    def compute_diversion_ratios(self, membership, features, feature_index=-1):
        diversion_ratios_by_market = np.zeros((num_prods, num_markets, num_prods))
        curr_probs, skjt = self.predict_choice_proba(self.mix_props, self.coefs_, membership, features, self.demand_shocks)
        est_price_betas = self.mix_props*self.coefs_[:, feature_index]
        dshare_price_own = np.sum(est_price_betas[:, np.newaxis, np.newaxis]*skjt*(1-skjt), axis=0)
        for prod in range(num_prods):
            dshare_price_other = np.sum(est_price_betas[:, np.newaxis, np.newaxis] * skjt * skjt[:, :, [prod + 1]], axis=0)
            div_ratios_prod = dshare_price_other/dshare_price_own[:, [prod + 1]]
            diversion_ratios_by_market[prod] = div_ratios_prod[:, 1:]
            diversion_ratios_by_market[prod, :, prod] = div_ratios_prod[:, 0]
            
        return np.mean(diversion_ratios_by_market, axis=1)


    def compute_long_run_diversion_ratios(self, membership, features):
        diversion_ratios_by_market = np.zeros((num_prods, num_markets, num_prods))
        curr_probs, _ = self.predict_choice_proba(self.mix_props, self.coefs_, membership, features, self.demand_shocks)
        for prod in range(num_prods):
            membership_prod = np.copy(membership)
            membership_prod[:, prod + 1] = 0
            probs_prod_removed, _ = self.predict_choice_proba(self.mix_props, self.coefs_, membership_prod, features, self.demand_shocks)
            div_ratios_prod = (probs_prod_removed - curr_probs)/curr_probs[:, [prod + 1]]
            diversion_ratios_by_market[prod] = div_ratios_prod[:, 1:]
            diversion_ratios_by_market[prod, :, prod] = div_ratios_prod[:, 0]

        return np.mean(diversion_ratios_by_market, axis=1)

    ## Generic BFGS approach for solving the subproblem ###
    def FW_MNL_subprob_objective_fixed_coef(self, x0, current_fixed_coef, membership, grad_sales, prod_feats_offered):
        chosen_probs = self._predict_MNL_proba_blp(membership, prod_feats_offered, x0, current_fixed_coef)
        if np.any(chosen_probs[membership > 0] <= 0):
            return 1e10, -np.ones_like(x0)
        weighted_sales = grad_sales * chosen_probs
        obj = np.sum(weighted_sales)
        grad = np.zeros_like(x0)
        for feat_id in range(len(prod_feats_offered)):
            feat_grad_vec = weighted_sales * (prod_feats_offered[feat_id] - np.sum(chosen_probs * prod_feats_offered[feat_id], axis=1, keepdims=True))
            grad[feat_id] = np.sum(feat_grad_vec)
        if np.any(np.isnan(grad)):
            embed()
        return obj, grad

    def pgd_step_size_obj(self, x0, curr_demand_shocks, demand_shocks_grad, membership, prod_feats_offered, observed_ms, offered_prods):
        next_xi_unconstr = curr_demand_shocks - x0*demand_shocks_grad
        delta_1 = np.dot(self.non_zero_eigenvectors, next_xi_unconstr)
        delta_2 = np.dot(self.zero_eigenvectors, next_xi_unconstr)
        zero_coefs = delta_2
        mult = np.sqrt(self.gmm_obj_ub)/(self.non_zero_eigenvalues[0]*np.linalg.norm(delta_1))
        non_zero_coefs = min(1, mult)*delta_1
        zetas = np.dot(self.non_zero_eigenvectors.T, non_zero_coefs) + np.dot(self.zero_eigenvectors.T, zero_coefs)
        demand_shocks = np.zeros_like(observed_ms)
        demand_shocks[:, 1:][offered_prods] = zetas
        # demand_shocks = np.hstack((self.nopurch_shocks, zetas.reshape((membership.shape[0], -1))))
        return self.compute_optimization_objective(self.predict_choice_proba(self.mix_props, self.coefs_, membership, prod_feats_offered, demand_shocks)[0], observed_ms)

    def _demand_shocks_subprob(self, curr_demand_shocks, gmm_matrix, demand_shocks_grad, membership, prod_feats_offered, observed_ms, exp_iter, offered_prods):
        # It is a linear program with convex constraint set; we can solve in closed form
        # min_{\zetas} np.dot(c, \zetas) s.t. \zetas^T W \zetas \leq \delta
        # optimal solution is \zetas = -sqrt(\delta)*np.dot(W^-1, c)/sqrt(c^T W^-1 c)
        # Ref: https://math.stackexchange.com/questions/3495898/constrained-convex-optimization/3496110
        # UPDATE: this doesn't work since W in our case is not positive definite.
        c = demand_shocks_grad[offered_prods]
        delta = self.gmm_obj_ub

        def obj(x0):
            return np.dot(x0, c)

        def obj_grad(x0):
            return c

        def obj_hess(x0):
            num_shocks = x0.shape[0]
            return np.zeros((num_shocks, num_shocks))

        def constr(x0):
            return np.linalg.multi_dot([x0, gmm_matrix, x0])

        def constr_grad(x0):
            return 2*np.dot(gmm_matrix, x0)[np.newaxis]

        def constr_hess(x0, v):
            return 2*gmm_matrix

        def lagrangian(x0):
            if x0 > 0:
                result = lsq_linear(gmm_matrix, -c/(2*x0), bounds=(-DEMAND_SHOCKS_MAX, DEMAND_SHOCKS_MAX))
                lg_obj = np.dot(c, result.x) + x0*(np.linalg.multi_dot([result.x, gmm_matrix, result.x]) - delta)
                return -lg_obj  # Take negative since we are minimizing
            else:
                return np.abs(c).sum()*DEMAND_SHOCKS_MAX

        def lbfgs_b_obj(x0):
            zetas = x0[1:]
            lmult = x0[0]
            grad_zetas = np.dot(gmm_matrix, zetas)
            grad_lmult = np.dot(zetas, grad_zetas) - delta
            obj = np.dot(c, zetas) + lmult*grad_lmult
            return obj, np.append(grad_lmult, c + 2*lmult*grad_zetas)

        if self.subprob_type == 'exact':
            grad_non_zero = np.dot(self.non_zero_eigenvectors, c)/self.non_zero_eigenvalues
            grad_zero = np.dot(self.zero_eigenvectors, c)
            # compute the two solutions
            grad_non_zero_norm = np.linalg.norm(grad_non_zero)
            grad_zero_norm = np.linalg.norm(grad_zero)
            if self.norm_ub_type == 'linf':
                # norm_ub = DEMAND_SHOCKS_MAX
                zero_coefs = -np.sign(grad_zero) * self.fixed_coef_ub
            else:
                # norm_ub = np.sqrt(c.shape[0])*DEMAND_SHOCKS_MAX
                zero_coefs = -(grad_zero / grad_zero_norm) * self.fixed_coef_ub * np.sqrt(c.shape[0])
            non_zero_coefs = -(grad_non_zero / grad_non_zero_norm) * np.sqrt(self.gmm_obj_ub)
            non_zero_coefs /= self.non_zero_eigenvalues
            zetas = np.dot(self.non_zero_eigenvectors.T, non_zero_coefs) + np.dot(self.zero_eigenvectors.T, zero_coefs)
            '''
            bounds = Bounds(-DEMAND_SHOCKS_MAX, DEMAND_SHOCKS_MAX)
            nonlinear_constraint = NonlinearConstraint(constr, -np.inf, self.fixed_coef_ub, jac=constr_grad, hess=constr_hess)
            result = minimize(obj, curr_demand_shocks, method='trust-constr', jac=obj_grad, hess=obj_hess, constraints=[nonlinear_constraint], bounds=bounds, options={'maxiter': 100})
            '''
        elif self.subprob_type == 'lagrangian_lsq':
            res1 = minimize_scalar(lagrangian, method='bounded', bounds=(0, 1000), options={'disp': 0})
            # res1 = minimize(lagrangian, prng.rand()*np.ones(1), method='L-BFGS-B', options={'disp': False, 'maxiter': 1000}, bounds=[(0, None)])
            minimizer_kwargs = {"method": "L-BFGS-B", "bounds": [(0, None)]}
            # res1 = basinhopping(lagrangian, prng.rand()*np.ones(1), minimizer_kwargs=minimizer_kwargs)
            # res1 = brute(lagrangian, ranges=((0, np.inf)), Ns=100)
            res2 = lsq_linear(gmm_matrix, -c / (2 * res1.x), bounds=(-DEMAND_SHOCKS_MAX, DEMAND_SHOCKS_MAX))
            zetas = res2.x

        elif self.subprob_type == 'lagrangian_bfgs':
            bounds = [(0, None)] + [(-DEMAND_SHOCKS_MAX, DEMAND_SHOCKS_MAX)]*len(c)
            res = minimize(lbfgs_b_obj, np.append(0, curr_demand_shocks), method='L-BFGS-B', jac=True, bounds=bounds, options={'disp': False, 'maxiter': 1000})
            zetas = res.x[1:]

        elif 'pgd' in self.subprob_type:
            max_step_size = int(self.subprob_type.split('_')[1])
            shocks_1d = curr_demand_shocks[offered_prods]
            if self.subprob_type.endswith('same'):
                opt_step_size = max_step_size
            else:
                opt_step_size = fminbound(self.pgd_step_size_obj, 0, max_step_size, args=(shocks_1d, c, membership, prod_feats_offered, observed_ms, offered_prods))
                if exp_iter == 0:
                    logger.warning('Optimal step size is {0}'.format(opt_step_size))

            zetas = shocks_1d - opt_step_size*c
            delta_1 = np.dot(self.non_zero_eigenvectors, zetas)
            delta_2 = np.dot(self.zero_eigenvectors, zetas)
            # zero_coefs = delta_2
            mult_1 = np.sqrt(self.gmm_obj_ub) / (self.non_zero_eigenvalues[0] * np.linalg.norm(delta_1))
            mult_2 = self.fixed_coef_ub*np.sqrt(c.shape[0])/np.linalg.norm(delta_2)
            non_zero_coefs = min(1, mult_1) * delta_1
            zero_coefs = min(1, mult_2) * delta_2
            zetas = np.dot(self.non_zero_eigenvectors.T, non_zero_coefs) + np.dot(self.zero_eigenvectors.T, zero_coefs)
        '''
        else:
            numerator = np.dot(inv_gmm_matrix, c)
            denom = np.dot(c, numerator)/self.gmm_obj_ub
            zetas = np.clip(-numerator / np.sqrt(denom), -DEMAND_SHOCKS_MAX, DEMAND_SHOCKS_MAX)  # assume true zetas are N(0,1) so w.h.p will be between -5 and 5
            if denom > 0:
                logger.info('Updated zetas')
            else:
                zetas = np.zeros_like(c)
            '''
        demand_shocks = np.zeros_like(observed_ms)
        demand_shocks[:, 1:][offered_prods] = zetas
        # demand_shocks = np.hstack((self.nopurch_shocks, zetas.reshape((demand_shocks_grad.shape[0], -1))))
        return demand_shocks

    def _base_subproblems(self, point_info, X_obs, gradk, F_obs, current_fixed_coef):
        start_point = point_info[1]
        return minimize(self.FW_MNL_subprob_objective_fixed_coef, start_point, args=(current_fixed_coef, X_obs, gradk, F_obs), method='BFGS', jac=True, options={'disp': False}), point_info[0]
        # return minimize(self.FW_MNL_subprob_objective_fixed_coef, start_point, args=(current_fixed_coef, X_obs, gradk, F_obs), method='L-BFGS-B', jac=True, bounds=[(-BETA_UB, BETA_UB)]*len(F_obs), options={'disp': False}), point_info[0]

    # ==============================================================================
    def _loss_gradient_wrt_demand_shocks(self, alphas, grad_sales, market_shares_by_k):
        grad_shocks = np.zeros_like(grad_sales)
        # derivative of log-ll w.r.t to \zeta_{jt} is grad_sales_{jt}*(\sum_{k} \alpha_k*(skjt*grad_sales_{jt} -  skjt*(\sum_{l}(sklt*grad_sales_{lt}))
        '''
        for k in range(len(alphas)):
            # skjt = self._predict_MNL_proba_blp(membership, prod_feats_offered, self.coefs_[k], curr_demand_shocks)
            skjt = self.component_market_shares[k]
            # this computes
            skjt_weighted = grad_sales*skjt
            sum_grads_kt = np.sum(skjt_weighted, axis=1)
            grad_shocks_k = skjt_weighted - sum_grads_kt[:, np.newaxis]*skjt
            grad_shocks += alphas[k]*grad_shocks_k
        '''
        skjt_weighted = grad_sales*market_shares_by_k
        sum_grads_kt = np.sum(skjt_weighted, 2)
        grad_shocks_k = skjt_weighted - sum_grads_kt[:, :, np.newaxis]*market_shares_by_k
        grad_shocks = np.sum(alphas[:, np.newaxis, np.newaxis]*grad_shocks_k, 0)
        return grad_shocks[:, 1:]

    # top-level function for solving the subproblem in each iteration of the Frank Wolfe algorithm
    def _FW_iteration(self, X_obs, xk, gradk, F_obs, gmm_matrix, n_iter, observed_ms, exp_iter):
        num_params = len(F_obs)
        num_tries = 1
        current_demand_shocks = self.demand_shocks
        prng = RandomState(n_iter)
        offered_prods = X_obs[:, 1:] > 0
        while num_tries < 5:
            # determine starting points for BFGS-based solution to each iteration
            cand_start_points = prng.randn(self.num_subprob_init, num_params)
            pool = mp.Pool(processes=5)
            results = pool.map(partial(self._base_subproblems, X_obs=X_obs, gradk=gradk, F_obs=F_obs, current_fixed_coef=current_demand_shocks), enumerate(cand_start_points))
            pool.close()
            pool.join()
            best_result = min(results, key=lambda w: w[0].fun)
            param_vector = best_result[0].x
            # if prng.rand() > 0.5 and self.subprob_type != 'ignore':
            if self.subprob_type != 'ignore' and '_' not in self.fwVariant:
                fixed_grad = self._loss_gradient_wrt_demand_shocks(self.mix_props, gradk, self.component_market_shares)
                next_demand_shocks = self._demand_shocks_subprob(self.demand_shocks[:, 1:], gmm_matrix, fixed_grad, X_obs, F_obs, observed_ms, exp_iter, offered_prods)
            else:
                fixed_grad = np.zeros_like(xk[:, 1:])
                next_demand_shocks = current_demand_shocks
            # compute choice probs under best param vector
            next_best_probs = self._predict_MNL_proba_blp(X_obs, F_obs, param_vector, current_demand_shocks)
            # compute next frank-wolfe direction
            next_dir = np.hstack((next_best_probs, next_demand_shocks[:, 1:])) - np.hstack((xk, current_demand_shocks[:, 1:]))
            curr_grad = np.hstack((gradk, fixed_grad))
            # check if descent direction
            if np.sum(curr_grad*next_dir) < 0:
                logger.debug('Found descent direction after %d trials', num_tries)
                return param_vector, next_demand_shocks
            else:
                next_dir = np.hstack((next_best_probs, current_demand_shocks[:, 1:])) - np.hstack((xk, current_demand_shocks[:, 1:]))
                # see if okay to move just in Q direction
                if np.sum(curr_grad * next_dir) < 0:
                    logger.debug('Found descent direction after %d trials by ignoring shocks', num_tries)
                    return param_vector, current_demand_shocks
                # else try again
            num_tries += 1

        logger.warning('Unable to find descent direction at iteration {0}'.format(n_iter))
        return None, None

    def wolfe_search_obj(self, zetas, observed_ms, membership, prod_feats_offered):
        demand_shocks = np.hstack((self.nopurch_shocks, zetas.reshape((observed_ms.shape[0], -1))))
        return self.compute_optimization_objective(self.predict_choice_proba(self.mix_props, self.coefs_, membership, prod_feats_offered, demand_shocks)[0], observed_ms)

    def wolfe_search_obj_grad(self, zetas, observed_ms, membership, prod_feats_offered):
        demand_shocks = np.hstack((self.nopurch_shocks, zetas.reshape((observed_ms.shape[0], -1))))
        curr_probs, market_shares_by_k = self.predict_choice_proba(self.mix_props, self.coefs_, membership, prod_feats_offered, demand_shocks)
        curr_grad = self.compute_objective_gradient(curr_probs, observed_ms)
        return np.ravel(self._loss_gradient_wrt_demand_shocks(self.mix_props, curr_grad, market_shares_by_k))

    def _perform_fully_corrective_xi_updates(self, x_init, alpha_init, observed_ms, X_obs, F_obs):
        curr_zetas = np.copy(x_init)
        alpha_coord = np.copy(alpha_init)
        curr_probs, market_shares_by_k = self.predict_choice_proba(self.mix_props, self.coefs_, X_obs, F_obs,
                                                                   curr_zetas)
        for iter in range(self.num_xi_updates):
            # compute current gradient
            curr_grad = self.compute_objective_gradient(curr_probs, observed_ms)
            fixed_grad = self._loss_gradient_wrt_demand_shocks(self.mix_props, curr_grad, market_shares_by_k)
            # compute FW vertex and direction
            fw_weights = [np.sum(fixed_grad*prev_zetas[:, 1:]) for prev_zetas in self.all_demand_shocks]
            fw_vertex = np.argmin(fw_weights)
            fw_direction = self.all_demand_shocks[fw_vertex] - curr_zetas

            # compute away vertex among vertices with non-zero weight
            away_weights = np.where(alpha_coord > 0, fw_weights, -np.inf)
            away_vertex = np.argmax(away_weights)
            away_direction = curr_zetas - self.all_demand_shocks[away_vertex]

            # do an away step
            if -1 * np.sum(fixed_grad * fw_direction[:, 1:]) < -1 * np.sum(fixed_grad * away_direction[:, 1:]) and (
                    away_vertex != fw_vertex) and alpha_coord[away_vertex] < 1:
                away_step = True
                dirk = away_direction
                gamma_max = alpha_coord[away_vertex] / (1 - alpha_coord[away_vertex])
            else:
                # do a frank-wolfe step
                away_step = False
                dirk = fw_direction
                gamma_max = 1.

            # do line search to compute step size
            # opt_step_size = self._perform_line_search_step(soln, dirk, n_counts, gamma_max)
            opt_step_size = fminbound(self._brent_line_search, 0, gamma_max, args=(self.mix_props, curr_zetas, 0, dirk, X_obs, F_obs, observed_ms), xtol=1e-12)

            # update barycentric coordinates alpha
            if not away_step:
                alpha_coord *= (1 - opt_step_size)
                alpha_coord[fw_vertex] += opt_step_size
            else:
                alpha_coord *= (1 + opt_step_size)
                alpha_coord[away_vertex] -= opt_step_size

            # update current solution
            curr_zetas += opt_step_size * dirk
            # clip alphas below precision
            alpha_coord[alpha_coord < 1e-15] = 0.
            curr_probs, market_shares_by_k = self.predict_choice_proba(self.mix_props, self.coefs_, X_obs, F_obs, curr_zetas)

        logger.debug('Performed %d corrective steps', iter)
        # update weights of mixture components
        non_zero_components = alpha_coord > 0
        self.all_demand_shocks = [self.all_demand_shocks[keep_idx] for keep_idx in np.where(non_zero_components)[0]]
        self.demand_shocks = curr_zetas
        self.component_market_shares = market_shares_by_k
        self.demand_shocks_weights = self.demand_shocks_weights[non_zero_components]
        return curr_probs

    def perform_xi_updates(self, xk, observed_ms, X_obs, F_obs, num_update_steps, exp_iter):
        curr_probs = np.copy(xk)
        prev_obj = self.compute_optimization_objective(curr_probs, observed_ms)
        offered_prods = X_obs[:, 1:] > 0
        for nupd in range(num_update_steps):
            curr_grad = self.compute_objective_gradient(curr_probs, observed_ms)
            fixed_grad = self._loss_gradient_wrt_demand_shocks(self.mix_props, curr_grad, self.component_market_shares)
            next_demand_shocks = self._demand_shocks_subprob(self.demand_shocks[:, 1:], None, fixed_grad, X_obs, F_obs, observed_ms, exp_iter, offered_prods)
            if self.subprob_type == 'exact':
                fw_direction = next_demand_shocks - self.demand_shocks
                # compute away vertex among vertices with non-zero weight
                if self.do_away_steps:
                    fw_weights = [np.sum(fixed_grad*prev_zetas[:, 1:]) for prev_zetas in self.all_demand_shocks]
                    away_weights = np.where(self.demand_shocks_weights > 0, fw_weights, -np.inf)
                    away_vertex = np.argmax(away_weights)
                    away_direction = self.demand_shocks - self.all_demand_shocks[away_vertex]
                else:
                    away_vertex = 0
                    away_direction = np.zeros_like(observed_ms)

                if self.do_away_steps and -1 * np.sum(fixed_grad * fw_direction[:, 1:]) < -1 * np.sum(fixed_grad * away_direction[:, 1:]) and self.demand_shocks_weights[away_vertex] < 1:
                    away_step = True
                    dirk = away_direction
                    gamma_max = self.demand_shocks_weights[away_vertex] / (1 - self.demand_shocks_weights[away_vertex])
                else:
                    # do a frank-wolfe step
                    away_step = False
                    dirk = fw_direction
                    gamma_max = 1.
                step_size = fminbound(self._brent_line_search, 0, gamma_max, args=(self.mix_props, self.demand_shocks, 0, dirk, X_obs, F_obs, observed_ms), xtol=1e-15)
                if not away_step:
                    self.demand_shocks_weights = np.append((1 - step_size)*self.demand_shocks_weights, step_size)
                    self.all_demand_shocks.append(next_demand_shocks)
                else:
                    self.demand_shocks_weights *= (1 + step_size)
                    self.demand_shocks_weights[away_vertex] -= step_size

                self.demand_shocks += step_size*dirk
            else:
                # no need to update xi when doing pgd
                self.demand_shocks = next_demand_shocks

            curr_probs = self.set_component_market_shares(X_obs, F_obs)
            curr_obj = self.compute_optimization_objective(curr_probs, observed_ms)
            #if curr_obj > prev_obj:
            #    logger.warning('..Breaking xi update loop...')
            #    break
            prev_obj = curr_obj
            # if n_iter > 0:
            # print(step_size, next_demand_shocks[0])
            # print('Working on iter {0}'.format((nupd)))

        return curr_probs

    # helper method to check if current solution lies within convex hull of vertices
    def _check_convex_combination(self, xk, alphas, vertices):
        result = np.abs(np.sum(alphas[:, np.newaxis, np.newaxis] * vertices, 0) - xk).sum()
        assert result < 1e-8, 'Not a convex combination:' + str(result)
        assert np.abs(1 - np.sum(alphas)) < 1e-10, 'Sum to 1 violated'

    # outer wrapper for performing FCFW
    def _perform_fully_corrective_step(self, x_init, n_counts, alpha_init, max_iter):
        soln = np.copy(x_init)
        alpha_coord = np.copy(alpha_init)
        # compute current objective value
        prev_obj = self.compute_optimization_objective(soln, n_counts)
        for iter in range(max_iter):
            # check if soln is in the polytope
            self._check_convex_combination(soln, alpha_coord, self.component_market_shares)
            # compute current gradient
            curr_grad = self.compute_objective_gradient(soln, n_counts)
            # compute FW vertex and direction
            fw_weights = np.sum(curr_grad * self.component_market_shares, axis=(1, 2))
            fw_vertex = np.argmin(fw_weights)
            fw_direction = self.component_market_shares[fw_vertex] - soln

            # compute away vertex among vertices with non-zero weight
            away_weights = np.where(alpha_coord > 0, fw_weights, -np.inf)
            away_vertex = np.argmax(away_weights)
            away_direction = soln - self.component_market_shares[away_vertex]

            # check duality gap
            gap = -1 * np.sum(curr_grad * (fw_direction + away_direction))
            if gap < CORRECTIVE_GAP:
                break

            # do an away step
            if -1 * np.sum(curr_grad * fw_direction) < -1 * np.sum(curr_grad * away_direction) and (
                        away_vertex != fw_vertex) and alpha_coord[away_vertex] < 1:
                away_step = True
                dirk = away_direction
                gamma_max = alpha_coord[away_vertex] / (1 - alpha_coord[away_vertex])
            else:
                # do a frank-wolfe step
                away_step = False
                dirk = fw_direction
                gamma_max = 1.

            # do line search to compute step size
            # opt_step_size = self._perform_line_search_step(soln, dirk, n_counts, gamma_max)
            opt_step_size = fminbound(self._brent_line_search_fixed_coef, 0, gamma_max, args=(soln, dirk, n_counts), xtol=1e-15)

            # update barycentric coordinates alpha
            if not away_step:
                alpha_coord *= (1 - opt_step_size)
                alpha_coord[fw_vertex] += opt_step_size
            else:
                alpha_coord *= (1 + opt_step_size)
                alpha_coord[away_vertex] -= opt_step_size

            # update current solution
            soln += opt_step_size * dirk
            # clip alphas below precision
            alpha_coord[alpha_coord < 1e-15] = 0.
            # update objective value
            curr_obj = self.compute_optimization_objective(soln, n_counts)
            if curr_obj > prev_obj:
                logger.debug('..Breaking Q update loop...')
                break
            prev_obj = curr_obj

        logger.debug('Performed %d corrective steps', iter)
        # update weights of mixture components
        non_zero_components = alpha_coord > 0
        self.mix_props = alpha_coord[non_zero_components]
        self.coefs_ = self.coefs_[non_zero_components]
        self.component_market_shares = self.component_market_shares[non_zero_components]
        # return current estimate of soln
        return soln

    # =================================================================
    # LINESEARCH ROUTINES
    # =================================================================
    def _brent_line_search_fixed_coef(self, alpha, curr_probs, next_dir, n_counts):
        return self.compute_optimization_objective(curr_probs + alpha * next_dir, n_counts)

    def _brent_line_search(self, gamma, curr_alphas, curr_fixed_coef, next_alphas_dir, next_fixed_coef_dir, membership, prod_feats_offered, n_counts):
        fixed_coef = curr_fixed_coef + gamma*next_fixed_coef_dir
        alphas = curr_alphas + gamma*next_alphas_dir
        return self.compute_optimization_objective(self.predict_choice_proba(alphas, self.coefs_, membership, prod_feats_offered, fixed_coef)[0], n_counts)

    # outer wrapper for fitting the model to aggregate sales data
    def fit_to_choice_data(self, X_obs, F_obs, Sales_obs, gmm_matrix, instruments, W, num_iter, exp_iter, expt_id, init_betas=None, init_shocks=None, init_mix_props=None):
        '''
        X_obs: T x (J+1) membership array where T is number of markets/offer-sets and J is num prods
        Sales_obs: T x (J+1) sales array where T is number of markets/offer-sets and J is num prods
        F_obs: list of T x (J+1) arrays representing product features, length of list == number of features
        instruments: JT x K matrix where K is number of instruments (Z matrix in blp notation)
        W: K x K GMM weighting matrix
        gmm_matrix: product of instruments, W, instruments.T (ZWZ^T in blp notation)
        num_iter : number of iterations of FW algo
        '''

        # can improve initialization: single-class MNL or 2-class LC-MNL
        self.coefs_ = np.zeros(len(F_obs))[np.newaxis]
        self.mix_props = np.array([1.])
        self.demand_shocks = np.zeros_like(Sales_obs)
        if init_betas is not None:
            self.coefs_ = np.copy(init_betas)
        if init_mix_props is not None:
            self.mix_props = np.copy(init_mix_props)
        if init_shocks is not None:
            self.demand_shocks = np.copy(init_shocks)

        self.all_demand_shocks = [self.demand_shocks]
        self.demand_shocks_weights = np.array([1.])
        curr_probs = self.set_component_market_shares(X_obs, F_obs)
        # compute the starting objective value
        prev_obj = self.compute_optimization_objective(curr_probs, Sales_obs)
        logger.debug('At iteration 0, current obj is %.10f for variant %s', prev_obj, self.fwVariant)
        change_norm = 1
        self.nopurch_shocks = np.zeros(len(X_obs))[:, np.newaxis]
        # curr_probs = self.perform_xi_updates(curr_probs, observed_ms, X_obs, F_obs, 100, False)
        # prev_obj = self.compute_optimization_objective(curr_probs, observed_ms)
        num_components_added = 0
        for iter in range(num_iter):
            if self.fwVariant != 'only_xi':
                est_info = self.fwVariant.split('_')
                num_Q_updates = int(est_info[-1]) if len(est_info) > 1 else 1
                for q_upd_iter in range(num_Q_updates):
                    (next_param_vector, next_fixed_coef) = self._FW_iteration(X_obs, curr_probs, self.compute_objective_gradient(curr_probs, Sales_obs), F_obs, gmm_matrix, iter, Sales_obs, exp_iter)
                    if next_param_vector is None:
                        break
                    num_components_added += 1
                    # add the new component
                    self.coefs_ = np.append(self.coefs_, next_param_vector[np.newaxis], 0)
                    # self.all_demand_shocks.append(next_fixed_coef)
                    curr_alphas = np.append(self.mix_props, 0.)
                    next_alphas = np.zeros_like(curr_alphas)
                    next_alphas[-1] = 1
                    # find the optimal step size
                    if 'fixed-step-size' in self.fwVariant:
                        step_size = 2 * self.learning_rate / (iter + 3)
                    else:
                        step_size = fminbound(self._brent_line_search, 0, 1, args=(curr_alphas, self.demand_shocks, next_alphas - curr_alphas, next_fixed_coef - self.demand_shocks, X_obs, F_obs, Sales_obs), xtol=1e-15)

                    self.mix_props = np.append((1-step_size)*self.mix_props, step_size)
                    # self.demand_shocks = (1 - step_size)*self.demand_shocks + step_size*next_fixed_coef
                    # self.demand_shocks_weights = np.append((1-step_size)*self.demand_shocks_weights, step_size)
                    curr_probs = self.set_component_market_shares(X_obs, F_obs)

                if 'swap' not in expt_id and 'corrective' in self.fwVariant:
                    curr_probs = self._perform_fully_corrective_step(curr_probs, Sales_obs, self.mix_props, MAX_CORRECTIVE_STEPS)

            if self.subprob_type != 'ignore':
                curr_probs = self.perform_xi_updates(curr_probs, Sales_obs, X_obs, F_obs, self.num_xi_updates, exp_iter)
                # curr_probs = self._perform_fully_corrective_xi_updates(self.demand_shocks, self.demand_shocks_weights,observed_ms, X_obs, F_obs)

            if 'swap' in expt_id and 'corrective' in self.fwVariant:
                 curr_probs = self._perform_fully_corrective_step(curr_probs, Sales_obs, self.mix_props, MAX_CORRECTIVE_STEPS)
            curr_obj = self.compute_optimization_objective(curr_probs, Sales_obs)
            if 'fixed-step-size' not in self.fwVariant and 'drop' not in self.fwVariant:
                assert (curr_obj <= prev_obj or curr_obj - prev_obj < 1e-4), embed()
                # 'Objective not increasing for variant %s:%s' % (self.fwVariant, self.subprob_type)
            change_norm = np.abs(curr_obj - prev_obj) / np.abs(prev_obj)
            if iter%50 == 0:
                # print('At iteration %d, current obj is %.10f', iter, curr_obj)
                logger.info('At iteration %d, current obj is %.10f', iter, curr_obj)
            prev_obj = curr_obj

        data_choice_probs = Sales_obs / np.sum(Sales_obs, 1, keepdims=True)
        data_entropy = -np.mean(Sales_obs * np.where(data_choice_probs > 0, np.log(data_choice_probs), 0))
        final_kldiv = curr_obj - data_entropy
        logger.info('Final KL divergence for run {1} is {0} after adding {2} types'.format(final_kldiv, exp_iter, num_components_added))
        zetas_Z = np.dot(np.ravel(self.demand_shocks[:, 1:]), instruments)
        final_gmm_obj = np.linalg.multi_dot([zetas_Z, W, zetas_Z])
        pickle.dump((self.mix_props, self.coefs_, final_gmm_obj, final_kldiv, self.demand_shocks), open(
            file_path_results+'{0}_BLPestimator_variant={3}_subprob_type={4}_away_xi_steps={5}_iter={1}_R={2}.stats'.format(expt_id, exp_iter, num_iter,
                                                                                                 self.fwVariant, self.subprob_type, self.do_away_steps), 'wb'))
        assert np.around(np.sum(self.mix_props) - 1, 7) == 0, embed()
        assert self.coefs_.shape[0] == self.mix_props.shape[0], embed()


    # ==============================================================================

