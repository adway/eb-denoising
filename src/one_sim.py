import importlib
import numpy as np
import scipy.stats as stats
from scipy.linalg import sqrtm, inv
import npeb
from npeb.GLMixture import GLMixture
import itertools


def generate_discrete_theta(support, probs, size):
    Theta = np.random.choice(support, size=size, p=probs)
    return Theta.reshape(-1, 1)

def generate_normal_mixture_theta(means, variances, weights, size):
    components = np.random.choice(len(weights), size=size, p=weights)
    Theta = np.array([np.random.normal(loc=means[c], scale=np.sqrt(variances[c])) for c in components])
    return Theta.reshape(-1, 1)

def generate_t_mixture_theta(dfs, locs, scales, weights, size):
    components = np.random.choice(len(weights), size=size, p=weights)
    Theta = np.array([np.random.standard_t(df=dfs[c]) * scales[c] + locs[c] for c in components])
    return Theta.reshape(-1, 1)

def generate_Z(theta, sigma):
    Z = theta + np.random.normal(loc=0, scale=sigma, size=theta.shape)
    return Z.reshape(-1, 1)

def mse(y_hat, y_true):
    y_hat = np.asarray(y_hat)
    y_true = np.asarray(y_true)
    return np.mean((y_hat - y_true)**2)

def mse_regret(y_emp, y_true, y_oracle):
    return mse(y_emp, y_true) - mse(y_oracle, y_true)
   

class OneSimulation:
  def __init__(self, prior_type, prior_params, n, sigma2, n_supp=150):
    self.prior_type = prior_type
    self.prior_params = prior_params
    self.n = n
    self.sigma2 = sigma2
    self.n_supp = n_supp # this defines the number of support points for which we should estimate/use for our prior. should be generated from true distribution for oracle; across a grid for empirical.
    self.Theta = self.generate_theta()
    self.Z = generate_Z(self.Theta, np.sqrt(sigma2))

  def generate_theta(self):
    if self.prior_type == "discrete":
        support = self.prior_params['support']
        probs = self.prior_params['probs']
        Theta = generate_discrete_theta(support, probs, self.n)
    elif self.prior_type == "normal_mixture":
        means = self.prior_params['support']
        variances = self.prior_params['variances']
        weights = self.prior_params['probs']
        Theta = generate_normal_mixture_theta(means, variances, weights, self.n)
    elif self.prior_type == "t_mixture":
        dfs = self.prior_params['dfs']
        means = self.prior_params['support']
        variances = self.prior_params['variances']
        weights = self.prior_params['probs']
        Theta = generate_t_mixture_theta(dfs, means, variances, weights, self.n)
    else:
        raise ValueError("Unsupported prior type")
    return Theta
  
  def get_observations(self):
    return self.Z
  
  def get_oracle_estimates(self):
    if self.prior_type == "discrete":
        support = self.prior_params['support']
        probs = self.prior_params['probs']
        prior = generate_discrete_theta(support, probs, self.n_supp)
    elif self.prior_type == "normal_mixture":
        means = self.prior_params['support']
        variances = self.prior_params['variances']
        weights = self.prior_params['probs']
        prior = generate_normal_mixture_theta(means, variances, weights, self.n_supp)
    elif self.prior_type == "t_mixture":
        dfs = self.prior_params['dfs']
        means = self.prior_params['support']
        variances = self.prior_params['variances']
        weights = self.prior_params['probs']
        prior = generate_t_mixture_theta(dfs, means, variances, weights, self.n_supp)
    else:
        raise ValueError("Unsupported prior type")
    prec = np.ones_like(self.Theta) / self.sigma2
    ob_model = GLMixture(prec_type="diagonal")
    ob_model.set_params(atoms=prior, weights=np.ones(self.n_supp)/self.n_supp)
    ob_means = ob_model.posterior_mean(self.Z, prec) # sample posterior means
    ob_indices, ob_samples = ob_model.posterior_sample(self.Z, prec, n_samples=self.n) # get samples from the mixture posterior
    # ob_each_samples = ob_model.each_posterior_sample(self.Z, prec, n_samples=self.n) # get samples from the mixture posterior for each observation
    return prior, ob_means, ob_indices, ob_samples, None
  
  def get_eb_estimates(self):
    z_min, z_max = np.min(self.Z), np.max(self.Z)
    grid = np.linspace(z_min, z_max, self.n_supp).reshape(-1, 1)
    prec = np.ones_like(self.Theta) / self.sigma2
    eb_model = GLMixture(prec_type="diagonal", atoms_init=grid)
    eb_model.fit(self.Z, prec, max_iter_em=50)
    prior, weights = eb_model.get_params()
    eb_means = eb_model.posterior_mean(self.Z, prec)
    eb_indices, eb_samples = eb_model.posterior_sample(self.Z, prec, n_samples=self.n)
    # eb_each_samples = eb_model.each_posterior_sample(self.Z, prec, n_samples=self.n)
    # get variance constrained posterior mean estimates
    c_means = np.mean(eb_means,axis=0)
    M_hat = (eb_means-c_means).T@(eb_means-c_means)/self.n
    mu_hat = np.mean(self.Z, axis = 0)
    A_hat = (self.Z - mu_hat).T@(self.Z - mu_hat)/self.n - self.sigma2*np.eye(self.Z.shape[1])
    transport_hat = inv(sqrtm(M_hat))@sqrtm(sqrtm(M_hat)@A_hat@sqrtm(M_hat))@inv(sqrtm(M_hat))
    evcb_means = (eb_means - c_means)@transport_hat + c_means

    return weights, prior, eb_means, eb_indices, eb_samples, evcb_means
  
  def get_similarity_metrics(self, estimates):
    o_prior, ob_means, ob_indices, ob_samples, _= estimates['oracle']
    eb_weights, eb_prior, eb_means, eb_indices, eb_samples, evcb_means = estimates['eb']
    prior_dist = stats.energy_distance(o_prior.flatten(), eb_prior.flatten(), u_weights=None, v_weights=eb_weights.flatten()) / np.sqrt(2)
    denoise_regret = mse_regret(eb_means, self.Theta, ob_means)
    post_dist = stats.energy_distance(ob_samples.flatten(), eb_samples.flatten()) / np.sqrt(2)
    # each_post_avg_dist = np.mean([stats.energy_distance(ob_each_samples[i].flatten(), eb_each_samples[i].flatten()) / np.sqrt(2) for i in range(self.n)])
    evcb_distance = stats.energy_distance(ob_samples.flatten(), evcb_means.flatten()) / np.sqrt(2)
    
    return {
        'prior_dist': prior_dist,
        'denoise_regret': denoise_regret,
        'post_dist': post_dist,
        'evcb_distance': evcb_distance
        # 'each_post_avg_dist': each_post_avg_dist
    }
  
  def run_simulation(self):
    estimates = {}
    estimates['oracle'] = self.get_oracle_estimates()
    estimates['eb'] = self.get_eb_estimates()
    metrics = self.get_similarity_metrics(estimates)
    return estimates, metrics