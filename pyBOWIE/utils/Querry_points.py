# *******************************************************
# Import libraries
# *******************************************************

import numpy as np
import sympy as sp
from sklearn.mixture import GaussianMixture
from scipy.optimize import minimize
from scipy.stats import t
from ..utils.Aux import Flatten

# *******************************************************
# ****** Find_descriptors ******
# *******************************************************

def Find_descriptors(connected_elements, n_elements, n_jobs, dims, alpha, af_params, constraints_method, model, models_const, engine, Acq_fun):

    # *************************
    # Statistical_descriptors
    # *************************

    def Statistical_descriptors(connected_elements, n_elements, dims, alpha):

        # x_c0_rand
        def Desc_one_dims(matrix, n_elements, alpha):

            mu = [np.mean(matrix[i], axis=0).reshape(-1, 1) for i in range(n_elements)]
            sigma = []
            t_critical = []

            for i in range(n_elements):
                n = len(matrix[i])
                df = n - 1
                t_critical.append(t.ppf((1 + alpha) / 2, df).item())
                var = np.std(matrix[i].T)
                if var == 0 or len(matrix[i]) < 3:
                    sigma.append(1e-6)
                else:
                    sigma.append(var/np.sqrt(n))

            return mu, sigma, t_critical

        # x_c0_rand
        def Desc_high_dims(matrix, n_elements, dims):

            mu = [np.mean(matrix[i], axis=0).reshape(-1, 1) for i in range(n_elements)]
            Sigma = []
            Sigma_inv = []

            for i in range(n_elements):
                if (len(matrix[i]) < 3):
                    cov = np.eye(dims)
                    Sigma.append(cov)
                    Sigma_inv.append(np.linalg.inv(cov))
                else:
                    # Covariance matrix
                    cov = np.cov(matrix[i].T)
                    # Check if the covariance matrix is singular. If so, add a small value to the diagonal
                    if np.linalg.det(cov) == 0:
                        cov = cov + 1e-6*np.eye(dims)
                    Sigma.append(cov)
                    # Inverse of the covariance matrix
                    Sigma_inv.append(np.linalg.inv(cov))

            return mu, Sigma_inv

        # Main program

        if dims == 1:
            mu, Sigma_inv, t_critical = Desc_one_dims(connected_elements, n_elements, alpha)
        else:
            mu, Sigma_inv = Desc_high_dims(connected_elements, n_elements, dims)
            t_critical = None

        return mu, Sigma_inv, t_critical
    
    # *************************
    # Clusters_GMM
    # *************************    

    def Clusters_GMM(connected_elements, n_jobs, dims, af_params, constraints_method, model, models_const, engine, Acq_fun):

        x_filtred = np.concatenate(connected_elements)
        f_filtred = Acq_fun(x_filtred, af_params, constraints_method, model, models_const, engine)
        gmm = GaussianMixture(n_components=n_jobs, random_state=0).fit(np.hstack((x_filtred, f_filtred.reshape(-1,1))))
        mu = gmm.means_[:,0:dims]
        mu = [mu[i].reshape(-1,1) for i in range(n_jobs)]
        Sigma = [gmm.covariances_[i][0:dims,0:dims] for i in range(n_jobs)]
        Sigma = np.array(Sigma)
        Sigma_inv = [np.linalg.inv(Sigma[i]) for i in range(n_jobs)]

        return mu, Sigma_inv
    
    # *************************
    # Main program
    # *************************

    if n_elements < n_jobs:
        mu, Sigma_inv = Clusters_GMM(connected_elements, n_jobs, dims, af_params, constraints_method, model, models_const, engine, Acq_fun)
        if dims == 1:
            t_critical = []
            for i in range(n_elements):
                n = len(connected_elements[i])
                df = n - 1
                t_critical.append(t.ppf((1 + alpha) / 2, df).item())
        else:
            t_critical = None
    else:
        if dims == 1:
            mu, Sigma_inv, t_critical = Statistical_descriptors(connected_elements, n_elements, dims, alpha)
        else:
            mu, Sigma_inv, t_critical = Statistical_descriptors(connected_elements, n_elements, dims, alpha)

    return mu, Sigma_inv, t_critical

# *******************************************************
# ****** Find_constrains ******
# *******************************************************

def Find_constrains(x_symb, mu, Sigma_inv, chi, t_critical, n_jobs, dims):

    # *************************
    # CI_one_dims
    # *************************    

    def CI_one_dims(n_jobs, mu, sigma, t_critical):

        CI = []

        for i in range(n_jobs):
            margin_of_error = t_critical[i] * sigma[i]
            CI.append([mu[i].item() - margin_of_error, mu[i].item() + margin_of_error])
        
        return CI
    
    # *************************
    # CI_high_dims
    # *************************
    
    def CI_high_dims(x_symb, n_jobs, mu, Sigma_inv, chi):

        CI = []

        for i in range(n_jobs):
            y = (x_symb-mu[i])
            dist = y.T*Sigma_inv[i]*y
            CI.append(sp.LessThan(dist[0], chi))
            
        return CI
    
    # *************************
    # Lambify_CI
    # *************************
    
    def Lambify_CI(x_symb, CI, n_jobs, dims):
        
        # Lamba_fun
        def Lamba_fun(fun):

            lam = lambda x: fun(*x)
            
            return lam
        
        if dims == 1:
            CI_lambda = CI
        else:
            const_lamb = [sp.lambdify(x_symb, (CI[i].rhs - CI[i].lhs), 'numpy') for i in range(n_jobs)]
            CI_lambda = [Lamba_fun(const_lamb[i]) for i in range(n_jobs)]
        
        return CI_lambda
    
    # *************************
    # Main program
    # *************************

    if dims == 1:                           
        CI = CI_one_dims(n_jobs, mu, Sigma_inv, t_critical)
    else:
        CI = CI_high_dims(x_symb, n_jobs, mu, Sigma_inv, chi)

    return Lambify_CI(x_symb, CI, n_jobs, dims)

# *******************************************************
# ****** Querry ******
# *******************************************************

def Querry(n_jobs, mu, CI_lambda, bnds, dims, af_params, constraints_method, sense, model, models_const, engine, Acq_fun):

    # Acquisition function for the optimization
    def Acquisition_function_opt(x, af_params, sense, model, models_const, engine):

        if sense == "maximize":
            af_opt = -Acq_fun(x.reshape(1,-1), af_params, constraints_method, model, models_const, engine)
        elif sense == "minimize":
            af_opt = Acq_fun(x.reshape(1,-1), af_params, constraints_method, model, models_const, engine)
        
        return af_opt
    
    # *************************
    # Main program
    # *************************
    
    x_opt = []

    for i in range(n_jobs):
        # Define the optimization problem
        if dims == 1:
            bnds = tuple(map(tuple, CI_lambda))
            res = minimize(Acquisition_function_opt, x0=mu[i].flatten(), args=(af_params, sense, model, models_const, engine), method='BFGS', bounds=(bnds[i],))
        else:
            res = minimize(Acquisition_function_opt, x0=mu[i].flatten(), args=(af_params, sense, model, models_const, engine), method='SLSQP', bounds=bnds, constraints={'type': 'ineq', 'fun': CI_lambda[i]})
        x_opt.append(res.x)

    return x_opt