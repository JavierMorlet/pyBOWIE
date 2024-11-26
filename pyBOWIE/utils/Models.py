# *******************************************************
# ****** Import libraries ******
# *******************************************************

import numpy as np
import pandas as pd
import properscoring as ps
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import kernels as kernels_sklear
from sklearn.gaussian_process import GaussianProcessRegressor as GPR_sklearn
from sklearn.gaussian_process import GaussianProcessClassifier as GPC_sklearn
from gpflow  import likelihoods
from gpflow import kernels as kernels_gpflow
from gpflow.models import GPR as GPR_gpflow
from gpflow.models import SGPR as SGPR_gpflow
from gpflow.models import VGP as GPC_gpflow
from gpflow.optimizers import Scipy as optimizer_gpflow

try: 
    from GPy import kern as kernels_gpy
    from GPy.models import GPRegression as GPR_gpy
    from GPy.models import SparseGPRegression as SGPR_gpy
    from GPy.models import GPClassification as GPC_gpy
except:
    pass

# *******************************************************
# ****** Train_model ******
# *******************************************************

def Train_model(x, f, kernel, surrogate, n_restarts, engine):

    def Select_model(x, f, kernel, surrogate, engine):

        if surrogate == "GP":
            if engine == 'gpflow':
                model = GPR_gpflow((x, f), kernel=kernel,)
            elif engine == 'sklearn':
                model = GPR_sklearn(kernel=kernel, n_restarts_optimizer=n_restarts)
            elif engine == 'GPy':
                model = GPR_gpy(x, f, kernel)
            else:
                pass
        elif surrogate == "SGP":
            if engine == 'gpflow':
                rng = np.random.default_rng(1234)
                n_inducing = 10
                inducing_variable = rng.choice(x, size=n_inducing, replace=False)
                model = SGPR_gpflow((x, f), kernel=kernel, inducing_variable=inducing_variable)
            elif engine == 'sklearn':
                raise ValueError("sklearn does not implement Sparse Gaussian Process")
            elif engine == 'GPy':
                model = SGPR_gpy(x, f, kernel)
            else:
                pass
        else:
            pass
        
        return model
    
    if x.dtype == object:
        x = np.asarray(x).astype(np.float64)
    
    model = Select_model(x, f, kernel, surrogate, engine)

    if engine == 'gpflow':
        opt = optimizer_gpflow()
        opt.minimize(model.training_loss, model.trainable_variables)
    elif engine == 'sklearn':
        model.fit(x, f)
    elif engine == 'GPy':
        model.optimize(optimizer='lbfgsb', max_iters=1000, messages=False)
        model.optimize_restarts(num_restarts=n_restarts, verbose=False)
    else:
        pass

    return model

# *******************************************************
# ****** Train_models_const ******
# *******************************************************

def Train_models_constraints(x, g, constraints, n_constraints, constraints_method, engine):

    
    def Train_PoF(x, g, n_const, engine):
        
        models = []
        for i in range(n_const):
            if engine == 'gpflow':
                model = GPR_gpflow((x, g[:,i].reshape(-1,1)), kernel=kernels_gpflow.SquaredExponential(),)
                opt = optimizer_gpflow()
                opt.minimize(model.training_loss, model.trainable_variables)
            elif engine == 'sklearn':
                model = GPR_sklearn().fit(x, g[:,i].reshape(-1,1))
            elif engine == 'GPy':
                model = GPR_gpy(x, g[:,i].reshape(-1,1))
                model.optimize()
                models.append(model)
            else: 
                pass

        return models
    
    def Train_GPC(x, g, engine):

        if engine == 'gpflow':
            model = GPC_gpflow((x, g),kernel=kernels_gpflow.RBF(),likelihood=likelihoods.Bernoulli(),)
            opt = optimizer_gpflow()            
            opt.minimize(model.training_loss, model.trainable_variables)
        elif engine == 'sklearn':
            model = GPC_sklearn().fit(x, g)
        elif engine == 'GPy':
            model = GPC_gpy(x, g)
            model.optimize()
        else:
            pass

        return model

    if constraints == None: 
        models_const = None
    else:
        if constraints_method == "PoF":
            models_const = Train_PoF(x, g, n_constraints, engine)
        elif constraints_method == "GPC":
            models_const = Train_GPC(x, g, engine)
        else:
            pass

    return models_const

# *******************************************************
# ****** Kernel_discovery ******
# *******************************************************

def Kernel_discovery(x, f, dims, surrogate, beta, evals, engine):

    def Search(x_train, x_test, f_train, f_test, n, kernels, surrogate, engine):

        BIC, CRPS, models = [], [], {}

        if x_train.dtype == object:
            x_train = np.asarray(x_train).astype(np.float64)
            x_test = np.asarray(x_test).astype(np.float64)
        
        for name, kernel in kernels.items():
            if surrogate == "GP":
                if engine == 'gpflow':
                    model = GPR_gpflow((x_train, f_train), kernel=kernel,)
                elif engine == 'sklearn':
                    model = GPR_sklearn(kernel=kernel)
                elif engine == 'GPy':
                    model = GPR_gpy(x_train, f_train.reshape(-1,1), kernel)
                else:
                    pass
            elif surrogate == "SGP":
                if engine == 'gpflow':
                    rng = np.random.default_rng(1234)
                    n_inducing = 10
                    inducing_variable = rng.choice(x_train, size=n_inducing, replace=False)
                    model = SGPR_gpflow((x_train, f_train), kernel=kernel, inducing_variable=inducing_variable)
                elif engine == 'sklearn':
                    raise ValueError("sklearn does not implement Sparse Gaussian Process")
                elif engine == 'GPy':
                    model = SGPR_gpy(x_train, f_train, kernel)
                else:
                    pass
            else:
                pass

            if engine == 'gpflow':
                opt = optimizer_gpflow()
                opt.minimize(model.training_loss, model.trainable_variables)
                ll, k = model.maximum_log_likelihood_objective().numpy(), sum(np.prod(v.shape) for v in model.trainable_variables)
                mean, var = model.predict_f(x_test)
                mean, var = mean.numpy(), var.numpy()
            elif engine == 'sklearn':
                model.fit(x_train, f_train)
                ll, k = model.log_marginal_likelihood_value_, len(model.kernel_.theta)
                mean, var = model.predict(x_test, return_std=True)
            elif engine == 'GPy':
                model.optimize()
                ll, k = model.log_likelihood(), model._size_transformed()
                mean, var = model.predict(x_test)
            else:
                pass
            
            BIC.append((-2*ll + k*np.log(n)).item())
            CRPS.append(ps.crps_gaussian(f_test, mean, var).mean())
            models[name] = model
        
        return BIC, CRPS, models

    def Get_best_model(BIC, CRPS, models, beta, eval):

        normalized_bics = (BIC - np.min(BIC)) / (np.max(BIC) - np.min(BIC))
        normalized_crpss = (CRPS - np.min(CRPS)) / (np.max(CRPS) - np.min(CRPS))

        # Combine scores (example: equal weight)
        
        scores = beta * normalized_bics + (1-beta) * normalized_crpss
        #df_BIC = pd.DataFrame(np.vstack((BIC, normalized_bics)), columns=models.keys())
        #df_CRPS = pd.DataFrame(np.vstack((CRPS, normalized_crpss)), columns=models.keys())
        #df_scores = pd.DataFrame(np.array(scores).reshape(1,-1), columns=models.keys())
        #df_BIC.to_csv('/Users/javiermorlet/Codes/Project_3/plots/Results/Example_1/Fig_1_3/df_BIC' + str(eval) + '.csv')
        #df_CRPS.to_csv('/Users/javiermorlet/Codes/Project_3/plots/Results/Example_1/Fig_1_3/df_CRPS' + str(eval) + '.csv')
        #df_scores.to_csv('/Users/javiermorlet/Codes/Project_3/plots/Results/Example_1/Fig_1_3/df_scores' + str(eval) + '.csv')
        models_ordered = [x for _, x in sorted(zip(scores, models.keys()))]
        
        return {models_ordered[0]: models[models_ordered[0]]}
    
    x_train, x_test, f_train, f_test = train_test_split(x, f, test_size=0.2)
    n = x_train.shape[0]
    # Base kernels: SE, Periodic, Linear, RatQuad
    if engine == 'gpflow':
        kernels_ = {"linear": kernels_gpflow.Linear(),
                "RBF": kernels_gpflow.SquaredExponential(),
                "Mater_52": kernels_gpflow.Matern52(),
                "Periodic": kernels_gpflow.Periodic(kernels_gpflow.SquaredExponential())
                }
    elif engine == 'sklearn':
        kernels_ = {"linear": kernels_sklear.DotProduct(),
                "RBF": kernels_sklear.RBF(),
                "Mater_52": kernels_sklear.Matern(nu=2.5),
                "Rational_Quadratic": kernels_sklear.RationalQuadratic()  
                }
    elif engine == 'GPy':
        kernels_ = {"linear": kernels_gpy.Linear(input_dim=dims),
                "RBF": kernels_gpy.RBF(input_dim=dims, variance=1.0, lengthscale=1.0),
                "Mater_52": kernels_gpy.Matern52(input_dim=dims, variance=1.0, lengthscale=1.0),
                "Periodic": kernels_gpy.StdPeriodic(input_dim=dims, variance=1.0, lengthscale=1.0, period=1.0)
                }
    else:
        pass

    for i in range(evals):
        base_kernels = kernels_.copy()
        BIC, CRPS, models = Search(x_train, x_test, f_train, f_test, n, base_kernels, surrogate, engine)
        best_model = Get_best_model(BIC, CRPS, models, beta, i)
        base_model = list(best_model.values())[0]
        base_model_name = list(best_model.keys())[0]
        if engine == 'gpflow':
            base_model_kern = base_model.kernel
        elif engine == 'sklearn':
            base_model_kern = base_model.kernel_
        elif engine == 'GPy':
            base_model_kern = base_model.kern
        else:
            pass
        if i == evals: break
        kernels_ = {}
        for name, kernel in base_kernels.items():
            if engine == 'gpflow':
                kernels_[base_model_name + "+" + name] = base_model_kern + kernel
                kernels_[base_model_name + "*" + name] = base_model_kern * kernel
            elif engine == 'sklearn':
                kernels_[base_model_name + "+" + name] = kernels_sklear.Sum(base_model_kern, kernel)
                kernels_[base_model_name + "*" + name] = kernels_sklear.Product(base_model_kern, kernel)
            elif engine == 'GPy':
                kernels_[base_model_name + "+" + name] = kernels_gpy.Add([base_model_kern, kernel])
                kernels_[base_model_name + "*" + name] = kernels_gpy.Prod([base_model_kern, kernel])
            else:
                pass

    return base_model_kern