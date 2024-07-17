# *******************************************************
# ****** Import libraries ******
# *******************************************************

import GPy
import numpy as np
import properscoring as ps
from sklearn.model_selection import train_test_split

# *******************************************************
# ****** Train_model ******
# *******************************************************

def Train_model(x, f, kernel, surrogate, n_restarts):

    def Select_model(x, f, kernel, surrogate):

        if surrogate == "GP":
            model = GPy.models.GPRegression(x, f, kernel)
        elif surrogate == "SGP":
            model = GPy.models.SparseGPRegression(x, f, kernel)
        else:
            pass
        
        return model
    
    model = Select_model(x, f, kernel, surrogate)
    model.optimize(optimizer='lbfgsb', max_iters=1000, messages=False)
    model.optimize_restarts(num_restarts=n_restarts, verbose=False)

    return model

# *******************************************************
# ****** Train_models_const ******
# *******************************************************

def Train_models_constraints(x, g, constraints, n_constraints, constraints_method):

    
    def Train_PoF(x, g, n_const):
        
        models = []
        for i in range(n_const):
            model = GPy.models.GPRegression(x, g[:,i].reshape(-1,1))
            model.optimize()
            models.append(model)

        return models
    
    def Train_GPC(x, g):

        model = GPy.models.GPClassification(x, g)
        model.optimize()

        return model

    if constraints == None: 
        models_const = None
    else:
        if constraints_method == "PoF":
            models_const = Train_PoF(x, g, n_constraints)
        elif constraints_method == "GPC":
            models_const = Train_GPC(x, g)
        else:
            pass

    return models_const

# *******************************************************
# ****** Kernel_discovery ******
# *******************************************************

def Kernel_discovery(x, f, dims, surrogate, evals):

    def Search(x_train, x_test, f_train, f_test, n, kernels, surrogate):

        BIC, CRPS, models = [], [], {}

        for name, kernel in kernels.items():
            if surrogate == "GP":
                model = GPy.models.GPRegression(x_train, f_train.reshape(-1,1), kernel)
            elif surrogate == "SGP":
                model = GPy.models.SparseGPRegression(x_train, f_train.reshape(-1,1), kernel)
            else:
                pass
            #model = GPy.models.GPRegression(x_train, f_train.reshape(-1,1), kernel)
            model.optimize()
            ll, k = model.log_likelihood(), model._size_transformed()
            BIC.append((-2*ll + k*np.log(n)).item())
            mean, var = model.predict(x_test)
            CRPS.append(ps.crps_gaussian(f_test, mean, var).mean())
            models[name] = model
        
        return BIC, CRPS, models

    def Get_best_model(BIC, CRPS, models, alpha = 0.5):

        normalized_bics = (BIC - np.min(BIC)) / (np.max(BIC) - np.min(BIC))
        normalized_crpss = (CRPS - np.min(CRPS)) / (np.max(CRPS) - np.min(CRPS))

        # Combine scores (example: equal weight)
        
        scores = alpha * normalized_bics + (1-alpha) * normalized_crpss
        models_ordered = [x for _, x in sorted(zip(scores, models.keys()))]
        
        return {models_ordered[0]: models[models_ordered[0]]}
    
    x_train, x_test, f_train, f_test = train_test_split(x, f, test_size=0.2)
    n = x_train.shape[0]
    # Base kernels: SE, Periodic, Linear, RatQuad  
    kernels = {"linear": GPy.kern.Linear(input_dim=dims),
            "RBF": GPy.kern.RBF(input_dim=dims, variance=1.0, lengthscale=1.0),
            "Mater_52": GPy.kern.Matern52(input_dim=dims, variance=1.0, lengthscale=1.0),
            "Periodic": GPy.kern.StdPeriodic(input_dim=dims, variance=1.0, lengthscale=1.0, period=1.0)
            }

    for i in range(evals):
        base_kernels = kernels.copy()
        BIC, CRPS, models = Search(x_train, x_test, f_train, f_test, n, base_kernels, surrogate)
        best_model = Get_best_model(BIC, CRPS, models)
        base_model = list(best_model.values())[0]
        base_model_name = list(best_model.keys())[0]
        base_model_kern = base_model.kern
        if i == evals: break
        kernels = {}
        for name, kernel in base_kernels.items():
            kernels[base_model_name + "+" + name] = GPy.kern.Add([base_model_kern, kernel])
            kernels[base_model_name + "*" + name] = GPy.kern.Prod([base_model_kern, kernel])

    return base_model_kern