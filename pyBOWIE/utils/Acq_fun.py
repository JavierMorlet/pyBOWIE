import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler

# *******************************************************
# ****** UCB ******
# *******************************************************

def UCB(mean, std, xi):
    # Predict the model
    af = mean + xi*std
    
    return af

# *******************************************************
# ****** PI ******
# *******************************************************

def PI(mean, std, xi, x_best):

    with np.errstate(divide='warn'):
        imp = mean - x_best - xi
        Z = imp / std
        af = norm.cdf(Z)
        af[std == 0.0] = 0.0

    return af

# *******************************************************
# ****** EI ******
# *******************************************************

def EI(mean, std, xi, x_best):
    
    with np.errstate(divide='warn'):
        imp = mean - x_best - xi
        Z = imp / std
        af = imp * norm.cdf(Z) + std * norm.pdf(Z)
        af[std == 0.0] = 0.0
    
    return af

# *******************************************************
# ****** PoF ******
# *******************************************************

def PoF(x, models, engine):
    
    pof = []

    for model in models:
        if engine == 'gpflow':
            mean, std = model.predict_f(x)
            mean, var = mean.numpy(), var.numpy()
        elif engine == 'sklearn':
            mean, std = model.predict(x, return_std=True)
        elif engine == 'GPy':
            mean, std = model.predict(x)
        else:
            pass
        with np.errstate(divide='warn'):
            Z = -mean/std
            af = norm.cdf(Z)
            af[std == 0.0] = 0.0
            pof.append(af)

    return np.prod(np.array(pof), axis=0)

# *******************************************************
# ****** Prob_GPC ******
# *******************************************************

def Prob_GPC(x, model, engine):

    if engine == 'gpflow':
        Fmean, _ = model.predict_f(x)
        gpc = model.likelihood.invlink(Fmean)
        gpc = gpc.numpy()
    elif engine == 'sklearn':
        gpc = model.predict_proba(x)[:,1]
        gpc = gpc.reshape(-1, 1)
    elif engine == 'GPy':
        gpc, _ = model.predict(x)

    return gpc

# *******************************************************
# ****** AF ******
# *******************************************************

def AF(x, params, constraints_method, model, models_const, engine):

    xi, _, x_best, AF_name = params.values()

    if engine == 'gpflow':
        mean, std = model.predict_f(x)
        mean, std = mean.numpy(), std.numpy()
    elif engine == 'sklearn':
        mean, std = model.predict(x, return_std=True)
    elif engine == 'GPy':
        mean, std = model.predict(x)
    else:
        pass

    if AF_name == 'UCB':
        score = UCB(mean, std, xi)
    elif AF_name == 'PI':
        score = PI(mean, std, xi, x_best)
    elif AF_name == 'EI':
        score = EI(mean, std, xi, x_best)
    else:
        pass

    if models_const is None:
        pass
    else:
        if constraints_method == "PoF":
            score_const = PoF(x, models_const, engine)
            score_const = score_const.reshape(-1,1)
        elif constraints_method == "GPC":
            score_const = Prob_GPC(x, models_const, engine)
        else:
            pass
        scaler = MinMaxScaler()
        score_const = scaler.fit_transform(score_const)
        score = score.reshape(-1,1)
        score_const = score_const.reshape(-1,1)
        #print(score.shape, score_const.shape)
        score = score*score_const

    return score
