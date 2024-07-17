import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler

# *******************************************************
# ****** UCB ******
# *******************************************************

def UCB(x, xi, model):
    # Predict the model
    mean, std = model.predict(x)
    af = mean + xi*std
    
    return af

# *******************************************************
# ****** PI ******
# *******************************************************

def PI(x, x_best, xi, model):

    mean, std = model.predict(x)
    with np.errstate(divide='warn'):
        imp = mean - x_best - xi
        Z = imp / std
        af = norm.cdf(Z)
        af[std == 0.0] = 0.0

    return af

# *******************************************************
# ****** EI ******
# *******************************************************

def EI(x, x_best, xi, model):
    
    mean, std = model.predict(x)
    with np.errstate(divide='warn'):
        imp = mean - x_best - xi
        Z = imp / std
        af = imp * norm.cdf(Z) + std * norm.pdf(Z)
        af[std == 0.0] = 0.0
    
    return af

# *******************************************************
# ****** PoF ******
# *******************************************************

def PoF(x, models):
    
    pof = []

    for model in models:
        mean, std = model.predict(x)
        with np.errstate(divide='warn'):
            Z = -mean/std
            af = norm.cdf(Z)
            af[std == 0.0] = 0.0
            pof.append(af)

    return np.prod(np.array(pof), axis=0)

# *******************************************************
# ****** Prob_GPC ******
# *******************************************************

def Prob_GPC(x, model):

    gpc, _ = model.predict(x)

    return gpc

# *******************************************************
# ****** AF ******
# *******************************************************

def AF(x, params, constraints_method, model, models_const):

    xi, _, x_best, AF_name = params.values()

    if AF_name == 'UCB':
        score = UCB(x, xi, model)
    elif AF_name == 'PI':
        score = PI(x, xi, x_best, model)
    elif AF_name == 'EI':
        score = EI(x, xi, x_best, model)
    else:
        pass

    if models_const is None:
        pass
    else:
        if constraints_method == "PoF":
            score_const = PoF(x, models_const)
        elif constraints_method == "GPC":
            score_const = Prob_GPC(x, models_const)
        else:
            pass
        scaler = MinMaxScaler()
        score_const = scaler.fit_transform(score_const)
        score = score.reshape(-1,1)
        score_const = score_const.reshape(-1,1)
        score = score*score_const

    return score
