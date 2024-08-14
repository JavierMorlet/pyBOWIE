# *******************************************************
# ****** Import libraries ******
# *******************************************************

import numpy as np
import multiprocess as mp
from sklearn.impute import SimpleImputer
from types import SimpleNamespace

# *******************************************************
# ****** Errors ******
# *******************************************************

def Errors(d_nc, d_nd, design_type, surrogate, engine, constraints_method, AF_name):

    valid_var_type = ["continuous", "integer", "categorical"]
    valid_design_type = ["random", "LHS", "Sobol", "Halton"]
    valid_surrogate_name = ["GP", "SGP"]
    valid_engines = ["gpflow", "sklearn", "GPy"]
    valid_constraints_method = ["PoF", "GPC"]
    valid_af_names = ['UCB', 'EI', 'PI']

    if d_nc <= 0 and d_nd <= 0:
        raise ValueError("Not valid variable type, valid names are:", *valid_var_type)

    if design_type not in valid_design_type:
        raise ValueError("Not valid design type, valid names are:", *valid_design_type)
    
    if surrogate not in valid_surrogate_name:
        raise ValueError("Not valid acquisition function name, valid names are:", *valid_surrogate_name)
    
    if engine not in valid_engines:
        raise ValueError("Not valid engine name, valid names are:", *valid_engines)
    
    if constraints_method not in valid_constraints_method:
        raise ValueError("Not valid method for constraints, valid names are:", *valid_constraints_method)

    if AF_name not in valid_af_names:
        raise ValueError("Not valid acquisition function name, valid names are:", *valid_af_names)
    
    return None

# *******************************************************
# ****** Best_values ******
# *******************************************************

def Flatten(l):

    return [item for sublist in l for item in sublist]

# *******************************************************
# ****** Eval_fun ******
# *******************************************************

def Eval_fun(x, jobs, function):

    if jobs == 1:
        x_new = np.array(x).reshape(1,-1)
        f_new = function(x_new)
    else:
        x_new = [x[i].reshape(1,-1) for i in range(jobs)]
        with mp.Pool(jobs) as pool:
            f_new = pool.map(function, x_new)
        
    return f_new

# *******************************************************
# ****** Eval_constrains ******
# *******************************************************

def Eval_const(x, const, n_const, const_method):

    if const == None: 
        y = None
        g = None
    else:
        if const_method == "PoF":
            data = [eval(const[i], None, {"x": x}) for i in range(n_const)]
        elif const_method == "GPC":
            data = [eval(const[i]['constraint'], None, {"x": x}) for i in range(n_const)]
        else:
            pass

        g = np.array(data).T

        if const_method == "PoF":
            y = g <= 0
            y = (y == True).all(axis=1)
        elif const_method == "GPC":
            y = (g == True).all(axis=1)
            g = y.astype(int).reshape(-1,1)
        else:
            pass

    return g, y

# *******************************************************
# ****** Imputation ******
# *******************************************************

def Imputation(g, const, const_method):

    if const == None: 
        g_new = None
    else:
        if const_method == "PoF":
            if np.isnan(g).any() == True:
                imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
                g_new = imputer.fit_transform(g)
            else:
                g_new = g
        elif const_method == "GPC":
            g_new = g
        else:
            pass

    return g_new

# *******************************************************
# ****** Best_values ******
# *******************************************************

def Best_values(x, f, y, sense):

    if y is None:
        pass
    else:
        f = f[y]
        x = x[y]

    if sense == "maximize":
        ix_best = np.argmax(f)
        f_best = np.max(f)
    elif sense == "minimize":
        ix_best = np.argmin(f)
        f_best = np.min(f)
    x_best = x[ix_best]

    return x_best, f_best

# *******************************************************
# ****** Check_if_improvement ******
# *******************************************************

def Check_if_improvement(f_new, f_best, jobs, sense):

    if jobs == 1:
        f_best_new = f_new
    else:
        if sense == "maximize":
            f_best_new = np.max(f_new)
        elif sense == "minimize":
            f_best_new = np.min(f_new)
        else:
            pass
    
    flag = 0
    
    if sense == "maximize":
        if f_best_new > f_best:
            flag = 1
    elif sense == "minimize":
        if f_best_new < f_best:
            flag = 1
    else:
        pass

    return flag

# *******************************************************
# ****** Regret ******
# *******************************************************

def Regret(f_true, x, n_elements, model, engine):

    # Return an average of the reward
    if engine == 'gpflow':
        f_pred, _ = model.predict_f(x)
    elif engine == 'sklearn':
        f_pred, _ = model.predict(x, return_std=True)
    elif engine == 'GPy':
        f_pred, _ = model.predict(x)
    else:
        pass

    rt = [(f_true[i] - f_pred[i]) for i in range(n_elements)]
    rt = sum(rt)

    return rt/n_elements

# *******************************************************
# ****** Print_header ******
# *******************************************************

def Print_header(names, x_symb_names, dims):

    if names is None:
        header = 'ite  ' +  '  f      ' + str(x_symb_names[0])
    else:
        header = 'ite  ' +  '  f      ' + str(names[0])
    for i in range(1, dims):
        if names is None:
            header += '      ' + str(x_symb_names[i])
        else:
            header += '      ' + str(names[i])
    
    return header

# *******************************************************
# ****** Print_results ******
# *******************************************************

def Print_results(x, f, cat_val):

    if cat_val == 0:
        x = x.reshape(-1)
        x_print = ["%.5f" % value if 1e-3 < abs(value) < 1e3 else "%0.1e" % value for value in x]
    else:
        #x = x[0]
        x_print = [] 
        for value in x:
            if type(value) == str:
                x_print.append(value)
            elif type(value) == float:
                if 1e-3 < abs(value) < 1e3:
                    x_print.append("%.5f" % value)
                else:
                    x_print.append("%0.1e" % value)
    
    f_print = "%.5f" % f if 1e-3 < abs(f) < 1e3 else "%0.1e" % f
        
    return x_print, f_print

# *******************************************************
# ****** Create_results ******
# *******************************************************

def Create_results(x_best, f_best, x, f, x_l, x_u, dims, max_iter, points, design, af_params, constraints_method, rt, models_const, model):

    res = {'x_best': x_best, 'f_best': f_best, 
           'x_init': x[0:points], 'f_init': f[0:points], 
           'x_iters': x[points:-1], 'f_iters': f[points:-1],
           'x_l': x_l, 'x_u': x_u, 
           'dims': dims,
           'iters': max_iter, 
           'initial_design': design,
           'initial_points': points,
           'xi': af_params['xi'], 
           'acquisition_function': af_params['AF_name'],
           'regret': rt,
           'constraint_method': constraints_method,
           'models_constraints': models_const,
           'model': model}
    
    return SimpleNamespace(**res)