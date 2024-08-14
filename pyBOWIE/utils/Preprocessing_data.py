# *******************************************************
# ****** Import libraries ******
# *******************************************************

import numpy as np
import pandas as pd
from prince import PCA, MCA, FAMD
from sklearn.metrics import r2_score
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split

# *******************************************************
# ****** U_scaling ******
# *******************************************************

def U_scaling(U, d_nc, d_ni, d_nq, x_l, x_u, int_val, cat_val, points, problem_type):

    def Scale_continuous(U, x_l, x_u):
        
        return (x_u - x_l) * U + x_l

    def Scale_discrete(U, d_nd, disc_val, points):

        size_y = [len(disc_val[i]) for i in range(d_nd)]
        l = [[i/size_y[j] for i in range(size_y[j]+1)] for j in range(d_nd)]
        y_new = np.empty([points, d_nd], dtype=object)
        for i in range(len(l)):
            l[i][-1] = l[i][-1] + 0.0001
        for i in range(d_nd):
            for j in range(len(U[:,i])):
                for k in range(len(l[i])-1):
                    if l[i][k] <= U[j,i] < l[i][k+1]:
                        y_new[j,i] = disc_val[i][k]
                        break
        
        return y_new

    if problem_type == "Continuous":
        vars = Scale_continuous(U, x_l, x_u)
    elif problem_type == "Discrete":
        vars = Scale_discrete(U, d_ni, int_val, points).astype(float)
    elif problem_type == "Categorical":
        vars = Scale_discrete(U, d_nq, cat_val, points)
    elif problem_type == "Mixed_integer":
        xc_vars = Scale_continuous(U[:,:d_nc], x_l, x_u)
        if d_ni == 1:
            xi_vars = Scale_discrete(U[:,d_nc:].reshape(-1,1), d_ni, int_val, points)
        else:
            xi_vars = Scale_discrete(U[:,d_nc:], d_ni, int_val, points)
        vars = np.hstack((xc_vars, xi_vars)).astype(float)
    elif problem_type == "Mixed_categorical":
        xc_vars = Scale_continuous(U[:,:d_nc], x_l, x_u)
        if d_ni == 1:
            xq_vars = Scale_discrete(U[:,d_nc:].reshape(-1,1), d_nq, cat_val, points)
        else:
            xq_vars = Scale_discrete(U[:,d_nc:], d_nq, cat_val, points)
        vars = np.hstack((xc_vars, xq_vars))
    elif problem_type == "Mixed_discrete":
        if d_ni == 1:
            xi_vars = Scale_discrete(U[:,0:d_ni].reshape(-1,1), d_ni, int_val, points)
        else:
            xi_vars = Scale_discrete(U[:,0:d_ni], d_ni, int_val, points)
        if d_ni == 1:
            xq_vars = Scale_discrete(U[:,d_ni:].reshape(-1,1), d_nq, cat_val, points)
        else:
            xq_vars = Scale_discrete(U[:,d_ni:], d_nq, cat_val, points)
        vars = np.hstack((xi_vars, xq_vars))
    elif problem_type == "Mixed_all":
        xc_vars = Scale_continuous(U[:,:d_nc], x_l, x_u)
        d_nt = d_nc + d_ni
        if d_ni == 1:
            xi_vars = Scale_discrete(U[:,d_nc:d_nt].reshape(-1,1), d_ni, int_val, points)
        else:
            xi_vars = Scale_discrete(U[:,d_nc:d_nt], d_ni, int_val, points)
        if d_ni == 1:
            xq_vars = Scale_discrete(U[:,d_nt:].reshape(-1,1), d_nq, cat_val, points)
        else:
            xq_vars = Scale_discrete(U[:,d_nt:], d_nq, cat_val, points)
        vars = np.hstack((xc_vars, xi_vars, xq_vars))
    else:
        pass

    return vars

# *******************************************************
# ****** U_scaling ******
# *******************************************************

def U_inverse_scaling(X, d_nc, d_ni, d_nq, x_l, x_u, int_val, cat_val, points, problem_type):

    def Scale_continuous_inv(X, x_l, x_u):
        
        return (X - x_l)/(x_u - x_l)
    
    def Scale_discrete_inv(X, d_nd, disc_val, points):

        y_new = np.empty([points+1, d_nd])
        size_y = [len(disc_val[i]) for i in range(d_nd)]
        for i in range(d_nd):
            #print(int_val)
            for j in range(len(disc_val[i])):
                y_new[:,i][np.where(disc_val[i][j] == X[:,i])[0].tolist()] = j/size_y[i]
        
        return y_new

    if problem_type == "Continuous":
        vars = Scale_continuous_inv(X, x_l, x_u)
    elif problem_type == "Discrete":
        vars = Scale_discrete_inv(X, d_ni, int_val, points).astype(float)
    elif problem_type == "Categorical":
        vars = Scale_discrete_inv(X, d_nq, cat_val, points)
    elif problem_type == "Mixed_integer":
        xc_vars = Scale_continuous_inv(X[:,:d_nc], x_l, x_u)
        if d_ni == 1:
            xi_vars = Scale_discrete_inv(X[:,d_nc:].reshape(-1,1), d_ni, int_val, points)
        else:
            xi_vars = Scale_discrete_inv(X[:,d_nc:], d_ni, int_val, points)
        vars = np.hstack((xc_vars, xi_vars)).astype(float)
    elif problem_type == "Mixed_categorical":
        xc_vars = Scale_continuous_inv(X[:,:d_nc], x_l, x_u)
        if d_nq == 1:
            xq_vars = Scale_discrete_inv(X[:,d_nc:].reshape(-1,1), d_nq, cat_val, points)
        else:
            xq_vars = Scale_discrete_inv(X[:,d_nc:], d_nq, cat_val, points)
        vars = np.hstack((xc_vars, xq_vars))
    elif problem_type == "Mixed_discrete":
        if d_ni == 1:
            xi_vars = Scale_discrete_inv(X[:,0:d_ni].reshape(-1,1), d_ni, int_val, points)
        else:
            xi_vars = Scale_discrete_inv(X[:,0:d_ni], d_ni, int_val, points)
        if d_ni == 1:
            xq_vars = Scale_discrete_inv(X[:,d_ni:].reshape(-1,1), d_nq, cat_val, points)
        else:
            xq_vars = Scale_discrete_inv(X[:,d_ni:], d_nq, cat_val, points)
        vars = np.hstack((xi_vars, xq_vars))
    elif problem_type == "Mixed_all":
        xc_vars = Scale_continuous_inv(X[:,:d_nc], x_l, x_u)
        d_nt = d_nc + d_ni
        if d_ni == 1:
            xi_vars = Scale_discrete_inv(X[:,d_nc:d_nt].reshape(-1,1), d_ni, int_val, points)
        else:
            xi_vars = Scale_discrete_inv(X[:,d_nc:d_nt], d_ni, int_val, points)
        if d_ni == 1:
            xq_vars = Scale_discrete_inv(X[:,d_nt:].reshape(-1,1), d_nq, cat_val, points)
        else:
            xq_vars = Scale_discrete_inv(X[:,d_nt:], d_nq, cat_val, points)
        vars = np.hstack((xc_vars, xi_vars, xq_vars))
    else:
        pass

    return vars

# *******************************************************
# ****** Trans_data_to_pandas ******
# *******************************************************

def Trans_data_to_pandas(data, d_nc, dims, problem_type):

    if problem_type == "Mixed_categorical" or problem_type == "Mixed_discrete" or problem_type == "Mixed_all":
        cols = np.arange(1,dims+1)
        cols = cols.astype(str)
        data = pd.DataFrame(data, columns=cols)
        data[cols[0:d_nc]] = data[cols[0:d_nc]].astype(float)
    else:
        data = pd.DataFrame(data)

    return data

# *******************************************************
# ****** Reduce ******
# *******************************************************

def Reduce(data, d_nc, dims, problem_type, reducer):
    
    data_pd = Trans_data_to_pandas(data, d_nc, dims, problem_type)
    
    return np.array(reducer.transform(data_pd))

# *******************************************************
# ****** Inverse ******
# *******************************************************

def Inverse(data, inverter_transform, inverter):

    def Back_projection(data, model):

        predictions = {}
        for col, mod in model.items():
            predictions[col] = mod.predict(data)
        
        return predictions
    
    data = pd.DataFrame(data)
    if inverter_transform == "yes":
        data_reconstructed = inverter.inverse_transform(data)
    elif inverter_transform == "no":
        predictions = Back_projection(data, inverter)
        data_reconstructed = np.vstack(([predictions[i] for i in range(len(inverter))])).T

    return data_reconstructed

# *******************************************************
# ****** Train_reducer ******
# *******************************************************

def Train_reducer(data, d_nc, dims, problem_type, n_components, reducer):

    data_pd = Trans_data_to_pandas(data, d_nc, dims, problem_type)
    if reducer.__module__ == 'prince.pca':
        reducer_train = PCA()
    elif reducer.__module__ == 'prince.famd':
        reducer_train = FAMD()
    elif reducer.__module__ == 'prince.mca':
        reducer_train = MCA()
    else:
        reducer_train = reducer

    reducer_train.n_components = n_components
    data_red = np.array(reducer_train.fit_transform(data_pd))
    
    return data_red, reducer_train

# *******************************************************
# ****** Train_inverter ******
# *******************************************************

def Train_inverter(data, data_reduced, dims, models):

    for i in range(dims):
        model = models[i]
        n_estimators_ = model.n_estimators
        model.set_params(n_estimators=n_estimators_+5)
        model.fit(data_reduced, data[:,i])
        models[i] = model

    return models

# *******************************************************
# ****** Find_reducer ******
# *******************************************************

def Find_reducer(data, d_nc, dims, problem_type):

    def Find_n_components_PCA(x_train, x_test, n, reducer, error_min = 0.25, n_max = 5):

        def Train_Red_comp(x_train, n_trial, reducer):

            reducer.n_components = n_trial
            reducer = reducer.fit(x_train)
            
            return reducer
        
        def Reconstruction_error(x_test, reducer):

            X_test_red = reducer.transform(x_test)
            X_test_reconstructed = reducer.inverse_transform(X_test_red)
            
            return r2_score(x_test, X_test_reconstructed)

        for n in range(1, n_max):
            reducer = Train_Red_comp(x_train, n, reducer)
            error = Reconstruction_error(x_test, reducer)
            if error >= error_min:
                break

        return n
    
    def Find_n_components(data, n, reducer, var_max = 75, n_max = 6):

        def Red(reducer, data, n_trial):

            reducer.n_components = n_trial
            reducer = reducer.fit(data)

            return reducer.cumulative_percentage_of_variance_[-1]
        
        for n in range(1, n_max):
            variance = Red(reducer, data, n)
            if variance >= var_max:
                break

        return n

    # Main program
    
    if problem_type == "Continuous" or problem_type == "Discrete" or problem_type == "Mixed_integer":
        reducer = PCA()
        reducer_search = PCA()
        reducer_train = PCA()
    elif problem_type == "Mixed_categorical" or problem_type == "Mixed_discrete" or problem_type == "Mixed_all":
        reducer = FAMD()
        reducer_train = FAMD()
    elif problem_type == "Categorical":
        reducer = MCA()
        reducer_train = MCA()
    else:
        pass

    data_pd = Trans_data_to_pandas(data, d_nc, dims, problem_type)
    if problem_type == "Continuous":
        x_train, x_test = train_test_split(data_pd)
        dims_red = Find_n_components_PCA(x_train, x_test, dims, reducer_search)
        reducer_train = PCA(n_components=dims_red)
        reducer_train = reducer_train.fit(data_pd)
    else:
        dims_red = Find_n_components(data_pd, dims, reducer_train)

    return reducer, reducer_train, dims_red

# *******************************************************
# ****** Find_inverter ******
# *******************************************************

def Find_inverter(data, data_reduced, d_nc, d_nd, dims, problem_type):

    def Find_best_models(x, z, dims, problem_type, method):

        param_clas = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }

        if (problem_type == "Discrete" and dims < 5) or (problem_type == "Categorical" and dims < 5) or (problem_type == "Mixed_discrete" and dims < 5):
            random_search = RandomizedSearchCV(method(), param_distributions=param_clas, n_iter=15, cv=dims)
        elif (problem_type == "Mixed_integer" and d_nd < 5) or (problem_type == "Mixed_categorical" and d_nd < 5) or (problem_type == "Mixed_all" and d_nd < 5):
            random_search = RandomizedSearchCV(method(), param_distributions=param_clas, n_iter=15, cv=d_nd)
        else:
            random_search = RandomizedSearchCV(method(), param_distributions=param_clas, n_iter=15, cv=5)
        random_search.fit(x, z)
        best_params = random_search.best_params_
        best_model = method(**best_params)
        
        return best_model

    def Train_models(x, z, problem_type, method):

        models = {}
        for col in z.columns:
            model = Find_best_models(x, z[col], problem_type, dims, method)
            model.warm_start = True
            models[col] = model

        return models
    
    data_pd = Trans_data_to_pandas(data, d_nc, dims, problem_type)

    if problem_type == "Continuous":
        model = Train_models(data_reduced, data_pd, problem_type, ExtraTreesRegressor)
    elif problem_type == "Discrete" or problem_type == "Categorical" or problem_type == "Mixed_discrete":
        model = Train_models(data_reduced, data_pd, problem_type, ExtraTreesClassifier)
    elif problem_type == "Mixed_integer" or problem_type == "Mixed_categorical" or problem_type == "Mixed_all":
        model_1 = Train_models(data_reduced, data_pd.iloc[:,:d_nc], problem_type, ExtraTreesRegressor)
        model_2 = Train_models(data_reduced, data_pd.iloc[:,d_nc:], problem_type, ExtraTreesClassifier)
        model = {**model_1, **model_2}
        model_names = np.arange(len(model))
        model = dict(zip(model_names, list(model.values())))

    return model

# *******************************************************
# ****** Red_bounds ******
# *******************************************************

def Red_bounds(x_l, x_u, int_val, cat_val, d_nc, d_ni, d_nq, dims, dims_tau, problem_type, reducer):

    if reducer is None:
        if problem_type == "Continuous":
            x_l_tau, x_u_tau = x_l, x_u
        else:
            x_l_tau, x_u_tau = np.zeros(dims), np.ones(dims)
    else:
        if problem_type == "Continuous":
            bounds = [[x_l[i], x_u[i]] for i in range(dims)]
        elif problem_type == "Discrete":
            bounds = [[int_val[i][0], int_val[i][-1]] for i in range(dims)]
        elif problem_type == "Mixed_integer":
            xc_bounds = [[x_l[i], x_u[i]] for i in range(d_nc)]
            xi_bounds = [[int_val[i][0], int_val[i][-1]] for i in range(d_ni)]
            bounds = np.vstack((xc_bounds, xi_bounds)).tolist()
        elif problem_type == "Categorical":
            bounds = [[cat_val[i][j] for j in range(len(cat_val[i]))] for i in range(dims)]
        elif problem_type == "Mixed_categorical":
            xc_bounds = [[x_l[i], x_u[i]] for i in range(d_nc)]
            xq_bounds = [[cat_val[i][j] for j in range(len(cat_val[i]))] for i in range(d_nq)]
            bounds = np.vstack((xc_bounds, xq_bounds)).tolist()
        elif problem_type == "Mixed_discrete":
            xi_bounds = [[int_val[i][0], int_val[i][-1]] for i in range(d_ni)]
            xq_bounds = [[cat_val[i][j] for j in range(len(cat_val[i]))] for i in range(d_nq)]
            bounds = np.vstack((xi_bounds, xq_bounds)).tolist()
        elif problem_type == "Mixed_discrete":
            xc_bounds = [[x_l[i], x_u[i]] for i in range(d_nc)]
            xi_bounds = [[int_val[i][0], int_val[i][-1]] for i in range(d_ni)]
            xq_bounds = [[cat_val[i][j] for j in range(len(cat_val[i]))] for i in range(d_nq)]
            bounds = np.vstack((xc_bounds, xi_bounds, xq_bounds)).tolist()
        mesh = np.meshgrid(*bounds)
        x_mesh = np.array(mesh).T.reshape(-1, dims)
        x_mesh_red = Reduce(x_mesh, d_nc, dims, problem_type, reducer)
        x_l_tau = np.array([x_mesh_red[0,i] for i in range(dims_tau)])
        x_u_tau = np.array([x_mesh_red[-1,i] for i in range(dims_tau)])

    return x_l_tau, x_u_tau

# *******************************************************
# ****** Red_bounds ******
# *******************************************************

def X_new_scaling(x_new_tau, d_nc, d_ni, d_nq, x_l, x_u, int_val, cat_val, jobs, reducer, inverter_transform, inverter, problem_type):

    if reducer == "no":
        if problem_type == "Continuous":
            x_new = np.array(x_new_tau)
        else:
            x_new_tau = np.array(x_new_tau)
            x_new = U_scaling(x_new_tau, d_nc, d_ni, d_nq, x_l, x_u, int_val, cat_val, jobs, problem_type)
    else:
        x_new = Inverse(x_new_tau, inverter_transform, inverter)
        x_new = np.array(x_new)

    return x_new