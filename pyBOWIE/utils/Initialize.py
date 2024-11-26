# *******************************************************
# ****** Import libraries ******
# *******************************************************

import time
import warnings
import numpy as np
import sympy as sp
import multiprocess as mp
from scipy.stats import qmc
from scipy.stats import chi2
from gpflow.kernels import RBF as RBF_gpflow
from sklearn.gaussian_process.kernels import RBF as RBF_sklearn
from ..utils.Preprocessing_data import U_scaling, U_inverse_scaling, Train_reducer, Train_inverter, Find_reducer, Find_inverter, Reduce
from ..utils.Models import Kernel_discovery

try:
    from GPy.kern import RBF as RBF_gpy
except:
    pass

# *******************************************************
# ****** Space ******
# *******************************************************

def Space(domain):

    """ 
    Gets tuple of dimension of the problem, lower and upper bounds of the continuous variables, values for the discrete variables, and names of the variables.
    """

    dims = len(domain)
    d_nc = 0
    d_ni = 0
    d_nq = 0
    x_l = []
    x_u = []
    int_val = []
    cat_val = []
    names = []

    for i in range(dims):
        try:
            names.append(domain[i]['name'])
        except:
            names = 0
        if domain[i]['type'] == "continuous":
            d_nc += 1
            x_l.append(domain[i]['domain'][0])
            x_u.append(domain[i]['domain'][1])
        elif domain[i]['type'] == "integer":
            d_ni += 1
            int_val.append(domain[i]['domain'])
        elif domain[i]['type'] == "categorical":
            d_nq += 1
            features = domain[i]['domain']
            cat_val.append(features)
        else:
            pass
            
    x_l, x_u = np.array(x_l), np.array(x_u)
    d_nd = d_ni + d_nq
    if d_nd == 0:
        int_val = 0
    if d_nq == 0:
        cat_val = 0
    if names == 0:
        names = None

    return dims, d_nc, d_ni, d_nq, d_nd, x_l, x_u, int_val, cat_val, names

# *******************************************************
# ****** Problem_type ******
# *******************************************************

def Problem_type(d_nc, d_ni, d_nq):

    """ 
    Returns the type of problem depending of the characteristics of the inputs:
    * Continuous if there are only continuous variables
    * Discrete if there are only discrete variables
    * Mixed if there are continuous and discrete variables
    """
    
    if d_nc > 0 and d_ni == 0 and d_nq == 0:
        problem_type = "Continuous"
    elif d_nc == 0 and d_ni > 0 and d_nq == 0:
        problem_type = "Discrete"
    elif d_nc == 0 and d_ni == 0 and d_nq > 0:
        problem_type = "Categorical"
    elif d_nc > 0 and d_ni > 0 and d_nq == 0:
        problem_type = "Mixed_integer"
    elif d_nc > 0 and d_ni == 0 and d_nq > 0:
        problem_type = "Mixed_categorical"
    elif d_nc == 0 and d_ni > 0 and d_nq > 0:
        problem_type = "Mixed_discrete"
    elif d_nc > 0 and d_ni > 0 and d_nq > 0:
        problem_type = "Mixed_all"
    else:
        pass

    return problem_type

# *******************************************************
# ****** Get_constraints ******
# *******************************************************

def Get_constraints(constraints, constraints_method):

    "Transforms tuple of constraints into lists"

    if constraints is None:
        const, d_nqonst = None, None
    else:
        d_nqonst = len(constraints)
        if constraints_method == "PoF":
            symbols = ['<=', '>=']
            const = []
            for i in range(d_nqonst):
                for s in symbols:
                    try:
                        index_const = constraints[i]['constraint'].index(s)
                        const.append(constraints[i]['constraint'][:index_const-1])
                    except:
                        pass
        elif constraints_method == "GPC":
            const = constraints
    
    return const, d_nqonst

# *******************************************************
# ****** Get_p_design ******
# *******************************************************

def Get_n_p_design(fun, d_nc, d_ni, d_nq, x_l, x_u, int_val, cat_val, c1_param, problem_type, design, n_p_design):

    """
    Computes the number of points of the initial design
    """

    # *************************
        # x0_trial
    # *************************

    def x0_trial(x_l, x_u, int_val, cat_val, d_nc, d_ni, d_nq, problem_type):
    
        """ 
        Perform a single random sampling of inputs
        """    

        # x_c0_rand
        def x_c0_rand(x_l, x_u, d_nc):

            return np.random.uniform(x_l, x_u, size=(d_nc))
        
        # x_c0_rand
        def x_d0_rand(disc_val, d_ni):

            return np.array([np.random.choice(disc_val[i]) for i in range(d_ni)])
        
        # Main program
        if problem_type == "Continuous":
            var = x_c0_rand(x_l, x_u, d_nc)
        elif problem_type == "Discrete":
            var = x_d0_rand(int_val, d_ni)
        elif problem_type == "Categorical":
            var = x_d0_rand(cat_val, d_nq)
        elif problem_type == "Mixed_integer":
            xc_var = x_c0_rand(x_l, x_u, d_nc)
            xi_var = x_d0_rand(int_val, d_ni)
            var = np.hstack((xc_var, xi_var))
        elif problem_type == "Mixed_categorical":
            xc_var = x_c0_rand(x_l, x_u, d_nc)
            xq_var = x_d0_rand(cat_val, d_nq)
            var = np.hstack((xc_var, xq_var.astype(object)))
        elif problem_type == "Mixed_discrete":
            xi_var = x_d0_rand(int_val, d_ni)
            xq_var = x_d0_rand(cat_val, d_nq)
            var = np.hstack((xi_var, xq_var.astype(object)))
        elif problem_type == "Mixed_all":
            xc_var = x_c0_rand(x_l, x_u, d_nc)
            xi_var = x_d0_rand(int_val, d_ni)
            xq_var = x_d0_rand(cat_val, d_nq)
            var = np.hstack((xc_var, xi_var, xq_var.astype(object)))
        else:
            pass

        return var

    # *************************
        # Times_fun
    # *************************

    def Times_fun(fun, x):

        """ 
        Determines the computing time for a random evaluation (x0_trial) of the objective function
        """
        
        start = time.time()
        f_eval = fun(x.reshape(1,-1))
        end = time.time()
        time_eval = (end - start)

        if time_eval < 1e-10:
            time_eval = 1e-8
        
        return time_eval, f_eval

    # *************************
        # Points_initial_design
    # *************************

    def Points_initial_design(times, design, c1_param):

        """
        Generates the number of points depending on the cost of the evaluation of the function
        """

        if times <= 1:
            exp_param = 0.25
        else: 
            exp_param = 0.95
        points_D = int(c1_param - c1_param/(1+(1/times)**exp_param))
        # Adjust points if design is Sobol. 
        if design == "Sobol":
            points_D = int(np.ceil(np.sqrt(points_D))**2)
        else:
            pass
        # Adjust points if number is below 3.
        if points_D < 3:
            points_D = int(3)

        return points_D

    # *************************
        # Main program
    # *************************

    x_trial = x0_trial(x_l, x_u, int_val, cat_val, d_nc, d_ni, d_nq, problem_type)
    times, f_trial = Times_fun(fun, x_trial)

    if n_p_design == None:
        n_p_design = Points_initial_design(times, design, c1_param)
    else:
        x_trial, f_trial = None, None

    return n_p_design, x_trial, f_trial

# *******************************************************
# ****** U_Generator ******
# *******************************************************

def U_Generator(dims, points, design_type):

    """ 
    Computes unitary design matrix
    """

    # *************************
    # Random_design
    # *************************

    def Random_design(dims, points):

        return np.random.rand(points, dims)
    
    # *************************
    # QMC_design
    # *************************

    def QMC_design(dims, points, design_type):
        
        # Main program
        if design_type == "LHS":
            method = qmc.LatinHypercube(d=dims)
        elif design_type == "Sobol":
            method = qmc.Sobol(d=dims)
        elif design_type == "Halton":
            method = qmc.Halton(d=dims)
        else:
            pass

        return method.random(n=points)

    # *************************
    # Main program
    # *************************
    
    if design_type == "random":
        variables = Random_design(dims, points)
    elif design_type == "LHS" or design_type == "Sobol" or design_type == "Halton":
        variables = QMC_design(dims, points, design_type)
    else:
        pass
    
    return variables

# *******************************************************
# ****** Get_x_and_z ******
# *******************************************************

def Get_x_and_z(fun, x_0, f_0, dims, d_nc, d_ni, d_nq, x_l, x_u, int_val, cat_val, c1_param, problem_type, design_type, n_p_design):

    """ 
    Gets the training data in the problem bounds
    """

    if x_0 is None:
        if n_p_design is None:
            n_p_design, x_trial, f_trial = Get_n_p_design(fun, d_nc, d_ni, d_nq, x_l, x_u, int_val, cat_val, c1_param, problem_type, design_type, n_p_design)
            U = U_Generator(dims, n_p_design, design_type)
            x_0 = U_scaling(U, d_nc, d_ni, d_nq, x_l, x_u, int_val, cat_val, n_p_design, problem_type)
            f_0 = fun(x_0).reshape(-1,1)
            x, f = np.vstack((x_0, x_trial)), np.vstack((f_0, f_trial))
        else:
            U = U_Generator(dims, n_p_design, design_type)
            x = U_scaling(U, d_nc, d_ni, d_nq, x_l, x_u, int_val, cat_val, n_p_design, problem_type)
            f = fun(x).reshape(-1,1)
    else: 
        x = x_0
        if f_0 is None:
            f = fun(x).reshape(-1,1)
        else:
            f = f_0
        n_p_design = len(f)

    return x, f, n_p_design

# *******************************************************
# ****** Get_constraints ******
# *******************************************************

def Get_constraints(constraints, constraints_method):

    """ 
    Transforms tuple of constraints into list
    """

    if constraints is None:
        const, n_const = None, None
    else:
        n_const = len(constraints)
        if constraints_method == "PoF":
            symbols = ['<=', '>=']
            const = []
            for i in range(n_const):
                for s in symbols:
                    try:
                        if s == '<=':
                            index_const = constraints[i]['constraint'].index(s)
                            const.append(constraints[i]['constraint'][:index_const-1])
                        elif s == '>=':
                            index_const = constraints[i]['constraint'].index(s)
                            const.append(-constraints[i]['constraint'][:index_const-1])
                    except:
                        pass
        elif constraints_method == "GPC":
            const = constraints
    
    return const, n_const

# *******************************************************
# ****** Get_search_space_params ******
# *******************************************************

def Get_search_space_params(x, dims, d_nc, d_ni, d_nq, d_nd, x_l, x_u, int_val, cat_val, points, problem_type, reducer, inverter_transform):

    """ 
    Return the parameters for the search space of the new points:
    * If dimension reduction is performed, it returns the training set in the reduced space and the trained reduction method. 
    """

    if reducer == "no":
        if dims > 6: warnings.warn('You can reach maximum depth of recursion')
        if (problem_type == "Continuous") and (d_nc <= 5):
            reducer_name, reducer_trained, dims_tau, x_tau = "no", None, dims, x
        else:
            reducer_name, reducer_trained, dims_tau = "no", None, dims
            x_tau = U_inverse_scaling(x, d_nc, d_ni, d_nq, x_l, x_u, int_val, cat_val, points, problem_type)
    elif reducer is None:
        if (problem_type == "Continuous") and (d_nc <= 5):
            reducer_name, reducer_trained, dims_tau, x_tau = "no", None, dims, x
        else:
            reducer_name, reducer_trained, dims_tau = Find_reducer(x, d_nc, dims, problem_type)
            x_tau = Reduce(x, d_nc, dims, problem_type, reducer_trained)
    else:
        dims_tau = reducer.n_components
        reducer_name = reducer
        x_tau, reducer_trained = Train_reducer(x, d_nc, dims, problem_type, dims_tau, reducer)

    if inverter_transform == "yes":
        if reducer_name.__module__ == 'prince.mca' or reducer_name.__module__ == 'prince.famd':
            raise ValueError("Reducer module has no inverse_transform")
        else:
            inverter = reducer_trained
    elif inverter_transform == "no":
        if reducer_name == "no":
            inverter = None
        else:
            inverter = Find_inverter(x, x_tau, d_nc, d_nd, dims, problem_type)
            inverter = Train_inverter(x, x_tau, dims, inverter)
    else:
        pass
    
    return x_tau, reducer_name, reducer_trained, inverter, dims_tau

# *******************************************************
# ****** Get_x_mesh ******
# *******************************************************

def Get_x_mesh(lower_bound, upper_bound, dims_tau, c2_param):

    """ 
    
    """
    # *************************
    # Random_design
    # *************************

    def Points_mesh(dims, c2_param):

        points = int(np.ceil(2**(c2_param/dims)))
        if points < 3:
            points = 3

        return points
    
    # *************************
    # Mesh_design
    # *************************

    def Mesh_design(lower_bound, upper_bound, dims, points):

        lists = [np.linspace(lower_bound[i], upper_bound[i], points) for i in range(dims)]
        mesh = np.meshgrid(*lists)
        
        return np.array(mesh).T.reshape(-1, dims)
    
    # *************************
    # Main program
    # *************************
    
    n_p_mesh = Points_mesh(dims_tau, c2_param)
    x_mesh = Mesh_design(lower_bound, upper_bound, dims_tau, n_p_mesh)
    
    return [x_mesh], [n_p_mesh], n_p_mesh

# *******************************************************
# ****** Bounds ******
# *******************************************************

def Bounds(x_l, x_u, dims):

    bnds = np.sort(np.array([[x_l[i], x_u[i]] for i in range(dims)]))
    return tuple(map(tuple, bnds))

# *******************************************************
# ****** Get_n_jobs ******
# *******************************************************

def Get_n_jobs(n_jobs):

    if n_jobs == -1:
        jobs = mp.cpu_count()
    elif n_jobs == None:
        jobs = 1
    else:
        jobs = n_jobs
    
    return jobs

# *******************************************************
# ****** Get_kernel ******
# *******************************************************

def Get_kernel(x, z, dims, surrogate, beta, kernel, kern_discovery, kern_discovery_evals, engine):

    if kern_discovery == "yes":
        kernel_ = Kernel_discovery(x, z, dims, surrogate, beta, kern_discovery_evals, engine)
    elif kern_discovery == "no" and kernel is None:
        if engine == 'gpflow':
            kernel_ = RBF_gpflow()
        elif engine == 'sklearn':
            kernel_ = RBF_sklearn()
        elif engine == 'GPy':
            kernel_ = RBF_gpy(input_dim=dims, variance=1.0, lengthscale=1.0)
        else:
            pass
    else:
        kernel_ = kernel

    return kernel_

# *******************************************************
# ****** Iter_params ******
# *******************************************************

def Iter_params(dims, dims_tau, max_iter, alpha):

    x_symb = sp.Matrix(sp.symbols('x:' + str(dims_tau)))
    x_symb_names = sp.Matrix(sp.symbols('x:' + str(dims)))
    flag = 0
    q0 = 25
    qf = 95
    delta_q = (qf-q0)/max_iter
    q_inc = q0
    
    if dims > 1:
        chi = chi2.ppf(alpha, dims_tau)
    else:
        chi = None

    if max_iter < 10:
        update_param = 1
    elif max_iter < 51:
        update_param = int(max_iter/10)
    elif max_iter < 101:
        update_param = 5
    elif max_iter < 501:
        update_param = 10
    elif max_iter < 1001:
        update_param = 50
    else:
        pass

    return x_symb, x_symb_names, flag, q0, q_inc, delta_q, chi, update_param

# *******************************************************
# ****** AF_params ******
# *******************************************************

def AF_params(z, xi_0, xi_f, xi_decay, iters, AF_name, sense):

    if xi_decay == "yes":
        xi_decay = (xi_f/xi_0)**(1/iters)
    elif xi_decay == "no":
        xi_decay = 1
    else:
        pass

    if sense == "maximize":
        f_best = np.max(z)
    elif sense == "minimize":
        f_best = np.min(z)
    else:
        pass

    return {'xi': xi_0, 'xi_decay': xi_decay, 'f_best': f_best, 'AF_name': AF_name}