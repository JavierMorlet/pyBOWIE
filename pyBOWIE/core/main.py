"pyBOWIE (python Bayesian Optimization WIth supErlevel set filtration)"

import numpy as np
import warnings
from ..utils.Initialize import Space, Problem_type, Get_x_and_z, Get_constraints, Get_search_space_params, Get_x_mesh, Bounds, Get_n_jobs, Get_kernel, Iter_params, AF_params
from ..utils.Aux import Errors, Best_values, Check_if_improvement, Eval_fun, Eval_const, Imputation, Regret, Print_header, Print_results, Create_results
from ..utils.Preprocessing_data import Red_bounds, X_new_scaling
from ..utils.Models import Train_model, Train_models_constraints
from ..utils.Acq_fun import AF
from ..utils.Superlevel_set_filtration import Sl_sf
from ..utils.Querry_points import Find_constrains, Find_descriptors, Querry
from ..utils.Update import Up_search_space_params, Up_mesh

class BO():

    def __init__(self, function, domain, sense,
                 surrogate = "GP",
                 engine = 'gpflow',
                 acquisition_function = "UCB",
                 xi_0 = 2,
                 xi_f = 0.1,
                 xi_decay = "yes",
                 kernel = None,
                 kern_discovery = "yes",
                 kern_discovery_evals = 2,
                 x_0 = None, 
                 f_0 = None,
                 design = "LHS",
                 n_p_design = None,
                 n_jobs = None,
                 n_restarts = 5,
                 max_iter = 100, 
                 constraints = None,
                 constraints_method = "PoF",
                 reducer = None,
                 n_components = None,
                 inverter_transform = "no",
                 alpha = 0.95,
                 beta = 0.5,
                 c1_param = 50,
                 c2_param = 10,
                 verbose = 0,
                 ):

        self.function = function
        self.domain = domain
        self.sense = sense
        self.surrogate = surrogate
        self.engine = engine
        self.acquisition_function = acquisition_function
        self.xi_0 = xi_0
        self.xi_f = xi_f
        self.xi_decay = xi_decay
        self.kernel = kernel
        self.kern_discovery = kern_discovery
        self.kern_discovery_evals = kern_discovery_evals
        self.x_0 = x_0
        self.f_0 = f_0
        self.design = design
        self.n_p_design = n_p_design
        self.n_jobs = n_jobs
        self.n_restarts = n_restarts
        self.max_iter = max_iter
        self.constraints = constraints
        self.constraints_method = constraints_method
        self.reducer = reducer
        self.n_components = n_components
        self.inverter_transform = inverter_transform
        self.alpha = alpha
        self.beta = beta
        self.c1_param = c1_param
        self.c2_param = c2_param
        self.verbose = verbose

    def optimize(self):

        # *************** Main program ********************
        # Ignore warnings
        warnings.filterwarnings("ignore")
        #
        dims, d_nc, d_ni, d_nq, d_nd, x_l, x_u, int_val, cat_val, names = Space(self.domain)
        #
        Errors(d_nc, d_nd, self.design, self.surrogate, self.engine, self.constraints_method, self.acquisition_function)
        #
        problem_type = Problem_type(d_nc, d_ni, d_nq)
        # Get initial points of evaluation
        x, f, n_p_design_ = Get_x_and_z(self.function, self.x_0, self.f_0, dims, d_nc, d_ni, d_nq, x_l, x_u, int_val, cat_val, self.c1_param, problem_type, self.design, self.n_p_design)
        # Get constraints
        constraints_, n_constraints_ = Get_constraints(self.constraints, self.constraints_method)
        # Evaluate constraints
        g, y = Eval_const(x, constraints_, n_constraints_, self.constraints_method)
        g = Imputation(g, constraints_, self.constraints_method)
        # Perform data preprocessing, and dimensionality reduction and reduce if required
        x_tau, reducer_name, reducer_trained, inverter, dims_tau = Get_search_space_params(x, dims, d_nc, d_ni, d_nq, d_nd, x_l, x_u, int_val, cat_val, n_p_design_, problem_type, self.reducer, self.inverter_transform)
        # Reduce bounds
        x_l_tau, x_u_tau = Red_bounds(x_l, x_u, int_val, cat_val, d_nc, d_ni, d_nq, dims, dims_tau, problem_type, reducer_trained)
        # Get mesh
        x_mesh, n_p_mesh, n_p_mesh_0 = Get_x_mesh(x_l_tau, x_u_tau, dims_tau, self.c2_param)
        # Tuple of the bounds
        bounds = Bounds(x_l_tau, x_u_tau, dims_tau)
        # Spawning processes
        n_jobs_ = Get_n_jobs(self.n_jobs)
        # Select covariance
        kernel_ = Get_kernel(x_tau, f, dims_tau, self.surrogate, self.beta, self.kernel, self.kern_discovery, self.kern_discovery_evals, self.engine)
        # Initialize iteration parameters
        x_symb, x_symb_names, flag, q, q_inc, delta_q, chi, update_param = Iter_params(dims, dims_tau, self.max_iter, self.alpha)
        # Initialize AF parameters
        af_params = AF_params(f, self.xi_0, self.xi_f, self.xi_decay, self.max_iter, self.acquisition_function, self.sense)
        # Get best values
        x_best, f_best = Best_values(x, f, y, self.sense)
        af_params["f_best"] = f_best
        rt = []
        # Print results if vervose is active
        if self.verbose == 1:
            print(Print_header(names, x_symb_names, dims))
            x_print, f_print = Print_results(x_best, f_best, cat_val)
            print(0, ' ', f_print, *x_print)
        # *****
        # Cycle
        # *****
        for ite in range(1, self.max_iter+1):
            # Train surrogate model
            model = Train_model(x_tau, f, kernel_, self.surrogate, self.n_restarts, self.engine)
            # Train surrogate model for constraints
            models_constraints = Train_models_constraints(x_tau, g, constraints_, n_constraints_, self.constraints_method, self.engine)
            # Find conected elements
            connected_elements, n_connected_elements = Sl_sf(x_mesh, n_p_mesh, dims_tau, n_jobs_, q, af_params, self.constraints_method, model, models_constraints, self.engine, self.sense, AF)
            # Find CI bounds
            try:
                conc = np.concatenate(connected_elements)
            except:
                conc = np.hstack(connected_elements)
            try:
                if n_connected_elements == 0 or len(conc) < n_jobs_:
                    mu, Sigma_inv, t_critical = Find_descriptors(x_mesh, len(x_mesh), n_jobs_, dims_tau, self.alpha, af_params, self.constraints_method, model, models_constraints, self.engine, AF)
                else:
                    mu, Sigma_inv, t_critical = Find_descriptors(connected_elements, n_connected_elements, n_jobs_, dims_tau, self.alpha, af_params, self.constraints_method, model, models_constraints, self.engine, AF)
            except:
                print(n_connected_elements, conc)
            # Lambdify CI bounds
            CI_lambda = Find_constrains(x_symb, mu, Sigma_inv, chi, t_critical, n_jobs_, dims_tau)
            # Find new point(s)
            x_new_tau = Querry(n_jobs_, mu, CI_lambda, bounds, dims_tau, af_params, self.constraints_method, self.sense, model, models_constraints, self.engine, AF)
            # Scale new point(s)
            x_new = X_new_scaling(x_new_tau, d_nc, d_ni, d_nq, x_l, x_u, int_val, cat_val, n_jobs_, reducer_name, self.inverter_transform, inverter, problem_type)
            # Evaluate objective function
            f_new = Eval_fun(x_new, n_jobs_, self.function)
            # Evaluate constraints
            g_new, y_new = Eval_const(x_new, constraints_, n_constraints_, self.constraints_method)
            # Update train data
            x, x_tau, f = np.vstack((x, x_new)), np.vstack((x_tau, x_new_tau)), np.vstack((f, f_new))
            if g is None:
                pass
            else:
                g = np.vstack((g, g_new))
                y = np.hstack((y, y_new))
                g = Imputation(g, constraints_, self.constraints_method)
            # Update reducer, inverter, inducing points
            if (ite % update_param) == 0:
                x_tau, connected_elements, reducer_trained, inverter = Up_search_space_params(x, x_tau, connected_elements, n_connected_elements, d_nc, dims, problem_type, dims_tau, reducer_name, inverter, self.inverter_transform)
            # Update x_mesh if the new point improves objetive function
            flag = Check_if_improvement(f_new, f_best, n_jobs_, self.sense)
            if flag == 1:
                try:
                    x_mesh, n_p_mesh = Up_mesh(x_mesh, x_l_tau, x_u_tau, connected_elements, n_connected_elements, n_p_mesh_0, dims_tau)
                except:
                    pass
            # Compute regret
            rt.append(Regret(f_new, np.array(x_new_tau), n_jobs_, model, self.engine))
            # Update iteration parameters
            q_inc += delta_q
            q = int(round(q_inc/5.0)*5.0)
            # Get best values
            x_best, f_best = Best_values(x, f, y, self.sense)
            # Update parameters of AF
            af_params["xi"] *= af_params["xi_decay"]
            af_params["f_best"] = f_best
            # Print results if verbose
            if self.verbose == 1:
                x_print, f_print = Print_results(x_best, f_best, cat_val)
                print(ite, ' ', f_print, *x_print)

        res = Create_results(x_best, f_best, x, f, x_l, x_u, dims, self.max_iter, n_p_design_, self.design, af_params, self.constraints_method, rt, models_constraints, model)

        return res