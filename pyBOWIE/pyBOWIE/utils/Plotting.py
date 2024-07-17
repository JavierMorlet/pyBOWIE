import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from ..utils.Acq_fun import AF

# *******************************************************

def Data_plot(x_l, x_u, dims, n_plot):

    lists = [np.linspace(x_l[i], x_u[i], n_plot) for i in range(dims)]
    # Create a meshgrid data
    x_plot = np.meshgrid(*lists)
    x_plot = np.array(x_plot).T.reshape(-1, dims)
    
    return x_plot

# *******************************************************

def Plot_surrogate(args):

    _, _, x_init, f_init, x_iters, f_iters, x_l, x_u, dims, _, _, initial_points, _, _, _, _, _, model = args.__dict__.values()
    # x_best, f_best, x_init, f_init, x_iters, f_iters, x_l, x_u, dims, iters, initial_design, initial_points, xi, acquisition_function, regret, constraint_method, models_constraints, model

    if dims == 1:
        
        #af_params = {'xi': xi, 'xi_decay': None, 'f_best': f_best, 'AF_name': acquisition_function}
        fig = plt.figure()
        n_plot = 1000
        x_plot = Data_plot(x_l, x_u, dims, n_plot)
        f_mean, f_std = model.predict(x_plot)

        plt.plot(x_plot, f_mean)
        plt.fill_between(x_plot.reshape(-1), (f_mean - f_std).reshape(-1), (f_mean + f_std).reshape(-1), alpha=0.2)
        plt.scatter(x_init, f_init, c='k', s=10, label='Training data')
        plt.scatter(x_iters, f_iters, c='r', s=10, label='Observations')
        plt.xlabel('$x$')
        plt.ylabel('$f(x)$')
        plt.title('Surrogate model')
        plt.legend(bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0.)
        
    elif dims == 2:

        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        n_plot = 100
        x_plot = Data_plot(x_l, x_u, dims, n_plot)
        f_mean, f_std = model.predict(x_plot)
        x1_plot, x2_plot = x_plot[:,0].reshape(n_plot, n_plot), x_plot[:,1].reshape(n_plot, n_plot)
        x1_init, x2_init = x_init[:,0], x_init[:,1]
        x1_iters, x2_iters = x_iters[:,0], x_iters[:,1]
        f_mean, f_std = f_mean.reshape(n_plot, n_plot), f_std.reshape(n_plot, n_plot)

        p1 = ax[0].contourf(x1_plot, x2_plot, f_mean, cmap='jet')
        ax[0].scatter(x1_init, x2_init, c='k', label='Training data')
        ax[0].scatter(x1_iters, x2_iters, c='r', label='Observations')
        ax[0].set_xlabel('$x_1$')
        ax[0].set_ylabel('$x_2$')
        ax[0].set_title('Surrogate model \n mean')

        p2 = ax[1].contourf(x1_plot, x2_plot, f_std, cmap='jet')
        ax[1].scatter(x1_init, x2_init, c='k', label='Training data')
        ax[1].scatter(x1_iters, x2_iters, c='r', label='Observations')
        ax[1].set_xlabel('$x_1$')
        ax[1].set_ylabel('$x_2$')
        ax[1].set_title('Surrogate model \n uncertanty')
        
        plt.colorbar(p1)
        plt.colorbar(p2)
        plt.legend(bbox_to_anchor=(1.3, 1), loc='upper left', borderaxespad=0.)

    elif dims > 2:

        print('not posible to plot')
        fig = None

    return fig

# *******************************************************

def Plot_AF(args):

    _, f_best, _, _, _, _, x_l, x_u, dims, _, _, _, xi, acquisition_function, _, constraints_method, models_const, model = args.__dict__.values()

    if dims == 1:
        
        af_params = {'xi': xi, 'xi_decay': None, 'f_best': f_best, 'AF_name': acquisition_function}
        fig = plt.figure()
        n_plot = 500
        x_plot = Data_plot(x_l, x_u, dims, n_plot)
        z_acq = AF(x_plot, af_params, constraints_method, model, models_const)

        plt.plot(x_plot, z_acq)
        plt.xlabel('$x$')
        plt.ylabel('$AF(x)$')
        plt.title('Acquisition function')
        
    elif dims == 2:

        af_params = {'xi': xi, 'xi_decay': None, 'f_best': f_best, 'AF_name': acquisition_function}
        fig = plt.figure()
        n_plot = 100
        x_plot = Data_plot(x_l, x_u, dims, n_plot)
        z_acq = AF(x_plot, af_params, constraints_method, model, models_const)
        x1_plot, x2_plot = x_plot[:,0].reshape(n_plot, n_plot), x_plot[:,1].reshape(n_plot, n_plot)
        z_acq = z_acq.reshape(n_plot, n_plot)        

        plt.contourf(x1_plot, x2_plot, z_acq, cmap='jet')
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.title('Acquisition function')

    elif dims > 2:

        print('not posible to plot')
        fig = None

    return fig

def Plot_regret(args):

    _, _, _, _, _, _, _, _, _, _, _, _, _, _, regret, _, _, _ = args.__dict__.values()
    R = np.cumsum(regret)

    fig = plt.figure()

    plt.plot(regret, label="instantaneous regret")
    plt.plot(R, label="cummulative regret")
    plt.xlabel('Iterations')
    plt.ylabel('R')
    plt.title('Regret')
    plt.legend(bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0.)

    return fig