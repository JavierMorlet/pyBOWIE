# *******************************************************
# ****** Import libraries ******
# *******************************************************

import numpy as np
from ..utils.Preprocessing_data import Train_reducer, Train_inverter, Reduce, Inverse

# *******************************************************
# ****** Up_search_space_params ******
# *******************************************************

def Up_search_space_params(x, x_tau, connected_elements, n_elements, d_nc, dims, problem_type, n_components, reducer, inverter, inverter_transform):

    if reducer == "no":
        x_tau_ = x_tau
        connected_elements_ = connected_elements
        reducer_trained = None
        inverter_ = None
    else:
        x_tau_, reducer_trained = Train_reducer(x, d_nc, dims, problem_type, n_components, reducer) 
        if inverter_transform == "no":
            inverter_ = Train_inverter(x, x_tau, dims, inverter)
        elif inverter_transform == "yes":
            inverter_ = reducer_trained
        else:
            pass
        connected_elements_inv = [Inverse(connected_elements[i], inverter_transform, inverter) for i in range(n_elements)]
        connected_elements_ = [Reduce(connected_elements_inv[i], d_nc, dims, problem_type, reducer_trained) for i in range(n_elements)]

    return x_tau_, connected_elements_, reducer_trained, inverter_

# *******************************************************
# ****** Up_mesh ******
# *******************************************************

def Up_mesh(x_mesh, x_l, x_u, connected_elements, n_elements, n_p_mesh, dims):

    def New_boundaries(mesh, n_mesh, dims):

        x_l_new, x_u_new = [], []

        for i in range(n_mesh):
            x_l_new.append([np.min(mesh[i][:,j]) for j in range(dims)])
            x_u_new.append([np.max(mesh[i][:,j]) for j in range(dims)])
        
        return x_l_new, x_u_new
    
    def Detect_equal_limits(x_l, x_u, x_l_new, x_u_new, n_elements, n_p_mesh, dims):

        for i in range(n_elements):
            for j in range(dims):
                if x_l_new[i][j] == x_u_new[i][j]:
                    diff = [(x_u[j] - x_l[j])/(n_p_mesh-1)]
                    x_l_new_temp, x_u_new_temp = np.array([(x_l_new[i][j] - diff)[0], (x_l_new[i][j] + diff)[0]])
                    if x_l_new_temp < x_l[j]:
                        x_l_new[i][j] = x_l[j]
                    else:
                        x_l_new[i][j] = x_l_new_temp
                    if x_u_new_temp > x_u[j]:
                        x_u_new[i][j] = x_l[j]
                    else:
                        x_u_new[i][j] = x_u_new_temp

        return x_l_new, x_u_new
    
    def Find_intersection(x_mesh, x_l_new, x_u_new, n_p_mesh, n_elements, dims):

        def Compute_irr_grids(x_l_new, x_u_new, pts, dims):

            lst = []
            for i in range(dims):
                lst.append(np.linspace(x_l_new[i], x_u_new[i], int(np.round(pts[i]+1))))
            x_mesh_grid = np.meshgrid(*lst)
            
            return np.array(x_mesh_grid).T.reshape(-1, dims)

        def Intersect(A, B):

            _, ncols = A.shape
            dtype={'names':['f{}'.format(i) for i in range(ncols)],
                'formats':ncols * [A.dtype]}
            
            return np.intersect1d(A.view(dtype), B.view(dtype))

        points_mesh_new = [(np.array(x_l_new[i]) - np.array(x_u_new[i]))/(x_mesh[0][0] - x_mesh[0][n_p_mesh+1]) for i in range(n_elements)]
        points_mesh_new = np.abs(points_mesh_new)
        A = Compute_irr_grids(x_l_new[0], x_u_new[0], points_mesh_new[0], dims)
        x_mesh_new_ = []
        x_mesh_new_.append(A)

        for i in range(1, n_elements):
            B = Compute_irr_grids(x_l_new[i], x_u_new[i], points_mesh_new[i], dims)
            intersect_ = Intersect(A, B)
            if not intersect_.tolist():
                x_mesh_new_.append(B)
            else:
                x_mesh_new_[0] = np.vstack((np.concatenate(*[x_mesh_new_]), B))

        if type(x_mesh_new_) is not list:
            x_mesh_new_ = [x_mesh_new_]
        
        n_mesh_new = len(x_mesh_new_)

        return x_mesh_new_, n_mesh_new
        
    def Points_mesh(x_l, x_u, points_mesh, n_mesh, dims):

        l = [[(x_u[i][j] - x_l[i][j]) for i in range(n_mesh)] for j in range(dims)]
        a = [np.prod(l[0][i]) for i in range(n_mesh)]
        pts = a/np.max(a)
        points_mesh_min = int(np.sqrt(points_mesh))

        return [points_mesh_min if int(x*points_mesh) < points_mesh_min else int(x*points_mesh) for x in pts]

    x_l_new, x_u_new = New_boundaries(connected_elements, n_elements, dims)
    x_l_new, x_u_new = Detect_equal_limits(x_l, x_u, x_l_new, x_u_new, n_elements, n_p_mesh, dims)
    pts_mesh = int(np.round(x_mesh[0].shape[0]**(1/dims)))
    n_mesh = int(x_mesh[0].shape[0]/(pts_mesh-1))
    connected_elements_new, n_elements_new = Find_intersection(x_mesh, x_l_new, x_u_new, n_mesh, n_elements, dims)
    x_l_new, x_u_new = New_boundaries(connected_elements_new, n_elements_new, dims)
    points_mesh_new = Points_mesh(x_l_new, x_u_new, n_p_mesh, n_elements_new, dims)
    x_mesh_new = []

    for i in range(n_elements_new):
        lists = [np.linspace(x_l_new[i][j], x_u_new[i][j], points_mesh_new[i]) for j in range(dims)]
        x_mesh_grid = np.meshgrid(*lists)
        x_mesh_new.append(np.array(x_mesh_grid).T.reshape(-1, dims))

    return x_mesh_new, points_mesh_new