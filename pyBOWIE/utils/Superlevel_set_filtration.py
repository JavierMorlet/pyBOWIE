# *******************************************************
# ****** Import libraries ******
# *******************************************************

import numpy as np
from itertools import product, permutations
from operator import itemgetter
from ..utils.Aux import Flatten

# *******************************************************
# ****** Sl_sf ******
# *******************************************************

def Sl_sf(mesh, p_mesh, dims, jobs, q0, af_params, constraints_method, model, models_const, engine, sense, Acq_fun):

    """ 
    Find connected component of the matrix of points x in the domain adove the level set
    """

    # *************************
    # Filtration
    # *************************

    def Filtration(x, score, p_x, dims, level, sense):
        if sense == "maximize":
            data_matrix = np.array([score > level])[0]
        elif sense == "minimize":
            data_matrix = np.array([score < level])[0]
        # Reshape the data
        data_matrix = data_matrix.reshape(*[p_x]*dims)

        return data_matrix.astype(int), x.reshape(*[p_x]*dims, dims)
    
    # *************************
    # Find_connected_elements_dfs
    # *************************

    def Find_connected_elements_dfs(nodes, matrix):
        # Number of dimensions
        D = [nodes.shape[i] for i in range(dims)]
        D0 = D[0]
        # Define the neighbors
        s_neg = [-1]
        s_pos = [1]
        for i in range(dims-1):
            s_neg.append(0)
            s_pos.append(0)
        neigh = tuple(set(permutations(s_neg, dims)) | set(permutations(s_pos, dims)))
        
        # dfs_connected
        def dfs_connected(element, value):
            #
            elem_tuple = tuple(element)
            # Check if the current cell is outside the matrix or has a different value
            if [False for i in range(len(element)) if element[i] < 0] or [False for i in range(len(element)) if element[i] >= D[i]] or nodes[elem_tuple] != value:
                return []
            
            # Mark the cell as visited
            nodes[elem_tuple] = -1
            # Initialize the connected component with the current cell
            nodes_conected = [elem_tuple]
            # Check the neighbors
            for dr in neigh:
                nodes_conected.extend(dfs_connected([element[i] + dr[i] for i in range(len(element))], value))
            
            return nodes_conected
        
        # main program
        list_connected_components = []
        d = product(range(D0), repeat=dims)
        # cycle
        for i in d:
            if nodes[i] == 1:
                nodes_conected = dfs_connected(i, nodes[i])
                list_connected_components.append(nodes_conected)
        
        connected_elements = [np.array([matrix[tuple(list_connected_components[k][i])] for i in range(len(list_connected_components[k]))]) for k in range(len(list_connected_components))]
        
        return connected_elements
    
    # *************************
    # List_of_elements
    # *************************

    def List_of_elements(x, score, p_x, n_elements, dims, q, sense):
        #
        connected_elements = []
        level = np.percentile(np.concatenate(score), q)
        if level == 0:
            level = np.mean(np.concatenate(score))
        #
        for j in range(n_elements):
            x_flitred = Filtration(x[j], score[j], p_x[j], dims, level, sense)
            connected_elements.append(Find_connected_elements_dfs(x_flitred[0], x_flitred[1])) 

        connected_elements = [x for x in connected_elements if x != []]
        connected_elements = Flatten(connected_elements)
        n_connected_elements = len(connected_elements)

        return connected_elements, n_connected_elements
    
    # *************************
    # Reduce_num_elements
    # *************************
    
    def Reduce_num_elements(x, mesh, score, n_elements, jobs, sense):

        def Replace_val(value, arr_1, arr_2):

            return arr_2[(arr_1 == value).all(axis=1)].reshape(-1)

        score_all = np.concatenate(score)
        mesh_all = np.concatenate(mesh)
        scores_lst = []
        scores_mean = []

        for i in range(n_elements):
            try:
                scores_lst.append(np.array([Replace_val(x[i][j], mesh_all, score_all) for j in range(len(x[i]))]))
            except:
                return [Replace_val(x[i][j], mesh_all, score_all) for j in range(len(x[i]))]

        for i in range(n_elements):
            if len(scores_lst[i].shape) == 1:
                scores_mean.append(np.mean([np.mean(scores_lst[i][j]) for j in range(len(scores_lst[i]))]))
            else:
                scores_mean.append(np.mean(scores_lst[i]))
                
        if sense == "maximize":
            sorted = np.sort(scores_mean)[::-1][:jobs]
        elif sense == "minimize":
            sorted = np.sort(scores_mean)[:jobs]
        l = np.array([np.where(scores_mean == sorted[i]) for i in range(len(sorted))]).reshape(-1)
        #print(l)
        try:
            l = Flatten(l)
            l = list(set(l))
        except:
            l = l
        connected_elements = [x[i] for i in l]
        n_connected_elements = len(connected_elements)

        return connected_elements, n_connected_elements
    
    # *************************
    #  Main
    # *************************

    score = [Acq_fun(mesh[i], af_params, constraints_method, model, models_const, engine) for i in range(len(mesh))]
    n_elements = len(mesh)
    # 
    if sense == "maximize":
        connected_elements, n_connected_elements = List_of_elements(mesh, score, p_mesh, n_elements, dims, q0, sense)
    elif sense == "minimize":
        connected_elements, n_connected_elements = List_of_elements(mesh, score, p_mesh, n_elements, dims, 100-q0, sense)
    else:
        pass
    #  
    if n_connected_elements > jobs:
        try:
            connected_elements, n_connected_elements = Reduce_num_elements(connected_elements, mesh, score, n_connected_elements, jobs, sense)
        except:
            b = np.random.choice(n_connected_elements, jobs, replace=False)
            connected_elements = itemgetter(*b)(connected_elements)
            n_connected_elements = jobs
        

    return connected_elements, n_connected_elements