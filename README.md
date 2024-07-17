# pyBOWIE (python Bayesian Optimization WIth supErlevel set filtration)

A Python library for Constrained Mixed-Integer Bayesian Optimization with Superlevel Set Filtration

![Superlevel set filtration](docsrc/Slf.png)

The library implements parallelization by exploiting the system structure to partition the design space using level-set partitioning, automatic covariance selection, a framework for working with constraint functions in case the optimization problem requires it, as well as the ability to work with continuous and/or discrete variables. 

The algorithm can be summarized as follows: It begin with estimating the size of the initial training data based on the calculation time of the cost function and generating samples using a space-filling strategy to evaluate the cost and constraint functions. The samples are mapped to the feature space. Then, the covariance function is determined automatically and the parameters of the optimization algorithm are initialized to search for new points iteratively. This process first trains a surrogate model for the cost function and surrogate models for the constraint function. The acquisition function is then evaluated in a grid layout over the design space. The superlevel set filtration is carried out to identify regions of interest that are used as constraints of the acquisition function and to focus the search for new points on which the objective and constraint functions are evaluated. The parameters of the optimization algorithm are updated, and the cycle is repeated until the maximum number of iterations is reached.

For tutorial notebooks, check out the examples folder.

## Getting started

**Installing with pip**

````
pip install pyBOWIE
````

**Notebook tutorials**

Read the [basic notebook](https://github.com/JavierMorlet/pyBOWIE/blob/main/examples/Basic_tour.ipynb)
for an introduction to the basic concepts of the library.

For more details please refer to the [advaned tour notebook](https://github.com/JavierMorlet/pyBOWIE/blob/main/examples/Advanced_tour.ipynb)

## Dependencies:

* numpy
* scipy
* sympy
* itertools
* pandas
* sklearn
* GPy
* prince
* properscoring
* multiprocess
* matplotlib.pyplot

## Library parameters

The library parameters are:

### Required arguments

<ul>
  <li>function :</li>
  <ul>
    <li>Python function with one output</li>
    <li>Cost or objective function</li>
  </ul>
  <li>domain :</li>
  <ul>
    <li>List</li>
    <li>The kind and bounds of each of the variables. The valid kinds are {'continuous', 'integer', 'categorical'}</li>
  </ul>
  <li>sense :</li>
  <ul>
    <li>{'maximize', 'minimize'}</li>
    <li>Specifies whether to minimize or maximize the objective function</li>
  </ul>
<ul>

### Default Arguments

<ul>
  <li>surrogate :</li>
  <ul>
    <li>{'GP', 'SGP'}, default 'GP'</li>
    <li>Specifies the surrogate model of the bayesian optimization algorithm</li>
  </ul>
  <li>acquisition_function :</li>
  <ul>
    <li>{'UCB', 'PI', 'EI'}, default 'UCB'</li>
    <li>Specifies the acquisition function of the bayesian optimization algorithm</li>
  </ul>
  <li>xi_0 :</li>
  <ul>
    <li>float, default 2</li>
    <li>Initial value of the acquisition function hyperparameter</li>
  </ul>
  <li>xi_f :</li>
  <ul>
    <li>float, default 0.1</li>
    <li>Final value of the acquisition function hyperparameter</li>
  </ul>
  <li>xi_decay :</li>
  <ul>
    <li>{'yes', 'no'}, default 'yes'</li>
    <li>Specifies whether the hyperparameter of the acquisition function decays.</li>
  </ul>
  <li>kernel :</li>
  <ul>
    <li>GPy.kern, default None</li>
    <li>The kernel is a function. If specified must be from GPy package</li>
  </ul>
  <li>kern_discovery :</li>
  <ul>
    <li>{'yes', 'no'}, default 'yes'</li>
    <li>Specifies whether the kernel function is selected automatically. If 'no', then specify a kernel from GPy</li>
  </ul>
  <li>kern_discovery_evals :</li>
  <ul>
    <li>int, default 2</li>
    <li>Specifies the number of evaluations to find the covariance function, only of kern_discovery is 'yes'</li>
  </ul>
  <li>x_0 :</li>
  <ul>
    <li>numpy array, default None</li>
    <li>Specifies the initial points to evaluate the surrogate model.</li>
  </ul>
  <li>f_0 :</li>
  <ul>
    <li>numpy array, default None</li>
    <li>Specifies the values ​​of the objective function evaluated at x_0, only of x_0 is specified.</li>
  </ul>
  <li>design :</li>
  <ul>
    <li>{'LHS', 'Sobol', 'Halton', 'random'}, default 'LHS'</li>
    <li>Specifies the initial design strategy to initialize x_0 if it is not specified.</li>
  </ul>
  <li>n_p_design :</li>
  <ul>
    <li>int, default None</li>
    <li>Specifies the number of points of the initial design, i.e. the size of the initial points matrix. If none, the value of n_p_design is calculated depending on the computing time of a random evaluation of the objective function</li>
  </ul>
  <li>n_jobs :</li>
  <ul>
    <li>int, default None</li>
    <li>The number of jobs to run in parallel for evaluations of the objective function. None means not using any processors (1 evaluation), and -1 means using all processors.</li>
  </ul>
  <li>n_restarts :</li>
  <ul>
    <li>int, default 5</li>
    <li>The number of times the surrogate model optimizer is restarted.</li>
  </ul>
  <li>max_iter :</li>
  <ul>
    <li>int, default 100</li>
    <li>Strict limit on the maximum number of iterations within the algorithm.</li>
  </ul>
  <li>constraints :</li>
  <ul>
    <li>tuple, default None</li>
    <li>Specifies the constraints functions that define the feasible region.</li>
  </ul>
  <li>constraints_method :</li>
  <ul>
    <li>{'PoF', 'GPC'}, default 'PoF'</li>
    <li>Specifies the method to model the constraints functions.</li>
  </ul>
  <li>reducer :</li>
  <ul>
    <li>A dimension reduction module, default None</li>
    <li>Specifies a dimension reduction technique specified by the user. If none, certain rules are used to define the dimension reduction technique depending in the characteristics of the design variables.</li>
  </ul>
  <li>inverter_transform :</li>
  <ul>
    <li>{'yes', 'no'}, default 'no'</li>
    <li>Specifies wheter to use an analytic inverse of the dimension reduction technique. Only valid for PCA</li>
  </ul>
  <li>alpha :</li>
  <ul>
    <li>float, default 0.95</li>
    <li>Confidence level for the Chi-squares and T-scores. Must be between 0 and 1.</li>
  </ul>
  <li>c1_param :</li>
  <ul>
    <li>int, default 50</li>
    <li>Value used to calculate the number of points for the matrix of initial estimates.</li>
  </ul>
  <li>c2_param :</li>
  <ul>
    <li>int, default 10</li>
    <li>Value used to control the number of grid sampling points.</li>
  </ul>
  <li>verbose :</li>
  <ul>
    <li>{0, 1}, default 0</li>
    <li>Enable verbose output.</li>
  </ul>
</ul>

## Returns

optimize() returns a dict with:

<ul>
  <li>x_best : The values ​​of the variables that yield the best value of objective function</li>
  <li>f_best : The best value ​​of the objective function</li>
  <li>x_init : The initial sampling points to evaluate the objective function</li>
  <li>f_init : The value of the objective function at the initial sampling points</li>
  <li>x_iters : The new points suggested during the progress of the algorithm.</li>
  <li>f_iters : The value of the objective function at the new suggested points.</li>
  <li>x_l : Lower bounds of the variables.</li>
  <li>x_u : Upper bounds of the variables.</li>
  <li>dims : Dimensions of the optimization problem.</li>
  <li>iters : Maximum number of iterations within the algorithm.</li>
  <li>initial_design : Initial design strategy for the initial sampling points.</li>
  <li>initial_points : Number of points of the initial sampling points matrix.</li>
  <li>acquisition_function : Acquisition function employed during the algorithm.</li>
  <li>xi : Last value of the acquisition function hyperparameter.</li>
  <li>regret : Metric to asses the performance of the algorithm.</li>
  <li>constraint_method : Stategy used to handle the constraint functions.</li>
  <li>models_constraints : Surrogate model for the constraint functions.</li>
  <li>model : Surrogate model for the objective function.</li>
</ul>