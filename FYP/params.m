param.analysis_cache = 1;
param.batch_mode = 0;
param.seed = 100;
param.preserve_cache = 1;
param.extra_outdirs  = {};

% ===  parallel evaluations  ===
param.max_simul_evals = 12;

% ===  evolutionary algorithm  ===
param.pop_size    = 96;
param.generations = 50;
param.crossover_prob    = 1.0;
param.crossover_sbx_eta = 10;
param.mutation_prob     = 0.1;
param.mutation_poly_eta = 20;
param.max_fn_evals = inf;

% ===  surrogate models  ===
param.train_period = 5;
param.max_trueeval = 100000;
param.min_surr_gen        = 5;                  % new !!
param.min_surr_feasnum    = 10;                 % new !!
param.surr_num_clusters   = 1;
param.surr_type = {'orsm', 'rsm', 'dace', 'dace2', 'orbf', 'rbf'};
param.surr_add_crit       = 1.e-3;
param.surr_mse_threshold  = {0.05};             % new !!
param.surr_pred_dist      = 0.05;
param.surr_feas_only      = 1;                  % new !!
param.surr_train_ratio    = 0.9;
param.surr_max_traincount = inf;

% ===  infeasibility-driven EA  ===
param.max_inf_ratio = 0.0;          % (ng>0 only)

% ===  hybrid (SA/PS/SQP) search  ===
param.max_hybr_ratio = 0.25;
param.hybrid_selection = 2;         % 1: ND only 2: poor inds
% simulated annealing (ng=0 only)
param.sa_flag     = 0;              % 0: no SA   1: true eval  2: surr only
param.sa_genlims  = [10 inf];       % apply SA for gmin <= gen <= gmax
param.sa_options  = {'MaxFunEvals',100,'MaxIter',100,'ReannealInterval',25,'Display','off'};
% pattern search
param.ps_flag     = 0;              % 0: no PS   1: true eval  2: surr only
param.ps_genlims  = [10 inf];       % apply PS for gmin <= gen <= gmax
param.ps_options  = {'MaxFunEvals',100,'MaxIter',100,'Display','off'};
% sequential quadratic programming
param.sqp_flag    = 0;              % 0: no SQP  1: true eval  2: surr only
param.sqp_genlims = [10 inf];       % apply SQP for gmin <= gen <= gmax
param.sqp_options = {'Display','off'};
% (!) flag = 1 (true eval) is not recommended, considering comp time & surr training (check add_point.m)

% ===  sensitivity analysis  ===
param.sens_analysis_gen = 100;       % must be > train period  (0: no analysis)
param.sens_analysis_N   = 10000;    % number of sample points (0: = eval sols) 

% ===  image display & output  ===
param.anim_speed = 0;               % anim speed 1: fast < 20 slow (0: none)
param.save_image = 2;               % image save 1: simple 2: with gen
param.image_type = {'fig'};         % image format: eps/fig/png
param.f_labels = {'T [mN]','n_{t}[%]','Isp [s]','P_{a} [W]'}; 
param.x_labels = {'\Phi_{a} [V]','I_{a} [A]','\deltam [sccm]','IMR','OMD','ISR','OSR','OER'};
param.f_ranges = {[]};


% param.pop_size = 12;
% param.generations = 20;
% 
% param.crossover_prob = 1.0;
% param.crossover_sbx_eta = 10;
% param.mutation_prob = 0.1;
% param.mutation_poly_eta = 20;
% 
% param.seed = 10;
% param.batch_mode = 0;
% param.max_fn_evals = 10000;
% param.analysis_cache = 1;
% 
% param.train_period = 5;
% %param.retain_count = 4;
% param.retain_count = param.pop_size;
% 
% param.surr_num_clusters = 1;
% param.surr_type = {'orsm', 'rsm', 'dace', 'orbf', 'rbf'};
% %param.surr_type = {'dace2'};
% param.surr_max_traincount = 400;
% param.surr_train_ratio = 0.9;
% param.surr_add_crit = 1.e-3;
% param.surr_mse_threshold = {0.05};
% param.surr_pred_dist = 0.05;
% 
% % for parallel computing optimisation
% param.max_simul_evals = 1;
% 
% % save optimisation plots
% %   1 for saving normally
% %   2 for saving including gen in name
% param.save_image = 1;
% param.image_type = {'fig'}; % options: eps/fig/png
% param.f_labels = {'f'};
% param.x_labels = {'x_1','x_2'};
% param.f_ranges = {[]};
% 
% % for sensitivity analysis 
% % gen must be >= train_period (0 if unnecessary)
% %   N is number of sampling   (0 to use # of evaluated solutions)
% param.sens_analysis_gen   = 100;
% param.sens_analysis_N     = 10000;
