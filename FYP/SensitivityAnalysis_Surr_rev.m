%
%   Perform Sensitivity Analysis to assess influence of design parameters
%   on objective functions (loading surrogate models from a file)
%

clear all;

%   set data information and other parameters (INPUT)
fnSAEA = 'CFTOpt';
path_Surr = pwd;
dfnSurr = [path_Surr '/CFTOpt-SAEA_rev-Sur.mat'];  % include all new values for hp&wp etc. without injection pressure
gen_end = 50;
addpath(path_Surr);

%   load surrogate models
load(dfnSurr);

%   perform sensitivity analysis
global state; 
state.gen_id = gen_end;
param.sens_analysis_N = 10000;
sensitivity_analysis(surr, fnSAEA, param.sens_analysis_N);

rmpath(path_Surr);