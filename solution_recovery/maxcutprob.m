%% Max-cut with CS regularizers
% minimize -1/2 sum W_ij*(1 - s_i*s_j) + \ell(s) subject to |s_i|<=1
%%
addpath './solver'
clear
% close all
rng(30)

%%
graph_source = 'G10';% ['medium_toy', 'my_toy', 'G10', 'G11', 'G12', 'G13']
comment = ['maxcut_',graph_source];

num_trials = 10;
max_restart_count = 1;
lambda = 1e-7; % for Gx graphs
% lambda = 1; % for small (toy) graphs
plot_hist = 0;
thr_rec = 1e-2;

g = @(x) 0; % if A*x = b, otherwise inf but that will never be used since iterates are the output of projection
eye_fun = @(x) x;

std_abs = @(x) std(abs(x), 1);
criterion_rec = @(x, xstar) norm(x-xstar)/norm(xstar) < thr_rec; % criterion to recover xstar

opts.maxIters = 10000;
opts.backtrack = 1;
opts.accelerate = 0;
opts.adaptive = 1;
opts.tol = 1e-8;  % Use super strict tolerance
opts.recordObjective = true; %  Record the objective function so we can plot it
opts.verbose=false;
opts.verbose=true;
opts.stringHeader='    ';      % Append a tab to all text output from FISTA.  This option makes formatting look a bit nicer.
% opts.stopRule = 'residual';


switch graph_source 
    case 'medium_toy'  % Medium forum example
        N = 5; E = 7; 
        is = [1,1,2,2,2,3,4];
        js = [2,5,3,4,5,4,5];
        ws = [1,1,-1, 1,-1,-1,1];
        s_star = [-1;1;-1;-1;1];
        if length(is) ~= E || length(js) ~= E || length(s_star) ~= N
            error('is and js length must match E, s_stars length must match N') 
        end
        G = create_graph_matrix(N, is, js, ws);
        target = -3;
    
    case 'my_toy' % My example
        N = 4; E = 5; % my example
        is = [1,1,1,2,3];
        js = [2,3,4,4,4];
        ws = [10, 20, 30, 40, 50];
        s_star = [-1;1;1;-1];
        if length(is) ~= E || length(js) ~= E || length(s_star) ~= N
            error('is and js length must match E, s_stars length must match N') 
        end
        G = create_graph_matrix(N, is, js, ws);
        target = -120;
   
    otherwise
        [N,E,G,target] = read_txt_into_graph(['graphs/',graph_source,'.txt']);
    
% % https://oxfordcontrol.github.io/COSMO.jl/stable/examples/maxcut/ example
% N = 4; E = 5;
% s_star = [1;1;1;-1];
% G = zeros(N);
% G(1,2) = 1; G(2,1) = 4;
% G(1, 4) = 8; G(4, 1) = 8;
% G(2,3) = 2; G(3,2) = 2;
% G(2,4) = 10; G(4,2) = 10;
% G(3,4) = 6; G(4,3) = 6;
end


% Maxcut objective
Lmc = @(W,s) -1/2 * sum(sum(triu(W .* (1 - s*s.'))));
gradLmc = @(W,s) 1/2 * W*s ;

% Lbin minimization objective
Lbin = @(x) N * sum(x.^4) - sum(x.^2)^2; % Lbin(x)
gradLbin = @(x) 4*N * x.^3 - 4 * sum(x.^2) .* x; 

%%

start_count = 0;
t0 = tic;

f = @(x) Lmc(G,x) + lambda * Lbin(x);
gradf = @(x) gradLmc(G,x) + lambda * gradLbin(x);

proxg = @(z,t)  z ./ max(abs(z),1); % sign(z); %

p_recovery_list = zeros(1,num_trials); % recovery xstar success list
iter_count_list = zeros(1,num_trials);
start_count_list = zeros(1,num_trials);
Lc_init_list = zeros(1,num_trials);
Lc_fin_list = zeros(1,num_trials);

for t=1:num_trials
    
    boo = true;
    start_count = 0;
    while start_count < max_restart_count && boo
        s0 = sign(randn(N,1)); % initial
        x0 = s0; 
        disp(['Start ', num2str(start_count)])

        if contains(graph_source, 'toy')
            Lmc_star = Lmc(G,s_star)
            gradLMc = gradLmc(G,s_star)
            Lmc0 = Lmc(G,s0)
            gradLmc0 = gradLmc(G,s0)
        end
        opts.function = @(x) Lmc(G,(x));
        criterion_rec_cur = @(x) (Lmc(G,(x)) <= target) && norm(x-sign(x))/norm(x) < thr_rec;
        opts.stopNow = @(x1,iter,resid,normResid,maxResidual,opts) criterion_rec_cur(x1);
            
        [xhat, outs] = fasta(eye_fun, eye_fun, f, gradf, g, proxg, x0, opts);

        Lmc_hat = Lmc(G,xhat)
        
        Lc_fin_list(t) = Lmc_hat;
        Lc_init_list(t) = outs.funcValues(1);

        boo = ~criterion_rec_cur(xhat);
        start_count = start_count + 1;
    end

    iter_count = outs.iterationCount;
    deltat = toc(t0);
    disp(['FASTA finished ', num2str(deltat)])
    if plot_hist %&& criterion_rec_cur(xhat) && ~criterion_rec_cur(xhat_inf)
        figure;
        histogram(xhat, 100)
        xlim([-1.2 1.2])
        xlabel('x_n'); ylabel('count')
        
        xlabel('x_n'); ylabel('count')
%         figure; plot(outs.objective, '-o')
        figure; plot(outs.funcValues, '-o')
        flag = 1;
    end
    p_recovery_list(t) = criterion_rec_cur(xhat);
    iter_count_list(t) = iter_count;
    start_count_list(t) = start_count;
end

Lc0_avg = mean(Lc_init_list)
Lc0_std = std(Lc_init_list)

Lmc_avg = mean(Lc_fin_list)
Lmc_std = std(Lc_fin_list)

runID = [comment,'-lambda',num2str(lambda),'start','-N',num2str(N),'_','acc',num2str(opts.accelerate),'_adaptive',num2str(opts.adaptive), '_bt', num2str(opts.backtrack),'_trial', num2str(num_trials),'_fastamaxiters', num2str(opts.maxIters)];
% if save_result
save(['results/',runID,'.mat']) %, 'p_success_list', 'p_recovery_list', 'iter_count_list', 'start_counts')


%% Function to read the txt in
function G = create_graph_matrix(N, rows, cols, values)
% Initialize the matrix with zeros
G = zeros(N);

% Fill in the upper triangular part of the matrix
for i = 1:length(rows)
    r = rows(i);
    c = cols(i);
    G(r, c) = values(i);   % Set the value at (r, c)
end

% Make the matrix symmetric 
G = G + G.' - diag(diag(G));

% % Display the matrix or perform further operations
% disp(matrix);

end
%%
function [N,E,G, target] = read_txt_into_graph(filename)
% Load data from the text file
data = readmatrix(filename);  % Replace 'yourfile.txt' with the actual filename

% Extract rows, columns, and values from the data
rows = data(2:end, 1);       % First column: row indices
cols = data(2:end, 2);       % Second column: column indices
values = data(2:end, 3);     % Third column: matrix values

% Create the n × n matrix 
N = data(1,1);
E = data(1,2);
G = create_graph_matrix(N,rows,cols,values);
target = - data(1,3);

end