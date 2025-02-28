% Orthogonal matrix recovery CS regularizer
% AX = B, where X has orthogonal columns, A is wide
% sueda
%%
addpath './solver'
clear
% close all

N = 100;
K = 100;
method_list = [1, 2];
method_names = {'$$\ell_{orth}$$', 'with plain gd'};
savepdf = 1;
method_list = [1];
method_names = {'FASTA'};
savepdf = 0; % only turn this on when at a final stage
comment = 'orth_recovery_gamma01to1by01';
max_restart_count = 1;
plot_hist = 0;  % only set to 1 in debug mode to look at each xhat, and run Linf first

epsilon = 1e-13;

num_trials = 1;
gamma_list = [0.1:0.1:0.2];
lambda = 1;
% thr_rec = 1e-2; % recovery
thr_orth = 1e-2;
thr_res = 1e-2; 
thr_rec = 1e-2;

simul.tau = 1e-8;
simul.max_iter = 1e3;

Lorth = @(X) N * norm(X'*X, 'fro')^2 - norm(X, 'fro')^4;
grad_Lorth = @(X) 4*N * (X*X'*X) - 4*norm(X, 'fro')^2 * X;

eye_fun = @(x) x;

how_far_from_orth = @(X) (norm(X' * X - eye(K), 'fro') / sqrt(K));
criterion_rec = @(x) how_far_from_orth(x) < thr_orth; % criterion to recover xstar

opts.maxIters = 1e3;
opts.backtrack = 1;
opts.accelerate = 1;
opts.adaptive = 0;
opts.tol = 1e-8;  % Use super strict tolerance
opts.recordObjective = true; %  Record the objective function so we can plot it
opts.verbose=false;
opts.verbose=true;
opts.stringHeader='    ';      % Append a tab to all text output from FISTA.  This option makes formatting look a bit nicer.
% opts.stopRule = 'residual';

M_list = round(N * gamma_list);
p_recovery_list = zeros(length(M_list), length(method_list), num_trials); % recovery xstar success list
p_recovery_list2 = zeros(length(M_list), length(method_list), num_trials); % recovery xstar success list
iter_count_list = zeros(length(M_list), length(method_list), num_trials);
start_count_list = zeros(length(M_list), length(method_list), num_trials);

for M_idx=1:length(M_list)
    M = M_list(M_idx);
    for t=1:num_trials
        g = @(x) 0; 
                
        % Create random orth Xstar
        tmp = randn(N);
        [U,~,~] = svd(tmp);
        Xstar = U(:,1:K);
        % Create random measurements
        A = randn(M, N);
        % Create B
        B = A * Xstar;
        
        G = (A*A');
        proxg = @(Z,t) Z - A'*(G\(A*Z-B)); % linear projection on to Ax=b
        
        [X0,~,~] = svd(randn(N));
        x0 = X0(:,1:K);

        % criterion_rec_cur = @(x) criterion_rec(A,B,x, thr_orth, thr_res); % current recovery criterion with xstar fixed
%         criterion_rec_cur = @(x) criterion_rec(x, Xstar); % current recovery criterion with xstar fixed
        criterion_rec_cur = @(x) criterion_rec(x); % current recovery criterion with xstar fixed
        howfarfromBstar_cur = @(x) how_far_from_Bstar(A,x,B);
        opts.stopNow = @(x1,iter,resid,normResid,maxResidual,opts) criterion_rec_cur(x1);
        
        for method_idx=1:length(method_list)
            method = method_list(method_idx);
            if method == 1
                f = @(z) Lorth(z);
                gradf = @(z) grad_Lorth(z);

                start_count = 0;
                t0 = tic;
                boo = true;
                while start_count < max_restart_count && boo
                    disp(['Start ', num2str(start_count)])
                    [xhat, outs] = fasta(eye_fun, eye_fun, f, gradf, g, proxg, x0, opts);
                    boo = ~criterion_rec_cur(xhat);
                    start_count = start_count + 1;
                end
                iter_count = outs.iterationCount;
                deltat = toc(t0);
                disp(['FASTA finished ', num2str(deltat)])

                if plot_hist
                    figure
                    plot(outs.objective)
                    flagg = 1;
                end
            
            elseif method == 2
                f = @(z) Lorth(z);
                gradf = @(z) grad_Lorth(z);
                
                start_count = 0;
                t0 = tic;
                boo = true;
                while start_count < max_restart_count && boo
                    disp(['Start ', num2str(start_count)])
                    [xhat, obj, crits] = gd(simul, f, gradf, x0, howfarfromBstar_cur);
                    boo = ~criterion_rec_cur(xhat);
                    start_count = start_count + 1;
                end
                iter_count = simul.max_iter;
                deltat = toc(t0);
                disp(['GD finished ', num2str(deltat)])

            end
            [U,S,V] = svd(xhat, "econ");
            xhat_orth_proj = U*V';

            p_recovery_list(M_idx,method_idx,t) = criterion_rec_cur(xhat);
            p_recovery_list2(M_idx,method_idx,t) = criterion_rec_cur(xhat_orth_proj);
            iter_count_list(M_idx,method_idx,t) = iter_count;
            start_count_list(M_idx,method_idx,t) = 1;
        end
        disp(['Trial ',num2str(t), ' for M = ', num2str(M)])
    end
    disp(['Finished for M ', num2str(M)])
end

p_recovery = squeeze(mean(p_recovery_list, 3));
p_recovery2 = squeeze(mean(p_recovery_list2, 3));
iter_count = squeeze(mean(iter_count_list, 3));

runID = [comment,'_Lorth_',num2str(max_restart_count),'start','-N',num2str(N),'-K',num2str(K),'_','acc',num2str(opts.accelerate),'_adaptive',num2str(opts.adaptive), '_bt', num2str(opts.backtrack),'_trial', num2str(num_trials),'_fastamaxiters', num2str(opts.maxIters)];
% if save_result
save(['results/',runID,'.mat']) %, 'p_success_list', 'p_recovery_list', 'iter_count_list', 'start_counts')
% end
%% Success rate plot
% get the error bars
se = std(p_recovery_list,0,3)/sqrt(num_trials); % standard error mean

se_extra_top = (p_recovery + se) - 1; % amount that exceeds 1
se_chop = max(0,se_extra_top); % amount to chop off
se_top = se - se_chop; % top of error bar to plot
se_extra_bot = 0 - (p_recovery - se); % amount that falls below 0
se_chop = max(0,se_extra_bot); % amount to chop off
se_bot = se - se_chop; % bottom of error bar to plot
%

fig = figure()
set(fig, 'Units', 'Inches', 'Position', [0, 0, 4.4, 3.3]);
errorbar(gamma_list, p_recovery(:,1), se_bot(:,1), se_top(:,1), '-o', ...
    "MarkerEdgeColor",[0 0.4470 0.7410],"MarkerFaceColor",[0 0.4470 0.7410])
% hold on
% errorbar(gamma_list, p_recovery(:,2), se_bot(:,2), se_top(:,2),  '-^', ...
%    "MarkerEdgeColor",[0.8500 0.3250 0.0980],"MarkerFaceColor", [0.8500 0.3250 0.0980])
xlabel('$$M/N$$', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('success rate', 'Interpreter', 'latex', 'FontSize', 18)
% title(['Success rate for $$N=',num2str(N),'$$'], 'Interpreter', 'latex', 'FontSize', 18)
xlim([min(gamma_list) max(gamma_list)])
ylim([0 1])
legend(method_names, 'Interpreter', 'latex', 'FontSize', 16, 'Location', 'southeast')
grid on
savefig(['results/',runID,'.fig'])

if savepdf
set(fig,'Units','Inches');
pos = get(fig,'Position');
set(fig,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(fig,['results/',runID],'-dpdf','-r0')
end

%%
function [x, obj, crits] = gd(simul, f, gradf, x0, criterion)
max_iter = simul.max_iter;
tau = simul.tau;

x = x0;
obj = f(x0);
crits = criterion(x0);
for t=1:max_iter
    z = x - tau * gradf(x);
    obj(end+1) = f(z);
    crits(end+1) = criterion(z);
    x = z;
end

if 0
    figure
    plot(obj)
    figure
    plot(crits)
end

end