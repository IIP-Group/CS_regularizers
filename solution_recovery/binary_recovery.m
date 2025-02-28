%% Recovering binary solutions to underdetermined linear systems
% Method 1) 
%   Apply accelerated FISTA for min Lbin(x) subject to y = Ax
%   Accelerated FISTA implementation uses: https://github.com/tomgoldstein/fasta-matlab
% Method 2) 
%   Apply Douglas-Rachford splitting for min Linf(x) subject to y = Ax

%%
addpath './solver'
clear
% close all
rng(10)
%%

N = 100;
method_list = [1,2,3,4,5,6,-1,-2,0];
method_names = {'$$\ell_{bin}$$','$$\ell_{bin,H}$$','$$\ell_{bin,si}$$', ...
                '$$\tilde\ell_{bin}$$', '$$\ell_{bbin}$$', '$$\tilde\ell_{bin,exp}$$', ...
                '$$\ell_{bin,\beta=1}$$','$$\ell_{bin,\beta}$$','$$\ell^{\infty}$$' };
method_list = [1,0]; % choose a subset
method_names = {'$$\ell_{bin}$$','$$\ell^{\infty}$$'};
savepdf = 1; % only turn this on when at a final stage
comment = 'binary_recovery_lbinandlinf_gamma03to09by002';
max_restart_count = 10;
plot_hist = 0;  % only set to 1 in debug mode to look at each xhat, and run Linf first

Linf_t = 10;
Linf_max_iter = 1e4;

epsilon = 1e-13;

num_trials = 1000;
gamma_list = [0.3:0.02:0.9];
thr_rec = 1e-2; % recovery

% Lbin minimization
Lbin = @(x) N * sum(x.^4) - sum(x.^2)^2; % Lbin(x)
gradLbin = @(x) 4*N * x.^3 - 4 * sum(x.^2) .* x; 

L4L2 = @(x) sqrt(N) * sqrt(sum(x.^4)) - sum(x.^2); % L4-L2
gradL4L2 = @(x) sqrt(N)*2 * 1/sqrt(sum(x.^4)) * x.^3 - 2 * x;

L4overL2 = @(x) sqrt(N) * sqrt(sum(x.^4)) / sum(x.^2) - 1; % L4/L2
gradL4overL2 = @(x) sqrt(N) * ( 2*(1/sqrt(sum(x.^4))*x.^3) * sum(x.^2) - sqrt(sum(x.^4)) * 2*x  ) / sum(x.^2)^2;

L2L1 = @(x) N*sum(x.^2) - sum(abs(x))^2; % L2 - L1
gradL2L1 = @(x) 2*N*x - 2*sum(abs(x))*sign(x);

Lbbin = @(x) N * sum(1./(1+x.^2).^2) - sum(1./(1+x.^2))^2; % Lbbin(x)
gradLbbin = @(x) -4*N * x ./ (1+x.^2).^3 + 4 * sum(1./(1+x.^2)) * x./ (1+x.^2).^2 ; 

Lexpbin = @(x) N * sum(exp(-2*x.^2)) - sum(exp(-x.^2))^2; % Lexpbin(x)
gradLexpbin = @(x) -4*N * x .* exp(-2*x.^2) + 4 * sum(exp(-x.^2)) * x .* exp(-x.^2); 

Lbeta = @(x,beta) sum((x.^2-beta).^2);
gradx_Lbeta = @(x,beta) 4 * (x.^2-beta) .* x;
gradbeta_Lbeta = @(x,beta) 2 * sum(-x.^2+beta);
Laug = @(z) Lbeta(z(1:end-1), z(end));
grad_Laug = @(z) [gradx_Lbeta(z(1:end-1),z(end)); 
                 gradbeta_Lbeta(z(1:end-1),z(end))];

% f = @(x) N * sum(abs(x).^4) - sum(abs(x).^2)^2; % Lbin(x)
% gradf = @(x) 4*N * x.^3 - 4 * sum(x.^2) .* x; 
    
g = @(x) 0; % if A*x = b, otherwise inf but that will never be used since iterates are the output of projection
eye_fun = @(x) x;

std_abs = @(x) std(abs(x), 1);
criterion_rec = @(x, xstar) norm(x-xstar)/norm(xstar) < thr_rec; % criterion to recover xstar
opts.function = std_abs;

opts.maxIters = 1e4;
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
iter_count_list = zeros(length(M_list), length(method_list), num_trials);
start_count_list = zeros(length(M_list), length(method_list), num_trials);
ber_list = zeros(length(M_list), length(method_list), num_trials);

for M_idx=1:length(M_list)
    M = M_list(M_idx);
    for t=1:num_trials
        A = randn(M,N);
        xstar = sign(randn(N,1));
        b = A * xstar;
        G = (A*A');
        criterion_rec_cur = @(x) criterion_rec(x(1:N),xstar); % current recovery criterion with xstar fixed
        opts.stopNow = @(x1,iter,resid,normResid,maxResidual,opts) criterion_rec_cur(x1);
        
        for method_idx=1:length(method_list)
            method = method_list(method_idx);
            
            % Linf min
            if method == 0
                t0 = tic;
                proxg = @(z) z - A'*(G\(A*z-b)); % linear projection on to Ax=b
                x0 = randn(N,1);
                [xhat,obj,track] = Linf_minim(proxg, Linf_t, x0, Linf_max_iter, std_abs, criterion_rec_cur);
                iter_count = length(obj);
                deltat = toc(t0);
                disp(['Linf finished ', num2str(deltat), ' in ', num2str(length(obj))])
                xhat_inf = xhat;
                boo = ~criterion_rec_cur(xhat);
                start_count = 1;
                
            else
                if method == 1
                    f = Lbin;
                    gradf = gradLbin;
                elseif method == 2
                    f = L4L2;
                    gradf = gradL4L2;
                elseif method == 3
                    f = L4overL2;
                    gradf = gradL4overL2;
                elseif method == 4
                    f = L2L1;
                    gradf = gradL2L1;
                elseif method == 5
                    f = Lbbin;
                    gradf = gradLbbin;
                elseif method == 6
                    f = Lexpbin;
                    gradf = gradLexpbin;
                elseif method == -1
                    f = @(x) Lbeta(x,1);
                    gradf = @(x) gradx_Lbeta(x,1);
                elseif method == -2
                    f = @(z) Laug(z);
                    gradf = @(z) grad_Laug(z);
                end
                start_count = 0;
                t0 = tic;
                if method > -2
                    proxg = @(z,t) z - A'*(G\(A*z-b)); % linear projection on to Ax=b
                elseif method == -2
                    proxg = @(z,t) [z(1:end-1) - A'*(G\(A*z(1:end-1)-b)); 
                                    z(end)]; % linear projection on to Ax=b and leaving beta as is
                end
                boo = true;
                while start_count < max_restart_count && boo
                    disp(['Start ', num2str(start_count)])
                    if method > -2
                        x0 = proxg(randn(N,1), 0);
                        [xhat, outs] = fasta(eye_fun, eye_fun, f, gradf, g, proxg, x0, opts);
                    elseif method == -2
                        x0_tmp = randn(N,1);
                        x0 = proxg([x0_tmp; mean(x0_tmp.^2)], 0);
                        [xhat_aug, outs] = fasta(eye_fun, eye_fun, f, gradf, g, proxg, x0, opts);
                        xhat = xhat_aug(1:N);
                    end
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
                    figure;
                    histogram(xhat_inf, 100)
                    xlim([-1.2 1.2])
                    xlabel('x_n'); ylabel('count')
                    figure; plot(outs.objective)
                    if method == -2
                        disp(xhat_aug(end))
                    end
                    flag = 1;
                end
            end

            p_recovery_list(M_idx,method_idx,t) = criterion_rec(xhat, xstar);
            iter_count_list(M_idx,method_idx,t) = iter_count;
            start_count_list(M_idx,method_idx,t) = start_count;
            ber_list(M_idx,method_idx,t) = mean(sign(xhat) ~= sign(xstar));
        end
        disp(['Trial ',num2str(t), ' for M = ', num2str(M)])
    end
    disp(['Finished for M ', num2str(M)])
end

p_recovery = squeeze(mean(p_recovery_list, 3));
iter_count = squeeze(mean(iter_count_list, 3));
%%
runID = [comment,'_Lbin_',num2str(max_restart_count),'start','-N',num2str(N),'_','acc',num2str(opts.accelerate),'_adaptive',num2str(opts.adaptive), '_bt', num2str(opts.backtrack),'_trial', num2str(num_trials),'_fastamaxiters', num2str(opts.maxIters), '_Linfmaxiters', num2str(Linf_max_iter)];
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

if length(method_list) > 2
    marker_list = {"-*", '-o', '-^', "-diamond", "-x", "->", "-<"}
else
    marker_list = {'-o', '-^'}
end
fig = figure()
set(fig, 'Units', 'Inches', 'Position', [0, 0, 4.4, 3.3]);
errorbar(gamma_list, p_recovery(:,1), se_bot(:,1), se_top(:,1), marker_list{1}, ...
    "MarkerEdgeColor",[0 0.4470 0.7410],"MarkerFaceColor",[0 0.4470 0.7410])
hold on
errorbar(gamma_list, p_recovery(:,2), se_bot(:,2), se_top(:,2), marker_list{2}, ...
   "MarkerEdgeColor",[0.8500 0.3250 0.0980],"MarkerFaceColor", [0.8500 0.3250 0.0980])
if size(p_recovery,2) > 2
    for i=3:size(p_recovery,2)
        errorbar(gamma_list, p_recovery(:,i), se_bot(:,i), se_top(:,i),  marker_list{i})
    end
end
xlabel('$$M/N$$', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('success rate', 'Interpreter', 'latex', 'FontSize', 18)
% title(['Success rate for $$N=',num2str(N),'$$'], 'Interpreter', 'latex', 'FontSize', 18)
xlim([min(gamma_list) max(gamma_list)])
ylim([0 1])
legend(method_names, 'Interpreter', 'latex', 'FontSize', 16, 'Location', 'southeast')
grid on
savefig(['results/',runID,'.fig'])

%
if savepdf
    set(fig,'Units','Inches');
    pos = get(fig,'Position');
    set(fig,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
    print(fig,['results/',runID],'-dpdf','-r0')
end
%% Functions
%% Linf-norm minimization using Douglas-Rachford splitting
function [x, obj, func_val] = Linf_minim(proxg,t,x0, max_iter, func, criterion)
%     PAR reduction by minimizing Linf norm using CRAMP (Algorithm 3) in
% [1] C. Studer, T. Goldstein, W. Yin, and R. G. Baraniuk, "Democratic
%     representations," Apr. 2015. [Online].
%     Available: http://arxiv.org/abs/1401.3420

x = x0;
z = zeros(size(x));
xprev = x;
for ii=1:max_iter
    % Douglas - Rachford splitting
    w = 2*x-z;
    y = proxInf(w,t);
    z = z + (y-x);
    x = proxg(z); % prox Ax=b
    
    % record stats
    obj(ii) = max(abs(x));
    func_val(ii) = func(x);
    if criterion(x)
        break
    end
    xprev = x;
end
end