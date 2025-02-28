%% Recovering one-sided binary or ternary solutions to underdetermined linear systems
% Method 1) 
%   Apply accelerated FISTA for min Losb (x)or Lter(x) subject to y = Ax
%   Accelerated FISTA implementation uses: https://github.com/tomgoldstein/fasta-matlab
% Method 2) 
%   Apply Douglas-Rachford splitting for min L1(x) subject to y = Ax

%%
addpath './solver'
clear
% close all
rng(0)
%%

N = 100;
M = 75;
method_list = [1,2];
method_names = {'CS-reg', '$$\ell^1$$'};
savepdf = 1;
comment = 'osb_ter-varydensity_02t08by002';
max_restart_count = 1;
init_from = 0;
plot_hist = 0; % only set to 1 in debug mode to look at each xhat

L1_t = 1;
L1_max_iter = 1e4;
L1_obj = @(x) sum(abs(x));

epsilon = 1e-13;

num_trials = 1000;
delta_list = [0.2:0.02:0.8];
thr_bin = 1e-2; % binary
thr_rec = 1e-2; % recovery

g = @(x) 0; % if A*x = b, otherwise inf but that will never be used since iterates are the output of projection
eye_fun = @(x) x;

criterion_rec = @(x, xstar) norm(x-xstar)/norm(xstar) < thr_rec;

opts.maxIters = 1e4;
% opts.tau = 1;
opts.backtrack = 1;
opts.accelerate = 0;
opts.adaptive = 1;
opts.accelerate = 1;
opts.adaptive = 0;
opts.tol = 1e-8;  % Use super strict tolerance
opts.recordObjective = true; %  Record the objective function so we can plot it
opts.verbose=false;
opts.verbose=true;
opts.stringHeader='    ';      % Append a tab to all text output from FISTA.  This option makes formatting look a bit nicer.

pgd_par.maxIters = 1e4;
pgd_par.tau = 1e-2;


p_recovery_list = zeros(length(delta_list), length(method_list), num_trials);
iter_count_list = zeros(length(delta_list), length(method_list), num_trials);
start_count_list = zeros(length(delta_list), length(method_list), num_trials);
ber_list = zeros(length(delta_list), length(method_list), num_trials);

for f_version=1:2

% f1
if f_version == 1 % ternary
    f = @(x) sum(x.^2) * sum(x.^6) - sum(x.^4)^2;
    gradf = @(x) 2 * x .* (sum(x.^6) + 3 * sum(x.^2) .* x.^4 - 4 * sum(x.^4) * x.^2);
elseif f_version == 2 % osb
    f = @(x) sum(x.^2) * sum(x.^4) - sum(x.^3)^2;
    gradf = @(x) 2 * x .* (sum(x.^4) + 2 * sum(x.^2) .* x.^2 - 3 * sum(x.^3) * x);
end
opts.function = @(x) f(x);


for idx=1:length(delta_list)
    delta = delta_list(idx);    
    num_nz = round(N*delta);
    num_zeros = round(N*(1-delta));
    for t=1:num_trials
        A = randn(M,N);
        if f_version == 1
            xstar = sign(randn(N,1));
        elseif f_version == 2
            xstar = ones(N,1);
        end
        zero_idcs = randsample(N,num_zeros);
        xstar(zero_idcs) = zeros(num_zeros,1);
        b = A * xstar;
        G = (A*A');
        
        criterion_rec_cur = @(x1)  criterion_rec(x1, xstar);
        opts.stopNow = @(x1,iter,resid,normResid,maxResidual,opts) criterion_rec_cur(x1);
        
        for method_idx=1:length(method_list)
            method = method_list(method_idx);
            
            if method == 2 % L1 min
                t0 = tic;
                proxg = @(z) z - A'*(G\(A*z-b)); % linear projection on to Ax=b
                x0 = randn(N,1);
                [xhat,obj] = L1_minim(proxg, L1_t, x0, L1_max_iter, criterion_rec_cur);
                iter_count = length(obj);
                deltat = toc(t0);
                disp(['L1 finished ', num2str(deltat)])
                xhat_L1 = xhat;
                start_count = 1;
                
            elseif method == 1 % CS regularizer
                start_count = 0;
                t0 = tic;
                proxg = @(z,t) z - A'*(G\(A*z-b)); % linear projection on to Ax=b
                x0 = randn(N,1);
                boo = true;
                while start_count < max_restart_count && boo
                    disp(['Start ', num2str(start_count)])
                    [xhat, outs] = fasta(eye_fun, eye_fun, f, gradf, g, proxg, x0, opts);
                    boo = ~criterion_rec(xhat, xstar);
                    start_count = start_count + 1;
%                     x0 = proxg(randn(N,1), 0);
                end
                if ~boo
                    hey = 1;
                end
                if plot_hist
                    figure; plot(outs.funcValues)
                    figure;histogram(xhat, 100)
                end
                iter_count = outs.iterationCount;
                deltat = toc(t0);

                disp(['FASTA finished ', num2str(deltat)])
                
            end
            
            p_recovery_list(idx,method_idx,t) = criterion_rec(xhat, xstar);
            iter_count_list(idx,method_idx,t) = iter_count;
            start_count_list(idx,method_idx,t) = start_count;
            ber_list(idx,method_idx,t) = sum(sign(xhat) ~= sign(xstar));
        end
        disp(['Trial ',num2str(t), ' for delta = ', num2str(delta)])
    end
    disp(['Finished for delta ', num2str(delta)])
end
p_recovery = squeeze(mean(p_recovery_list, 3));
iter_count = squeeze(mean(iter_count_list, 3));
%%
runID = [comment,'_f',num2str(f_version),'_init',num2str(init_from),'_',num2str(max_restart_count),'start','-N',num2str(N),'_','acc',num2str(opts.accelerate),'_adaptive',num2str(opts.adaptive), '_bt', num2str(opts.backtrack),'_trial', num2str(num_trials),'_fastamaxiters', num2str(opts.maxIters), '_L1maxiters', num2str(L1_max_iter)];

if f_version == 1
    f_str = '$$\ell_{ter}$$'
else 
    f_str = '$$\ell_{osb}$$'
end
method_names{1} = f_str;

%%
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

fig = figure()
set(fig, 'Units', 'Inches', 'Position', [0, 0, 4.4, 3.3]);
errorbar(delta_list, p_recovery(:,1), se_bot(:,1), se_top(:,1), '-o', ...
    "MarkerEdgeColor",[0 0.4470 0.7410],"MarkerFaceColor",[0 0.4470 0.7410])
hold on
errorbar(delta_list, p_recovery(:,2), se_bot(:,2), se_top(:,2),  '-^', ...
   "MarkerEdgeColor",[0.8500 0.3250 0.0980],"MarkerFaceColor", [0.8500 0.3250 0.0980])
xlabel('$$K/N$$', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('success rate', 'Interpreter', 'latex', 'FontSize', 18)
% title(['Success rate for $$N=',num2str(N),'$$'], 'Interpreter', 'latex', 'FontSize', 18)
xlim([min(delta_list) max(delta_list)])
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
end
%% Functions
%% Linf-norm minimization using Douglas-Rachford splitting
function [x, obj] = L1_minim(proxg,t,x0, max_iter, criterion)
x = x0;
z = zeros(size(x));
xprev = x;
for ii=1:max_iter
    % Douglas - Rachford splitting
    w = 2*x-z;
    y = proxL1(w,t);
    z = z + (y-x);
    x = proxg(z); % prox Ax=b
    
    % record stats
    obj(ii) = sum(abs(x));
    if criterion(x)
        break
    end
    xprev = x;
end
end
