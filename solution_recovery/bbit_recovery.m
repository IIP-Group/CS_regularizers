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

N = 10;
method_list = [1];
method_names = {'$$\ell_{equ}$$' };

savepdf = 1; % only turn this on when at a final stage
comment = 'bbit_recovery-gamma03to09by01';
max_restart_count = 10;
plot_hist = 0;  % only set to 1 in debug mode to look at each xhat, and run Linf first

B = 2;
lambda = 1e-5; % 1e-3;
epsilon = 1e-13;

num_trials = 1000;
gamma_list = [0.3:0.1:0.9];

thr_rec = 1e-2; % recovery

% Lbin minimization
K = sum(4.^(2*[0:B-1]));
Lequ = @(Y) K*N * sum(sum(Y.^4)) - sum(4.^([0:B-1]).*sum(Y.^2,1))^2; 
gradLequ = @(Y) 4*Y.*(K*N*Y.^2 - 4.^([0:B-1])*sum(4.^([0:B-1]).*sum(Y.^2,1)));
Ltot = @(A,b,Y,lambda) norm(A*sum(Y,2)-b)^2 + lambda * Lequ(Y); 
gradLtot = @(A,b,Y,lambda) 2*repmat(A'*(A*sum(Y,2)-b),1,B) + lambda * gradLequ(Y); 

% f = @(x) N * sum(abs(x).^4) - sum(abs(x).^2)^2; % Lbin(x)
% gradf = @(x) 4*N * x.^3 - 4 * sum(x.^2) .* x; 
    
g = @(x) 0; % if A*x = b, otherwise inf but that will never be used since iterates are the output of projection
eye_fun = @(x) x;

std_abs = @(x) mean(std(abs(x), 1));
criterion_rec = @(x, xstar) norm(x-xstar)/norm(xstar) < thr_rec; % criterion to recover xstar

opts.maxIters = 1e3;
opts.backtrack = 1;
opts.accelerate = 0;
opts.adaptive = 1;
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
        Ystar = sign(randn(N,B)) .* 2.^[0:B-1];
        xstar = sum(Ystar,2);
        b = A * xstar;
        G = (A*A');
        criterion_rec_cur = @(Y) criterion_rec(sum(Y,2),xstar); % current recovery criterion with xstar fixed
        opts.stopNow = @(x1,iter,resid,normResid,maxResidual,opts) criterion_rec_cur(x1);
        
        for method_idx=1:length(method_list)
            method = method_list(method_idx);
            
            if method == 1
                f = @(Y) Ltot(A,b,Y,lambda);
                gradf = @(Y) gradLtot(A,b,Y,lambda);
                opts.function = f;
            end
            start_count = 0;
            t0 = tic;
            proxg = @(z,t) z;
            
            boo = true;
            while start_count < max_restart_count && boo
                disp(['Start ', num2str(start_count)])
                Y0 = proxg(randn(N,B), 0);
                [Yhat, outs] = fasta(eye_fun, eye_fun, f, gradf, g, proxg, Y0, opts);
                xhat = sum(Yhat,2);
                boo = ~criterion_rec_cur(xhat);
                start_count = start_count + 1;
            end
            iter_count = outs.iterationCount;
            deltat = toc(t0);
            disp(['FASTA finished ', num2str(deltat)])
            if plot_hist %&& criterion_rec_cur(xhat) && ~criterion_rec_cur(xhat_inf)
                figure;
                histogram(xhat, 100)
                xlim([-(2^B-1) 2^B-1])
                xlabel('x_n'); ylabel('count')
                figure;
                histogram(xstar, 100)
                xlim([-(2^B-1) 2^B-1])
                xlabel('xstar_n'); ylabel('count')
                figure; plot(outs.objective); ylabel('objective func')
                flag=1;
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
runID = [comment,'_B',num2str(B),'_lambda',num2str(lambda),'_',num2str(max_restart_count),'start','-N',num2str(N),'_','acc',num2str(opts.accelerate),'_adaptive',num2str(opts.adaptive), '_bt', num2str(opts.backtrack),'_trial', num2str(num_trials),'_fastamaxiters', num2str(opts.maxIters)];
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
hold on
% errorbar(gamma_list, p_recovery(:,2), se_bot(:,2), se_top(:,2),  '-^', ...
%    "MarkerEdgeColor",[0.8500 0.3250 0.0980],"MarkerFaceColor", [0.8500 0.3250 0.0980])
if size(p_recovery,2) > 2
    for i=3:size(p_recovery,2)
        errorbar(gamma_list, p_recovery(:,i), se_bot(:,i), se_top(:,i),  '-^')
    end
end
xlabel('$$M/N$$', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('success rate', 'Interpreter', 'latex', 'FontSize', 18)
% title(['Success rate for $$N=',num2str(N),'$$'], 'Interpreter', 'latex', 'FontSize', 18)
xlim([min(gamma_list) max(gamma_list)])
ylim([0 1])
legend(method_names, 'Interpreter', 'latex', 'FontSize', 16, 'Location', 'northwest')
grid on
% savefig(['results/',runID,'.fig'])

%%
if savepdf
    set(fig,'Units','Inches');
    pos = get(fig,'Position');
    set(fig,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
    print(fig,['results/',runID],'-dpdf','-r0')
end