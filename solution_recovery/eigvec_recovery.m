%% Recovering solutions which are eigenvectors of a matrix to underdetermined linear systems
% Method 1)
%   Apply accelerated FISTA for min Ler(x) subject to y = Ax
%   Accelerated FISTA implementation uses: https://github.com/tomgoldstein/fasta-matlab

%%
addpath './solver'
clear
% close all
rng(10)
%%

N = 100;

method_list = [1,2];
method_names = {'$$\ell_{eig}$$','$$\ell_{\mu}$$'};
savepdf = 1; % only turn this on when at a final stage
comment = 'testeigvec_recovery_all2_gamma03to07by005';
max_restart_count = 10;
plot_obj = 0;  

epsilon = 1e-13;

num_trials = 1000;
gamma_list = [0.3:0.05:0.7];
thr_rec = 1e-2; % recovery

% Ler minimization
f_Cx = @(C,x) norm(C*x)^2 * norm(x)^2 - (x' * C*x)^2;
gradf_Cx = @(C,x) 2*(C' * (C*x))*norm(x)^2 + 2*norm(C*x)^2 * x - 4*(x' * C*x)*(C*x);

Llambda = @(C,x,lambda) norm((C-lambda*eye(N))*x)^2; % sum(abs(C*x - lambda*x).^2);
% ||Ax||_2^2 = x^TA^T Ax -> deriv is 2 A^TAx
% (Cx- lambdaIx)^T (Cx - lambdaIx)
% (x^TC^T - lambdax^T) (Cx - lambdaIx)
% x^TC^TCx - lambda x^TC^Tx - lambda x^T Cx + lambda^2 x^Tx
gradx_Llambda = @(C,x,lambda) 2*(C-lambda*eye(N))' * ((C-lambda*eye(N))*x);
gradlambda_Llambda = @(C,x,lambda) -2*x'*C*x + 2*lambda*(x'*x);
Laug = @(C,z) Llambda(C,z(1:end-1), z(end));
grad_Laug = @(C,z) [gradx_Llambda(C,z(1:end-1),z(end));
    gradlambda_Llambda(C,z(1:end-1),z(end))];

g = @(x) 0; % if A*x = b, otherwise inf but that will never be used since iterates are the output of projection
eye_fun = @(x) x;

criterion_rec = @(x, xstar) norm(x(1:N)-xstar)/norm(xstar) < thr_rec; % criterion to recover xstar
% opts.function = std_abs;

opts.maxIters = 10000;
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

for M_idx=1:length(M_list)
    M = M_list(M_idx);
    for t=1:num_trials
        tmpC = randn(N,N);
        C = (tmpC + tmpC')/2;
        [V,D] = eig(C);
        xstar = V(:,randi(N));

        A = randn(M,N);
        b = A * xstar;
        G = (A*A');
        criterion_rec_cur = @(x) criterion_rec(x,xstar); % current recovery criterion with xstar fixed
        opts.stopNow = @(x1,iter,resid,normResid,maxResidual,opts) criterion_rec_cur(x1);

        for method_idx=1:length(method_list)
            method = method_list(method_idx);

            if method == 1
                f = @(x) f_Cx(C,x);
                gradf = @(x) gradf_Cx(C,x);
            elseif method == 2
                f = @(z) Laug(C,z);
                gradf = @(z) grad_Laug(C,z);
            end
            start_count = 0;
            t0 = tic;
            if method == 1
                proxg = @(z,t) z - A'*(G\(A*z-b)); % linear projection on to Ax=b
            elseif method == 2
                proxg = @(z,t) [z(1:end-1) - A'*(G\(A*z(1:end-1)-b));
                    z(end)]; % linear projection on to Ax=b and leaving beta as is
            end
            boo = true;
            while start_count < max_restart_count && boo
                disp(['Start ', num2str(start_count)])
                if method == 1
                    x0 = proxg(randn(N,1), 0);
                    [xhat, outs] = fasta(eye_fun, eye_fun, f, gradf, g, proxg, x0, opts);
                elseif method == 2
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
            if plot_obj %&& criterion_rec_cur(xhat) && ~criterion_rec_cur(xhat_inf)
                figure; plot(outs.objective)
                flag=1;
            end

            p_recovery_list(M_idx,method_idx,t) = criterion_rec(xhat, xstar);
            iter_count_list(M_idx,method_idx,t) = iter_count;
            start_count_list(M_idx,method_idx,t) = start_count;
        end
        disp(['Trial ',num2str(t), ' for M = ', num2str(M)])
    end
    disp(['Finished for M ', num2str(M)])
end

p_recovery = squeeze(mean(p_recovery_list, 3));
iter_count = squeeze(mean(iter_count_list, 3));
%%
runID = [comment,'_',num2str(max_restart_count),'start','-N',num2str(N),'_','acc',num2str(opts.accelerate),'_adaptive',num2str(opts.adaptive), '_bt', num2str(opts.backtrack),'_trial', num2str(num_trials),'_fastamaxiters', num2str(opts.maxIters)];
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
errorbar(gamma_list, p_recovery(:,2), se_bot(:,2), se_top(:,2),  '-^', ...
   "MarkerEdgeColor",[0.8500 0.3250 0.0980],"MarkerFaceColor", [0.8500 0.3250 0.0980])

xlabel('$$M/N$$', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('success rate', 'Interpreter', 'latex', 'FontSize', 18)
% title(['Success rate for $$N=',num2str(N),'$$'], 'Interpreter', 'latex', 'FontSize', 18)
xlim([min(gamma_list) max(gamma_list)])
ylim([0 1])
legend(method_names, 'Interpreter', 'latex', 'FontSize', 16, 'Location', 'southeast')
grid on
savefig(['results/',runID,'.fig'])

%%
if savepdf
    set(fig,'Units','Inches');
    pos = get(fig,'Position');
    set(fig,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
    print(fig,['results/',runID],'-dpdf','-r0')
end
