clear; clc; close all;

%% Parameters
N = 100; M = 40; max_iter = 2000; %取一个实验上限
trials = 10;
SNR_dB_range = 0:5:25; 
pi_list = 0.5:0.1:1.0;
tol_mu = 1e-14;tol_sigma = 1e-12;  % 收敛阈值后续有平方

nmse_pi_support = zeros(length(SNR_dB_range), length(pi_list));
nmse_pi_baseline = zeros(length(SNR_dB_range), length(pi_list));
nmse_trials_support = zeros(trials, length(SNR_dB_range), length(pi_list));
nmse_trials_baseline = zeros(trials, length(SNR_dB_range), length(pi_list));

for p_idx = 1:length(pi_list)
    pi_val = pi_list(p_idx);
    disp(['Running π = ', num2str(pi_val)]);

    for snr_idx = 1:length(SNR_dB_range)
        SNR_dB = SNR_dB_range(snr_idx);

        for trial = 1:trials
            rng(710 + trial);

            %% === Clustered sparse signal generation ===
            x_true = zeros(N,1);
            cluster_size = 5;
            num_clusters = 2;
            s_true_idx = [];

            for c = 1:num_clusters
                start_idx = randi([1, N - cluster_size + 1]);
                idx = start_idx:(start_idx + cluster_size - 1);
                x_true(idx) = randn(cluster_size,1);
                s_true_idx = [s_true_idx, idx];
            end
            s_true_idx = unique(s_true_idx);

            A = randn(M,N); A = A ./ vecnorm(A);
            sigma2 = norm(A*x_true)^2 / (M*10^(SNR_dB/10));
            y = A*x_true + sqrt(sigma2)*randn(M,1);

            a0 = 1e-6; b0 = 1e-6; 
            E_beta = 1 / sigma2;

            %% === Prior π ===
            pi0 = 0.01 * ones(N,1);
            pi0(s_true_idx) = pi_val;  % oracle prior
            hard_mask = (pi0 == 1);

            %% === Support-based VBI with Turbo-style support update (Eq. 41-44, with active/inactive prior) ===
            mu_x = zeros(N,1); Sigma_x = eye(N);
            a_hat = zeros(N,1); b_hat = zeros(N,1);
            E_alpha = ones(N,1);
            s_hat = pi0;  % soft initialization
            
            % 设置 active/inactive 的 Gamma 超参数
            % active: 表示支持上的先验精度应非常小（鼓励非零）
            % inactive: 非支持上的先验应设置为 alpha = 100
            alpha_inactive = 1e6;
            a0_act = 1e-6; b0_act = 1e-6;
            a0_inact = 1; b0_inact = 1e-6;
            
            for iter = 1:max_iter
                mu_x_old = mu_x;
                Sigma_old_diag = diag(Sigma_x);
            
                Lambda = s_hat .* E_alpha + (1 - s_hat) .* alpha_inactive;
                try
                    L = chol(E_beta*A'*A + diag(Lambda), 'lower');
                    Sigma_x = inv(L') * inv(L);
                catch
                    Sigma_x = inv(E_beta*A'*A + diag(Lambda));
                end
                mu_x = E_beta * Sigma_x * A' * y;
            
                % === α 的更新（使用 π 加权 active/inactive 先验） ===
                E_x2 = mu_x.^2 + diag(Sigma_x);
                for i = 1:N
                    a_hat(i) = s_hat(i) * a0_act + (1 - s_hat(i)) * a0_inact + 1;
                    b_hat(i) = E_x2(i) + s_hat(i) * b0_act + (1 - s_hat(i)) * b0_inact;
                end
                E_alpha = a_hat ./ b_hat;
                E_ln_alpha = psi(a_hat) - log(b_hat);%注意函数使用
            
                % === π 后验更新（支持后验 s_hat） ===
                for i = 1:N
                    if ~hard_mask(i)
                        log_p1 = log(pi0(i)) + (a_hat(i) - 1) * E_ln_alpha(i) - b_hat(i) * E_alpha(i);
                        log_p0 = log(1 - pi0(i)) + (a_hat(i) - 1) * log(alpha_inactive) - b_hat(i) * alpha_inactive;
                        max_log = max([log_p1, log_p0]);
                        s_hat(i) = exp(log_p1 - max_log) / (exp(log_p1 - max_log) + exp(log_p0 - max_log));
                    else
                        s_hat(i) = 1;
                    end
                end
            
                % === 收敛判断 ===
                delta_mu = norm(mu_x - mu_x_old)^2; 
                delta_sigma = norm(diag(Sigma_x) - Sigma_old_diag)^2;
                if delta_mu < tol_mu && delta_sigma < tol_sigma
                    break;
                end
            end
            mu_x_sup = mu_x;
            nmse_trials_support(trial, snr_idx, p_idx) = norm(mu_x_sup - x_true)^2 / (norm(x_true)^2 + 1e-6);



            %% === Baseline VBI ===
            mu_x = zeros(N,1); Sigma_x = eye(N);
            a_hat = a0*ones(N,1); b_hat = b0*ones(N,1);
            E_alpha = a_hat ./ b_hat;

            for iter = 1:max_iter
                mu_x_old = mu_x;
                Sigma_old_diag = diag(Sigma_x);

                Lambda = E_alpha;
                try
                    L = chol(E_beta*A'*A + diag(Lambda), 'lower');
                    Sigma_x = inv(L') * inv(L);
                catch
                    Sigma_x = inv(E_beta*A'*A + diag(Lambda));
                end
                mu_x = E_beta * Sigma_x * A' * y;

                for i = 1:N
                    a_hat(i) = a0 + 0.5;
                    b_hat(i) = b0 + 0.5 * (mu_x(i)^2 + Sigma_x(i,i));
                end
                E_alpha = a_hat ./ b_hat;

                % 收敛判断
                delta_mu = norm(mu_x - mu_x_old)^2;
                delta_sigma = norm(diag(Sigma_x) - Sigma_old_diag)^2;
                if delta_mu < tol_mu && delta_sigma < tol_sigma
                    break;
                end
            end
            mu_x_baseline = mu_x;
            nmse_trials_baseline(trial, snr_idx, p_idx) = norm(mu_x_baseline - x_true)^2 / (norm(x_true)^2 + 1e-6);

%             % === 估计信号与真实信号对比 ===
%             if trial == 1 && SNR_dB == 20
%                 mu_x_sup_all(:, p_idx) = mu_x_sup;
%                 x_true_sample = x_true;
%                 s_hat_sample = s_hat;

%                 if p_idx == length(pi_list)
%                    % ========== 可视化所有结果 ==========
%                    % 1. Baseline VBI
%                   figure;
%                   stem(1:N, x_true_sample, 'k', 'LineWidth', 1.5); hold on;
%                   stem(1:N, mu_x_baseline, 'b');
%                   title('Baseline VBI vs True x'); legend('True x', 'Estimated x');
%                   xlabel('Index'); ylabel('Amplitude'); grid on;
%     
%                   % 2. 每个 π 下的 Support VBI
%                   for i = 1:length(pi_list)
%                       figure;
%                       stem(1:N, x_true_sample, 'k', 'LineWidth', 1.5); hold on;
%                       stem(1:N, mu_x_sup_all(:, i), 'r');
%                       title(['Support VBI Recovery (π = ', num2str(pi_list(i), '%.1f'), ')']);
%                       legend('True x', 'Estimated x');
%                       xlabel('Index'); ylabel('Amplitude'); grid on;
%                   end
%                end
% 
%                 figure;
%                     subplot(3,1,1);
%                     stem(1:N, x_true_sample, 'k', 'LineWidth', 1.5);
%                     title('True x (ground truth signal)');
%                     xlabel('Index'); ylabel('Amplitude'); grid on;
%                 
%                     subplot(3,1,2);
%                     stem(1:N, mu_x_sup, 'r');
%                     title(['Estimated \mu_x (Support VBI, \pi = ', num2str(pi_list(p_idx)), ')']);
%                     xlabel('Index'); ylabel('\mu_x'); grid on;
%                 
%                     subplot(3,1,3);
%                     stem(1:N, s_hat_sample, 'b');
%                     title('Posterior support probability s\_hat');
%                     xlabel('Index'); ylabel('s\_hat'); grid on;
%                 
%                     % === 可视化误检测情况（FP & FN）===
%                     s_true = double(x_true_sample ~= 0);
%                     s_est = double(s_hat_sample > 0.5);  % 或者设更高阈值
%                 
%                     false_positive = (s_true == 0) & (s_est == 1);
%                     false_negative = (s_true == 1) & (s_est == 0);
%                 
%                     figure;
%                     hold on;
%                     stem(find(s_true), ones(nnz(s_true),1), 'ko', 'DisplayName', 'True support');
%                     stem(find(s_est), 0.9*ones(nnz(s_est),1), 'r+', 'DisplayName', 'Estimated support');
%                     stem(find(false_positive), 0.5*ones(nnz(false_positive),1), 'm*', 'DisplayName', 'False Positives');
%                     stem(find(false_negative), 0.2*ones(nnz(false_negative),1), 'cx', 'DisplayName', 'False Negatives');
%                     title(['Support Recovery Comparison (\pi = ', num2str(pi_list(p_idx)), ')']);
%                     xlabel('Index'); ylim([0 1.2]);
%                     legend(); grid on;
%            end

        end

        nmse_pi_support(snr_idx,p_idx)  = mean(nmse_trials_support(:, snr_idx, p_idx));
        nmse_pi_baseline(snr_idx,p_idx) = mean(nmse_trials_baseline(:, snr_idx, p_idx));
    end
end

%% === NMSE：将所有 π 的曲线绘制在一张图中 ===
figure; hold on;

% === baseline 平均 NMSE（所有 π 下是一致的）===
avg_base = 10*log10(mean(mean(nmse_trials_baseline, 3), 1));
plot(SNR_dB_range, avg_base, 'k--', 'LineWidth', 2, 'DisplayName', 'Baseline VBI');

% === 每个 π 对应的 support VBI 曲线 ===
colors = lines(length(pi_list));
for p_idx = 1:length(pi_list)
    plot(SNR_dB_range, 10*log10(nmse_pi_support(:,p_idx)), '-o', ...
        'Color', colors(p_idx,:), ...
        'DisplayName', ['Support VBI, \pi = ', num2str(pi_list(p_idx), '%.1f')]);
end

xlabel('SNR (dB)'); ylabel('NMSE (dB)');
title('NMSE vs. SNR for Baseline and Support VBI with Different \pi');
legend('Location','southwest'); grid on;


%% === π vs 平均 NMSE 图 ===
figure;
avg_nmse_per_pi = squeeze(mean(mean(nmse_trials_support,1),2));
plot(pi_list, 10*log10(avg_nmse_per_pi), '-o','LineWidth',2);
xlabel('\pi'); ylabel('Average NMSE (dB)');
title('Support VBI: Avg NMSE vs. \pi');
grid on;
