clear; clc; close all;

%% Parameters
N = 100; M = 40; max_iter = 2000;
trials = 100;
SNR_dB_range = 0:1:25;
pi_list = 0.5:0.1:1.0;
tol_mu = 1e-4; tol_sigma = 1e-4;

nmse_trials_support = zeros(trials, length(SNR_dB_range), length(pi_list));
nmse_trials_baseline = zeros(trials, length(SNR_dB_range));  % 只保存一份 baseline

for p_idx = 1:length(pi_list)
    pi_val = pi_list(p_idx);

    for snr_idx = 1:length(SNR_dB_range)
        SNR_dB = SNR_dB_range(snr_idx);

        for trial = 1:trials
            rng(710 + trial);  % 保证每组 π 使用相同信号
            
            %% === Sparse signal generation ===
            x_true = zeros(N,1);
            cluster_size = 5; num_clusters = 2;
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
            E_beta = 1 / sigma2;

            a0 = 1e-6; b0 = 1e-6;

            %% === Prior π ===
            pi0 = 0.01 * ones(N,1);
            pi0(s_true_idx) = pi_val;
            hard_mask = false(N,1);  % 所有位置都允许更新

            %% === Support-VBI ===
            mu_x = zeros(N,1); Sigma_x = eye(N);
            a_hat = zeros(N,1); b_hat = zeros(N,1);
            E_alpha = ones(N,1);
            s_hat = pi0;

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

                E_x2 = mu_x.^2 + diag(Sigma_x);
                for i = 1:N
                    a_hat(i) = s_hat(i) * a0_act + (1 - s_hat(i)) * a0_inact + 1;
                    b_hat(i) = E_x2(i) + s_hat(i) * b0_act + (1 - s_hat(i)) * b0_inact;
                end
                E_alpha = a_hat ./ b_hat;
                E_ln_alpha = psi(a_hat) - log(b_hat);

                for i = 1:N
                    if ~hard_mask(i)
                        log_p1 = log(pi0(i)) + (a_hat(i)-1)*E_ln_alpha(i) - b_hat(i)*E_alpha(i);
                        log_p0 = log(1 - pi0(i)) + (a_hat(i)-1)*log(alpha_inactive) - b_hat(i)*alpha_inactive;
                        max_log = max([log_p1, log_p0]);
                        s_hat(i) = exp(log_p1 - max_log) / (exp(log_p1 - max_log) + exp(log_p0 - max_log));
                    else
                        s_hat(i) = 1;
                    end
                end

                delta_mu = norm(mu_x - mu_x_old);
                delta_sigma = norm(diag(Sigma_x) - Sigma_old_diag);
                if delta_mu < tol_mu && delta_sigma < tol_sigma
                    break;
                end
            end

            nmse_trials_support(trial, snr_idx, p_idx) = norm(mu_x - x_true)^2 / (norm(x_true)^2 + 1e-12);

            % === baseline 仅需执行一次 ===
            if p_idx == 1
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

                    delta_mu = norm(mu_x - mu_x_old);
                    delta_sigma = norm(diag(Sigma_x) - Sigma_old_diag);
                    if delta_mu < tol_mu && delta_sigma < tol_sigma
                        break;
                    end
                end
                nmse_trials_baseline(trial, snr_idx) = norm(mu_x - x_true)^2 / (norm(x_true)^2 + 1e-12);
            end

            %% === 输出当前进度 ===
            fprintf('[π = %.2f] SNR = %2d dB, Trial = %3d, Iterations = %d\n', ...
                    pi_val, SNR_dB, trial, iter);
        end
    end
end

%% === 平均 NMSE 计算 ===
nmse_pi_support = squeeze(mean(nmse_trials_support, 1)); % [SNR, π]
nmse_baseline = squeeze(mean(nmse_trials_baseline, 1));  % [SNR]

%% === 绘图：Support-VBI 不同 π 对比 + Baseline ===
figure; hold on;
plot(SNR_dB_range, 10*log10(nmse_baseline), 'k--', 'LineWidth', 2, 'DisplayName', 'Baseline VBI');

colors = lines(length(pi_list));
for p_idx = 1:length(pi_list)
    plot(SNR_dB_range, 10*log10(nmse_pi_support(:,p_idx)), '-o', ...
        'Color', colors(p_idx,:), ...
        'DisplayName', ['Support-VBI, \pi = ', num2str(pi_list(p_idx), '%.2f')]);
end
xlabel('SNR (dB)'); ylabel('NMSE (dB)');
title('NMSE vs. SNR for Support-VBI with Different \pi');
legend('Location', 'southwest'); grid on;
