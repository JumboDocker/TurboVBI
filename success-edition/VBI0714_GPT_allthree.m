clear; clc; close all;

%% Parameters
N = 100; M = 40; max_iter = 5000;
trials = 100;
SNR_dB_range = 0:5:25;

tol_mu = 1e-10; tol_sigma = 1e-8;

nmse_support = zeros(trials, length(SNR_dB_range));
nmse_baseline = zeros(trials, length(SNR_dB_range));
nmse_oracle = zeros(trials, length(SNR_dB_range));

save_visual = struct();

for snr_idx = 1:length(SNR_dB_range)
    SNR_dB = SNR_dB_range(snr_idx);

    for trial = 1:trials
        rng(910  + trial);

        %% === Generate clustered sparse signal ===
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
        s_true = zeros(N,1); s_true(s_true_idx) = 1;

        A = randn(M,N); A = A ./ vecnorm(A);
        sigma2 = norm(A*x_true)^2 / (M*10^(SNR_dB/10));
        y = A*x_true + sqrt(sigma2)*randn(M,1);

        E_beta = 1 / sigma2;
        a0 = 1e-6; b0 = 1e-6;
        a0_act = 1e-6; b0_act = 1e-6;
        a0_oracleinact = 1; b0_inact = 1e-6;
        a0_adaptinact = 5e-5; %adaptive pi

        %% === Support-VBI with adaptive pi ===
        mu_x = zeros(N,1); Sigma_x = eye(N);
        a_hat = zeros(N,1); b_hat = zeros(N,1);
        E_alpha = ones(N,1);
        s_hat = 0.1 * ones(N,1);
        alpha_inactive = 1e2;

        pi_evolution = zeros(N, max_iter);

        for iter = 1:max_iter
            mu_x_old = mu_x;
            Sigma_diag_old = diag(Sigma_x);

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
                a_hat(i) = s_hat(i) * a0_act + (1 - s_hat(i)) * a0_adaptinact + 1;
                b_hat(i) = E_x2(i) + s_hat(i) * b0_act + (1 - s_hat(i)) * b0_inact;
            end
            E_alpha = a_hat ./ b_hat;
            E_ln_alpha = psi(a_hat) - log(b_hat);

            for i = 1:N
                log_p1 = log(s_hat(i)) + (a_hat(i)-1)*E_ln_alpha(i) - b_hat(i)*E_alpha(i);
                log_p0 = log(1 - s_hat(i)) + (a_hat(i)-1)*log(alpha_inactive) - b_hat(i)*alpha_inactive;
                max_log = max([log_p1, log_p0]);
                s_hat(i) = exp(log_p1 - max_log) / (exp(log_p1 - max_log) + exp(log_p0 - max_log));
            end

            pi_evolution(:, iter) = s_hat;

            delta_mu = norm(mu_x - mu_x_old)^2;
            delta_sigma = norm(diag(Sigma_x) - Sigma_diag_old)^2;
            if delta_mu < tol_mu && delta_sigma < tol_sigma
                break;
            end
        end
        nmse_support(trial, snr_idx) = norm(mu_x - x_true)^2 / (norm(x_true)^2 + 1e-12);
        fprintf('[Support-VBI] SNR=%ddB, Trial=%d, Iter=%d\n', SNR_dB, trial, iter);

        if SNR_dB == 20 && trial == 10
            save_visual.mu_x_support = mu_x;
            save_visual.s_hat = s_hat;
            save_visual.pi_evolution = pi_evolution;
            save_visual.iter_support = iter;
        end

        %% === Baseline-VBI ===
        mu_x = zeros(N,1); Sigma_x = eye(N);
        a_hat = a0 * ones(N,1); b_hat = b0 * ones(N,1);
        E_alpha = a_hat ./ b_hat;

        for iter = 1:max_iter
            mu_x_old = mu_x;
            Sigma_diag_old = diag(Sigma_x);

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

            delta_mu = norm(mu_x - mu_x_old)^2;
            delta_sigma = norm(diag(Sigma_x) - Sigma_diag_old)^2;
            if delta_mu < tol_mu && delta_sigma < tol_sigma
                break;
            end
        end
        nmse_baseline(trial, snr_idx) = norm(mu_x - x_true)^2 / (norm(x_true)^2 + 1e-12);
        fprintf('[Baseline-VBI] SNR=%ddB, Trial=%d, Iter=%d\n', SNR_dB, trial, iter);

        if SNR_dB == 20 && trial == 10
            save_visual.mu_x_baseline = mu_x;
        end

        %% === Oracle-VBI ===
        mu_x = zeros(N,1); Sigma_x = eye(N);
        a_hat = zeros(N,1); b_hat = zeros(N,1);
        E_alpha = ones(N,1);
        s_oracle = s_true;
        alpha_inactive = 1e6;

        for iter = 1:max_iter
            mu_x_old = mu_x;
            Sigma_diag_old = diag(Sigma_x);

            Lambda = s_oracle .* E_alpha + (1 - s_oracle) .* alpha_inactive;
            try
                L = chol(E_beta*A'*A + diag(Lambda), 'lower');
                Sigma_x = inv(L') * inv(L);
            catch
                Sigma_x = inv(E_beta*A'*A + diag(Lambda));
            end
            mu_x = E_beta * Sigma_x * A' * y;

            E_x2 = mu_x.^2 + diag(Sigma_x);
            for i = 1:N
                a_hat(i) = s_oracle(i) * a0_act + (1 - s_oracle(i)) * a0_oracleinact + 1;
                b_hat(i) = E_x2(i) + s_oracle(i) * b0_act + (1 - s_oracle(i)) * b0_inact;
            end
            E_alpha = a_hat ./ b_hat;

            delta_mu = norm(mu_x - mu_x_old)^2;
            delta_sigma = norm(diag(Sigma_x) - Sigma_diag_old)^2;
            if delta_mu < tol_mu && delta_sigma < tol_sigma
                break;
            end
        end
        nmse_oracle(trial, snr_idx) = norm(mu_x - x_true)^2 / (norm(x_true)^2 + 1e-12);
        fprintf('[Oracle-VBI] SNR=%ddB, Trial=%d, Iter=%d\n', SNR_dB, trial, iter);

        if SNR_dB == 20 && trial == 10
            save_visual.mu_x_oracle = mu_x;
            save_visual.x_true = x_true;
            save_visual.s_true = s_true;
        end
    end
end

%% === 可视化（仅 SNR=20dB，第10次实验） ===
figure;
subplot(3,1,1); stem(1:N, save_visual.x_true, 'k--'); hold on;
stem(1:N, save_visual.mu_x_support, 'r'); title('Support-VBI');
subplot(3,1,2); stem(1:N, save_visual.x_true, 'k--'); hold on;
stem(1:N, save_visual.mu_x_baseline, 'b'); title('Baseline-VBI');
subplot(3,1,3); stem(1:N, save_visual.x_true, 'k--'); hold on;
stem(1:N, save_visual.mu_x_oracle, 'g'); title('Oracle-VBI');

figure;
max_valid_iter = save_visual.iter_support;
pi_valid = save_visual.pi_evolution(:,1:max_valid_iter);
[X,Y] = meshgrid(1:max_valid_iter, 1:N);
surf(X,Y,pi_valid, 'EdgeColor', 'none'); shading interp;
view(-30,30);
xlabel('Iteration'); ylabel('Index'); zlabel('\pi');
title('\pi Evolution in Support-VBI'); colormap(jet); colorbar;

figure;
subplot(3,1,1); stem(save_visual.s_true, 'k'); title('True Support');
subplot(3,1,2); stem(double(save_visual.s_hat > 0.5), 'r'); title('Estimated Support (Support-VBI)');
subplot(3,1,3); stem(save_visual.s_true, 'g'); title('Oracle Support');

figure;
plot(SNR_dB_range, 10*log10(mean(nmse_support,1)), '-or', 'DisplayName', 'Support-VBI'); hold on;
plot(SNR_dB_range, 10*log10(mean(nmse_baseline,1)), '--b', 'DisplayName', 'Baseline');
plot(SNR_dB_range, 10*log10(mean(nmse_oracle,1)), '-.g', 'DisplayName', 'Oracle');
xlabel('SNR (dB)'); ylabel('Average NMSE (dB)'); title('NMSE vs. SNR');
legend; grid on;
