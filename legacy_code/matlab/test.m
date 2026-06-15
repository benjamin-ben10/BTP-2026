clc; clear; close all;

%% 1. Simulation Parameters
N = 64;               % Number of subcarriers
M = 1000;             % Total IRS elements (M) for baseline OFDM
Q = 250;              % Number of IRS groups (Q) for proposed SEFDM
L = M / Q;            % Elements per group (L = 4)
alpha = 0.85;         % SEFDM compression factor (0 < alpha < 1)

SNR_dB = 0:2:20;      
num_blocks = 10000;    

BER_OFDM = zeros(length(SNR_dB), 1);
BER_SEFDM = zeros(length(SNR_dB), 1);

%% 2. Waveform Matrices (Time-Frequency Domain)
n_idx = (0:N-1)';
k_idx = 0:N-1;

% Generate transformation matrices 
F_OFDM = exp(1j * 2 * pi * n_idx * k_idx / N) / sqrt(N);
F_SEFDM = exp(1j * 2 * pi * n_idx * k_idx * alpha / N) / sqrt(N);

% Zero-Forcing Equalizer for OFDM (Orthogonal, so pinv is stable)
W_OFDM = pinv(F_OFDM);

% gram matrix for mmse 
Gram_SEFDM = F_SEFDM' * F_SEFDM; 

%% 3. Main Monte Carlo Loop
for snr_idx = 1:length(SNR_dB)
    snr = SNR_dB(snr_idx);
    err_ofdm = 0; 
    err_sefdm = 0;
    total_bits = 0;
    
    noise_var = 10^(-snr/10);
    
    for blk = 1:num_blocks
        
        % Generate BPSK Data 
        bits = randi([0 1], N, 1);
        s = 1 - 2*bits; % Maps 0 to +1, 1 to -1
        
        %  Waveform Modulation 
        x_ofdm = F_OFDM * s;
        x_sefdm = F_SEFDM * s;
        
        % Spatial Channel (IEG-IRS vs U-IRS)
        h_BI = (randn(M, 1) + 1j*randn(M, 1)) / sqrt(2);
        h_IU = (randn(M, 1) + 1j*randn(M, 1)) / sqrt(2);
        
        % Baseline 1 (U-IRS): Individual phase alignment
        v_U = exp(-1j * angle(h_BI .* h_IU));
        h_eff_ofdm = sum(v_U .* h_BI .* h_IU);
        
        % Proposed (IEG-IRS): Grouped phase alignment
        h_eff_sefdm = 0;
        for q = 1:Q
            idx = (q-1)*L + 1 : q*L;
            cascaded_group = h_BI(idx) .* h_IU(idx);
            v_q = exp(-1j * angle(sum(cascaded_group)));
            h_eff_sefdm = h_eff_sefdm + v_q * sum(cascaded_group);
        end
        
        % Normalize channel gain
        h_eff_ofdm = h_eff_ofdm / (M * (pi/4));
        h_eff_sefdm = h_eff_sefdm / (M * (pi/4));
        
        % Synthesized Received Signal 
        noise = sqrt(noise_var/2) * (randn(N, 1) + 1j*randn(N, 1));
        
        y_ofdm = h_eff_ofdm * x_ofdm + noise;
        y_sefdm = h_eff_sefdm * x_sefdm + noise;
        
        % Demodulation and Equalization 
        % 1. Baseline OFDM (Zero-Forcing)
        s_est_ofdm = W_OFDM * (y_ofdm / h_eff_ofdm);
        
        % 2. Proposed SEFDM (MMSE Equalization)
        % Calculate instantaneous SNR linear scale for the effective channel
        inst_snr_linear = (abs(h_eff_sefdm)^2) / noise_var;
        
        % Apply MMSE matrix inversion using the fast backslash (\) operator
        A_mmse = Gram_SEFDM + (1 / inst_snr_linear) * eye(N);
        b_mmse = F_SEFDM' * (y_sefdm / h_eff_sefdm);
        s_est_sefdm = A_mmse \ b_mmse;
        
        %  Bit Extraction & Error Counting 
        bits_est_ofdm = real(s_est_ofdm) < 0;
        bits_est_sefdm = real(s_est_sefdm) < 0;
        
        err_ofdm = err_ofdm + sum(bits ~= bits_est_ofdm);
        err_sefdm = err_sefdm + sum(bits ~= bits_est_sefdm);
        total_bits = total_bits + N; 
    end
    
    % BER
    BER_OFDM(snr_idx) = err_ofdm / total_bits;
    BER_SEFDM(snr_idx) = err_sefdm / total_bits;
end

%% 4. Plotting Results
figure('Color', 'w', 'Position', [100, 100, 700, 500]);
semilogy(SNR_dB, BER_OFDM, 'b-o', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'b');
hold on; grid on;
semilogy(SNR_dB, BER_SEFDM, 'r-s', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'r');

xlabel('Signal-to-Noise Ratio (SNR) [dB]', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Bit Error Rate (BER)', 'FontSize', 12, 'FontWeight', 'bold');
title('ISAC System: SEFDM+IEG-IRS vs Baseline OFDM (BPSK)', 'FontSize', 14);
legend('Baseline 1: OFDM with U-IRS (M=1000 elements)', ...
       sprintf('Proposed: SEFDM with IEG-IRS (Q=250 groups, \\alpha=%.2f)', alpha), ...
       'Location', 'southwest', 'FontSize', 11);

axis([min(SNR_dB) max(SNR_dB) 1e-5 1]);
set(gca, 'YMinorTick', 'on', 'FontSize', 11);