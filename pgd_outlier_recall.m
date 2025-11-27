% -----------------------------------------------------------
% clear; clc;


%% ----------------------------------------------------------

% ==== Experiment hyper-params ===============================
% Choose your synthetic data shape & clustering
K           = 400;        % <-- true number of mixture components
d           = 200;        % <-- dimensionality
N           = 50000;       % <-- total number of points

% K           = 800;        % <-- true number of mixture components
% d           = 200;        % <-- dimensionality
% N           = 100000;       % <-- total number of points

% Mixture geometry (tune to make clusters tighter/looser)
clusterStd  = 1.0;       % isotropic std for each component
separation  = 5.0;       % centroid scale; larger => more separated

% Outlier & evaluation settings (same spirit as your original)
kList       = [K];       % which k to run for k-means/DR-KM (default = K)
numTrials   = 1;
factor      = 1e3;         % corruption scale for outliers
rngOffset   = 2025;      % seed base
f           = 0.1;      % outlier fraction

% For compatibility with your downstream code:
dim         = d;         % (your drlm uses 'd' and 'dim')
nTrain      = N;         % keep your 'nTrain' usage
Ntrain      = nTrain;


%% 1) Generate synthetic Gaussian Mixture data (d × N)
% Uniform component weights; change if you want non-uniform
weights = ones(1, K) / K;

rng(rngOffset);                % deterministic data for a given trial setup
MU = separation * randn(d, K); % random centroids, scaled by 'separation'

% isotropic covariances (clusterStd^2 * I)
Sigmas = zeros(d, d, K);
for j = 1:K
    Sigmas(:, :, j) = (clusterStd^2) * eye(d);
end

% Sample component assignments
% (requires Statistics & Machine Learning Toolbox)
zComp = randsample(K, N, true, weights).';  % 1×N component labels

% Draw points per component and assemble X (d × N)
data_d = zeros(d, N);
for j = 1:K
    idx = (zComp == j);
    Nj  = nnz(idx);
    if Nj > 0
        % mvnrnd expects row-mean; returns Nj×d -> transpose to d×Nj
        data_d(:, idx) = mvnrnd(MU(:, j).', Sigmas(:, :, j), Nj).';
    end
end

% For minimal changes to your downstream code, keep the name 'data50'
data50 = data_d;

% Optional: keep true labels if you want to analyze later
trueLabels = zComp;


%% 2) Storage for results -----------------------------------
rec_km  = zeros(numTrials, numel(kList));
rec_dr  = zeros(numTrials, numel(kList));
rec_tkm = zeros(numTrials, numel(kList));   % (kept for compatibility)


%% 3) Trial loop ---------------------------------------------
for t = 1:numTrials
    seed = rngOffset + t;
    rng(seed);

    % Outlier indices & mask
    z = ceil(f * Ntrain);                 % number of outliers
    outIdx  = randperm(Ntrain, z);
    X       = data50;                     % fresh copy
    trueOut = false(1, Ntrain); 
    trueOut(outIdx) = true;

    % -------- Add NeurIPS-like uniform noise (scaled by data range) -----
    gMin   = min(data50(:)) * factor;
    gMax   = max(data50(:)) * factor;
    noiseMat = gMin + (gMax - gMin) .* rand(dim, z, 'like', data50);  % d × z
    signFlip = randi([0 1], 1, z, 'like', data50); 
    signFlip(signFlip==0) = -1;

    X(:, outIdx) = X(:, outIdx) + noiseMat .* signFlip;

    % ----------------------- for each k ----------------------
    for kIdx = 1:numel(kList)
        k = kList(kIdx);
        disp(k);

        % ---------- baseline k-means++ ----------------------
        tic;
        [~, C] = kmeans(X', k, 'Start','plus','MaxIter',1);
        dists  = pdist2(X', C, 'squaredeuclidean');
        minD   = min(dists, [], 2);
        [~, w] = maxk(minD, z);
        pred   = false(1, Ntrain); pred(w) = true;
        rec_km(t, kIdx) = nnz(pred & trueOut) / z;
        toc;
        disp('=============== kmeans++ done =====================');

        % ---------- DR-K-Means (your routine) ---------------
        tic;
        gamma = 1.1; r = 3;
        [m, ~, ~, ~] = drlm(gamma, C', r, d, X, Ntrain);
        d2     = pdist2(X', m', 'squaredeuclidean');
        minD2  = min(d2, [], 2);
        [~, w2] = maxk(minD2, z);
        pred2  = false(1, Ntrain); pred2(w2) = true;
        rec_dr(t, kIdx) = nnz(pred2 & trueOut) / z;
        toc;
        disp('=============== DR - kmeans++ done ================');
    end

    fprintf("trial %2d/%d done (seed %d)\n", t, numTrials, seed);
end

%% 4) Summary (uncomment if you want standard prints) --------
fprintf("\n==== Outlier-Recall  (mean ± std over %d trials) ====\n", numTrials);
for kIdx = 1:numel(kList)
    k = kList(kIdx);
    fprintf("k = %2d |  KM  %.4f ± %.4f   DR-KM  %.4f ± %.4f\n", ...
            k,  mean(rec_km(:,kIdx)),  std(rec_km(:,kIdx)), ...
            mean(rec_dr(:,kIdx)),  std(rec_dr(:,kIdx)));
end

fprintf("\n==== Outlier-Recall  (max  /  avg  /  min over %d trials) ====\n", numTrials);
for kIdx = 1:numel(kList)
    k = kList(kIdx);
    km_vals  = rec_km(:,kIdx);
    dr_vals  = rec_dr(:,kIdx);
    fprintf("k = %2d | KM-++   %.4f / %.4f / %.4f   DR-KM   %.4f / %.4f / %.4f\n", ...
            k,  max(km_vals),  mean(km_vals),  min(km_vals), ...
               max(dr_vals),  mean(dr_vals),  min(dr_vals));
end


% ================== your function (unchanged) ==================
function [a, b, c, d] = drlm(gamma, M0, r, d, X, N)
    ITER_MAX = 100;
    REL_ERR_TOL = 1E-3;
    K = size(M0,2);

    pd = makedist('Normal');

    M = zeros(ITER_MAX, d, K);
    wQE = zeros(ITER_MAX-1,1);
    M(1,:, :) = M0;

    L = 1;
    beta = 1e-2;
    PI = ones(K,N)/K;
    wmin = Inf;
    sd = sum(X.*X, 1)';

    tic;
    for t = 1:ITER_MAX-1
        [M_next, PI, wQE_curr] = findNextM_v7_fast(M(t, :, :), PI, X, gamma, L, beta, d, sd);
        % [M_next, PI, wQE_curr] = findNextM_v8_gram(M(t, :, :), PI, X, gamma, L, beta, d, sd);

        M(t+1,:, :) = M_next;
        wQE(t) = wQE_curr;

        if t > 1 && abs((wQE(t) - wQE(t -1))/wQE(t -1)) < 1e-2
            M(end, :, :) = M_next;
            wQE_f = wQE(t);
            break;
        end
        if t == ITER_MAX - 1
            wQE_f = wQE(t);
            disp('not converged')
        end
    end
    toc;

    alpha = 1/(gamma-1); %#ok<NASGU>
    M_end = reshape(M(end, :, :), d, K);

    grad = -1; % placeholder

    a = M_end; b = grad; c = wQE_f; d = wQE;
end
