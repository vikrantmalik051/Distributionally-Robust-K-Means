% File: gmm_classif_trials.m
% -----------------------------------------------------------
% Synthetic GMM classification: k-means++ vs DR-K-Means (your drlm)
% Uses majority-vote labeling of centroids from TRAIN set,
% then evaluates classification accuracy on a fresh TEST set.

% clear; clc;

%% -----------------------------------------------------------

% ==== Experiment hyper-params ===============================
K            = 200;        % <-- true number of mixture components (= #classes)
d            = 200;       % <-- dimensionality
Ntrain       = 100000;     % <-- #train points
Ntest        = 80000;     % <-- #test points (independent draw)

% Mixture geometry (tune to change separability)
clusterStd   = 3.0;       % isotropic within-cluster std
separation   = 5.0;       % centroid scale; larger => more separated

% Labeling & evaluation
kList        = [K];       % k to run for kmeans/DR-KM (default: K)
numTrials    = 3;         % #independent mixture draws/evals
rngOffset    = 2025;      % base seed

% DR-K-Means hyper-params (same spirit as your previous runs)
mu_p         = 1.04;      % your gamma (>= 1)
r            = 3;         % your parameter r

% K-Means options
useParallel  = false;     % set true if you have Parallel Toolbox
maxIterKM    = 1e3;

% Print compact progress
fprintf('==== GMM classification trials ====\n');
fprintf('K=%d, d=%d, Ntrain=%d, Ntest=%d | sep=%.2f, std=%.2f\n', ...
        K, d, Ntrain, Ntest, separation, clusterStd);

%% --------------- storage for results -----------------------
acc_km  = zeros(numTrials, numel(kList));
acc_dr  = zeros(numTrials, numel(kList));

%% ================== TRIAL LOOP ==============================
for t = 1:numTrials
    seed = rngOffset + t;
    rng(seed);

    %% (A) Draw one GMM; sample TRAIN & TEST from the same mixture
    % Uniform component weights (change if you want non-uniform)
    weights = ones(1, K) / K;

    % Random centroids (each column is a centroid)
    MU = separation * randn(d, K);

    % Isotropic covariances
    Sigmas = zeros(d, d, K);
    for j = 1:K
        Sigmas(:, :, j) = (clusterStd^2) * eye(d);
    end

    % ---- TRAIN set ----
    [Xtrain, Ytrain] = sample_gmm(MU, Sigmas, weights, Ntrain);  % d×Ntrain, 1×Ntrain

    % ---- TEST set ----
    [Xtest,  Ytest ] = sample_gmm(MU, Sigmas, weights, Ntest );  % d×Ntest , 1×Ntest

    %% (B) For each k, fit & evaluate
    for kIdx = 1:numel(kList)
        k = kList(kIdx);
        fprintf('\n[trial %d/%d]  k = %d\n', t, numTrials, k);

        % ---------- baseline k-means++ ----------------------
        opts = statset('MaxIter', maxIterKM);
        if useParallel
            opts = statset(opts, 'UseParallel', true);
        end
        tic;
        [idxKM, Ckm] = kmeans(Xtrain', k, 'Start','plus', 'MaxIter', maxIterKM, 'Options', opts);
        t_km = toc;
        fprintf('k-means++ trained in %.2fs\n', t_km);

        % Majority label per centroid (from TRAIN)
        labelKM = majority_labels_from_idx(idxKM, Ytrain, k, Xtrain, Ckm);

        % Predict TEST labels via nearest centroid
        YhatKM = predictLabels(Xtest, Ckm, labelKM);
        % acc_km(t, kIdx) = mean(YhatKM == Ytest);
        acc_km(t, kIdx) = mean( YhatKM(:) == Ytest(:) );
        fprintf('  acc(k-means++): %.4f\n', acc_km(t, kIdx));

        % ---------- DR-K-Means (your routine) ---------------
        % Initialize with k-means++ centroids
        tic;
        d2  = d; %#ok<NASGU> % keep naming compatibility
        sd  = sum(Xtrain.*Xtrain, 1)';  % for findNextM_v7_fast speed-up

        % drlm returns centers as d×k (named 'm' below)
        [m, ~, ~, ~] = drlm(mu_p, Ckm', r, d, Xtrain, Ntrain);

        t_dr = toc;
        fprintf('DR-K-Means trained in %.2fs\n', t_dr);

        % Assign training points to DR centers to compute majority labels
        [~, idxDR] = min(pdist2(Xtrain', m', 'squaredeuclidean'), [], 2);
        labelDR    = majority_labels_from_idx(idxDR, Ytrain, size(m,2), Xtrain, m');

        % Predict TEST labels and measure accuracy
        YhatDR = predictLabels(Xtest, m', labelDR);
        % acc_dr(t, kIdx) = mean(YhatDR == Ytest);
        acc_dr(t, kIdx) = mean( YhatDR(:) == Ytest(:) );
        fprintf('  acc(DR-K-Means): %.4f\n', acc_dr(t, kIdx));
    end

    fprintf('trial %d/%d done (seed %d)\n', t, numTrials, seed);
end

%% --------------- Summary prints -----------------------------
fprintf('\n==== Classification Accuracy (mean ± std over %d trials) ====\n', numTrials);
for kIdx = 1:numel(kList)
    k = kList(kIdx);
    fprintf('k = %3d | KM++  %.4f ± %.4f   DR-KM  %.4f ± %.4f\n', ...
        k,  mean(acc_km(:,kIdx)),  std(acc_km(:,kIdx)), ...
           mean(acc_dr(:,kIdx)),  std(acc_dr(:,kIdx)));
end

fprintf('\n==== Classification Accuracy (max / avg / min over %d trials) ====\n', numTrials);
for kIdx = 1:numel(kList)
    k = kList(kIdx);
    km_vals  = acc_km(:,kIdx);
    dr_vals  = acc_dr(:,kIdx);
    fprintf('k = %3d | KM++   %.4f / %.4f / %.4f   DR-KM   %.4f / %.4f / %.4f\n', ...
            k,  max(km_vals),  mean(km_vals),  min(km_vals), ...
               max(dr_vals),  mean(dr_vals),  min(dr_vals));
end

% (Optional) confusion matrix on the last trial if K is modest
% if K <= 30
%     figure; confusionchart(Ytest, YhatDR); title('DR-K-Means confusion (test set)');
% end

%% ================= helper functions =========================

function [X, zComp] = sample_gmm(MU, Sigmas, weights, N)
    % MU: d×K, Sigmas: d×d×K, weights: 1×K, N: scalar
    [d, K] = size(MU);
    zComp  = randsample(K, N, true, weights).';   % 1×N
    X      = zeros(d, N);
    for j = 1:K
        idx = (zComp == j);
        Nj  = nnz(idx);
        if Nj > 0
            X(:, idx) = mvnrnd(MU(:, j).', Sigmas(:, :, j), Nj).';
        end
    end
end

function labVec = majority_labels_from_idx(idxTrain, Ytrain, k, Xtrain, C)
    % Robust majority labeling. If a centroid received no TRAIN points,
    % fall back to nearest training point's label.
    labVec = zeros(1, k);
    for j = 1:k
        mask = (idxTrain == j);
        if any(mask)
            % majority vote among true labels of points assigned to centroid j
            labVec(j) = mode(Ytrain(mask));
        else
            % empty cluster fallback: label of nearest training point
            dj = pdist2(C(j, :), Xtrain', 'squaredeuclidean'); % 1×N
            [~, ii] = min(dj);
            labVec(j) = Ytrain(ii);
        end
    end
end

function yhat = predictLabels(Xtest_dN, centroids_kd, labVec_1k)
    % Xtest_dN:  d×Nt
    % centroids_kd: k×d
    % labVec_1k: 1×k integer labels
    D            = pdist2(Xtest_dN', centroids_kd, 'squaredeuclidean'); % Nt×k
    [~, idxNear] = min(D, [], 2);         % Nt×1
    yhat         = labVec_1k(idxNear)';   % 1×Nt
end

% ================== your function (unchanged) ==================
function [a, b, c, d] = drlm(gamma, M0, r, d, X, N)
    % NOTE: This version expects findNextM_v7_fast(.) on your path.
    ITER_MAX = 100;
    REL_ERR_TOL = 1E-3; %#ok<NASGU>
    K = size(M0,2);

    M   = zeros(ITER_MAX, d, K);
    wQE = zeros(ITER_MAX-1,1);
    M(1,:, :) = M0;

    L   = 50;
    beta = 1e-3;
    PI  = ones(K,N)/K; %#ok<NASGU>
    sd  = sum(X.*X, 1)';   % precompute ||x||^2 for speed in v7_fast

    tic;
    for t = 1:ITER_MAX-1
        [M_next, PI, wQE_curr] = findNextM_v7_fast(M(t, :, :), PI, X, gamma, L, beta, d, sd);
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

    M_end = reshape(M(end, :, :), d, K);
    grad  = -1; 

    a = M_end; b = grad; c = wQE_f; d = wQE;
end
