% File: mnist_outlier_trials.m
% -----------------------------------------------------------
% clear; clc;

%% -------- user-editable paths -----------------------------
matPath   = "datasets/nips.mat";
resultDir = "datasets/results";
if ~exist(resultDir,"dir"); mkdir(resultDir); end
%% ----------------------------------------------------------

% Results

% ==== Outlier-Recall  (mean ± std over 4 trials) ====
% kList      = [10 20 30];    % same three as the NeurIPS paper
% numTrials  = 4;            % average over 10 random seeds
% factor      = 4;           % corruption scale (≈ KM recall 0.95)
% dim        = 50;            % PCA dimensionality
% 5%, gamma = 1.01, 1000
% k = 10 |  KM  0.9450 ± 0.0342   DR-KM  1.0000 ± 0.0000 TKM - 97
% k = 20 |  KM  0.7900 ± 0.0622   DR-KM  1.0000 ± 0.0000 TKM - 96
% k = 30 |  KM  0.5650 ± 0.0526   DR-KM  1.0000 ± 0.0000 TKM - 91


% ==== Outlier-Recall  (mean ± std over 5 trials) ====
% kList      = [10 20 30];    % same three as the NeurIPS paper
% numTrials  = 5;            % average over 10 random seeds
% factor      = 5;           % corruption scale (≈ KM recall 0.95)
% dim        = 50;            % PCA dimensionality
% rngOffset  = 2025;          % first seed will be 2026, etc.
% nTrain = 800;
% f = 0.05; % ouliers
% 
% k = 10 |  KM  0.8450 ± 0.0447   DR-KM  1.0000 ± 0.0000 TKM - 100
% k = 20 |  KM  0.6050 ± 0.0326   DR-KM  1.0000 ± 0.0000 TKM - 97
% k = 30 |  KM  0.3300 ± 0.0447   DR-KM  0.9750 ± 0.0000 TKM - 85

%% --------------- experiment hyper-params -------------------
kList      = [10 20];    % same three as the NeurIPS paper
numTrials  = 1;            % average over 10 random seeds
factor      = 5;           % corruption scale (≈ KM recall 0.95)
dim        = 50;            % PCA dimensionality
rngOffset  = 2025;          % first seed will be 2026, etc.
nTrain = 800;
f = 0.05; % ouliers


%% -----------------------------------------------------------
%% 1. Load NIPS and build 50-D doc vectors exactly like Python
load(matPath,"counts");                    % sparse 11 463 × 5 811
Xsub = full(double(counts)).';        % 1000 × 11 463  (docs × words)

% rng(rngOffset);                            % seed 2025
% dim  = 50;                               % truncated SVD dimension
% [U,S,~] = svds(Xsub, dim);               % ***UN-centred*** SVD (like sklearn.TruncatedSVD)
% data50  = (U * S)';                      % 50 × 1000  (same scaling as Python)
% load("nips_doc50.mat", "data50");

dimPCA = 50;
% tic;
%% --- compute (once) or load (next time) the 50-PC basis ----------
pcaFile = "/Users/vikrant/Library/Mobile Documents/com~apple~CloudDocs/template/datasets/nips_coeff50.mat";

if exist(pcaFile,"file")
    load(pcaFile,"coeff");          % <-- fast path: just grab the matrix
else
    coeff = pca(Xsub','NumComponents',dimPCA,'Centered',false);
    save(pcaFile,"coeff","-v7.3");  % <-- one-time save
end

% toc;
data50_pca = coeff' * Xsub;                % 40×60000

% disp(size(data50))
% data50 = data50';
Ntrain = nTrain;
z      = ceil(f * Ntrain);           % 25 outliers

%%
idxSel  = randperm(size(data50_pca,2), nTrain);      % pick 1 000 docs

data50 = full(double(data50_pca(:, idxSel)));        % 1000 × 11 463  (docs × words)

%% 2.  Storage for results -----------------------------------
rec_km = zeros(numTrials, numel(kList));
rec_dr = zeros(numTrials, numel(kList));
rec_tkm = zeros(numTrials, numel(kList));

%% 3.  Trial loop -------------------------------------------
for t = 1:numTrials
    seed = rngOffset + t;
    % rng(seed);

    outIdx             = randperm(Ntrain, z);
    X                  = data50;                     % fresh copy

    % -------- NeurIPS uniform noise (factor = 4) -----------------
    gMin   = min(data50(:)) * factor;        % factor = 2 or 4
    gMax   = max(data50(:)) * factor;
    
    noiseMat = gMin + (gMax - gMin) .* rand(dim, z, 'like', data50);  % 50 × z
    signFlip = randi([0 1], 1, z, 'like', data50);  signFlip(signFlip==0) = -1;
    
    X(:, outIdx) = X(:, outIdx) + noiseMat .* signFlip;  % overwrite z points
    trueOut            = false(1,Ntrain); trueOut(outIdx)=true;

    for kIdx = 1:numel(kList)
        k = kList(kIdx);
        disp(k)

        % ---------- baseline k-means++ ----------------------
        tic;
        [~, C] = kmeans(X', k, 'Start','plus','MaxIter',1e3);
        dists  = pdist2(X', C, 'squaredeuclidean');
        minD   = min(dists,[],2);
        [~,w]  = maxk(minD, z);
        pred   = false(1,Ntrain);
        pred(w)=true;
        rec_km(t,kIdx) = nnz(pred & trueOut)/z;
        toc;

        disp('===============kmeans++=====================')

        % ---------- DR-K-Means ------------------------------
        tic;
        mu_p = 1.01; r = 3; d = dim;
        [m,~,~,~] = drlm(mu_p, C', r, d, X, Ntrain);   % your routine
        % m = C';
        d2     = pdist2(X', m', 'squaredeuclidean');
        minD2  = min(d2,[],2);
        [~,w2] = maxk(minD2, z);
        pred2  = false(1,Ntrain); pred2(w2)=true;
        rec_dr(t,kIdx) = nnz(pred2 & trueOut)/z;
        toc;
        disp('===============DR - kmeans++=====================')

    end
    fprintf("trial %2d/%d done (seed %d)\n", t, numTrials, seed);
end

%% 4.  Summary ----------------------------------------------
% fprintf("\n==== Outlier-Recall  (mean ± std over %d trials) ====\n", numTrials);
% for kIdx = 1:numel(kList)
%     k = kList(kIdx);
%     fprintf("k = %2d |  KM  %.4f ± %.4f   DR-KM  %.4f ± %.4f\n", ...
%             k,  mean(rec_km(:,kIdx)),  std(rec_km(:,kIdx)), ...
%             mean(rec_dr(:,kIdx)),  std(rec_dr(:,kIdx)));
% 
% end
% 
% fprintf("\n==== Outlier-Recall  (max  /  avg  /  min over %d trials) ====\n", numTrials);

% for kIdx = 1:numel(kList)
%     k        = kList(kIdx);
% 
%     km_vals  = rec_km(:,kIdx);   % all trial recalls for k-means++
%     dr_vals  = rec_dr(:,kIdx);   % all trial recalls for DR-K-Means
% 
%     fprintf("k = %2d | KM-++   %.4f / %.4f / %.4f   DR-KM   %.4f / %.4f / %.4f\n", ...
%             k,  max(km_vals),  mean(km_vals),  min(km_vals), ...
%                max(dr_vals),  mean(dr_vals),  min(dr_vals));
% end


function [a, b, c, d] = drlm(gamma, M0, r, d, X, N)
    ITER_MAX = 100;
    REL_ERR_TOL = 1E-3;
    K = size(M0,2);

    pd = makedist('Normal');


    M = zeros(ITER_MAX, d, K);
    wQE = zeros(ITER_MAX-1,1);
    M(1,:, :) = M0;
    
    % X = random(pd, d, N);
    L = 5000;
    beta = 1e-5;
    PI = ones(K,N)/K;
    wmin = Inf;
    
    tic;
    for t = 1:ITER_MAX-1
        % indices = randperm(N, round(0.95 * N)); % Randomly select 90% of the indices
        % shuffledData = X(:, indices); % Select and shuffle the data

        % t
        % [M_next, PI, wQE_curr] = findNextM_v5(M(t, :, :), PI, X , gamma, L, beta, d);

        % [M_next, PI_next, gamma_next, wQE_curr] = findNextM_v5_withgamma(M(t, :, :), PI, X, r, L, beta, d, gamma);
        % gamma = gamma_next;
        % PI = PI_next;

        % [M_next, PI, wQE_curr] = findNextM_v4(M(t, :, :), X, gamma, d);

        [M_next, PI, wQE_curr] = findNextM_v6(M(t, :, :), PI, X, gamma, L, beta, d);

        % [M_next, PI, wQE_curr] = findNextM_v7(M(t, :, :), X, gamma, d);

        % [M_next, PI, wQE_curr] = findNextM_v8(M(t, :, :), X, gamma, d);

        % [M_next, PI, wQE_curr] = findNextM_v10(M(t, :, :), X, gamma, d);        
        % [M_next, PI, wQE_curr] = findNextM_v11(M(t, :, :), PI, X, gamma, L, d);        

        M(t+1,:, :) = M_next;
        wQE(t) = wQE_curr;
        % wQE(t) = wQE_curr;
        % wQE_curr


        if t > 1 && abs((wQE(t) - wQE(t -1))/wQE(t -1)) < 1e-2
        % if t > 1 && abs((wQE(t) - wQE(t -1))) < 1e-5

            M(end, :, :) = M_next;
            % disp('converged')
            wQE_f = wQE(t);
            break;
        end

        if t == ITER_MAX - 1
            wQE_f = wQE(t);
           disp('not converged')
        end


    % grad = r^2 - ((1/(gamma-1))^2 * norm(X-M_next*PI,'fro')^2)/N;

    end
    toc;
    alpha = 1/(gamma-1);
    
    M_end = reshape(M(end, :, :), d, K);
    % % grad = r^2 - (alpha^2 * norm(X-M_end*PI,'fro')^2)/N;


grad = -1;
    
    a = M_end; b = grad; c = wQE_f; d = wQE;
    % a = Mf; b = grad; c = wQE_f; d = wQE;

end
