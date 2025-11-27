function [M_next, PI_next, wQE] = findNextM_v7_fast(M_in, PI0, X, gamma, pgd_iters, tol, d, sd)
% Faster replacement for findNextM_v6: projected gradient on simplex.
%   M_in      : (1 x d x K) or (d x K); reshaped to (d x K) internally
%   PI0       : (K x N) warm start (optional; can be [])
%   X         : (d x N)
%   gamma     : > 1 (paper's parameter)
%   pgd_iters : max PGD iterations per point (e.g., 100)
%   tol       : stopping tol on l1 change per point (e.g., 1e-5)
%   d, sd     : same as your original signature (sd = sum(X.*X,1)')

    if nargin < 6 || isempty(tol),       tol = 1e-5;      end
    if nargin < 5 || isempty(pgd_iters), pgd_iters = 100; end

    % Shapes
    if ndims(M_in) == 3, M = reshape(M_in, d, []); else, M = M_in; end
    [dchk, K] = size(M);
    if dchk ~= d, error('Dimension mismatch in M.'); end
    N = size(X,2);

    % Precompute pieces common to all points
    mu2   = sum(M.^2, 1).';           % K x 1  (||mu_k||^2)
    baseC = (gamma - 1) * mu2;        % K x 1
    % Safe global Lipschitz for gradient (H = 2 M'M): L = 2 * ||M||_F^2
    L     = 2 * sum(mu2);
    step  = 1 / max(L, 1e-12);

    % Warm start
    if isempty(PI0), PI0 = ones(K, N) / K; end
    PI_next = zeros(K, N);

    % Per-point PGD (parallel)
    parfor n = 1:N
        x  = X(:, n);
        fx = baseC - 2*gamma * (M.' * x);   % K x 1 (drop const (gamma-1)||x||^2)
        p  = PI0(:, n);                      % init on simplex
        y  = p;

        for it = 1:pgd_iters
            a     = M * y;                   % d x 1
            grad  = fx + 2*(M.' * a);        % K x 1
            z     = y - step * grad;
            p_new = proj_simplex(z);

            if norm(p_new - p, 1) <= tol
                % disp('done')
                break;
            end
            % Nesterov momentum
            y = p_new + (it-1)/(it+2) * (p_new - p);
            p = p_new;

            % if it == pgd_iters
            %     disp(norm(p_new - p, 1))
            % end
        end
        PI_next(:, n) = p;
    end



    % ----- Closed-form centroid update (matrix form) -----
    s     = sum(PI_next, 2);          % K x 1
    s     = max(s, eps);              % avoid divide-by-zero
    B     = PI_next * PI_next.';      % K x K
    D     = B ./ s.';                 % <-- column-wise scaling: D_{ij} = B_{ij}/s_j
    CC    = (X * PI_next.') ./ s.';   % d x K; CC(:,k) = (X*pi_k)/s_k
    I     = eye(K);
    M_next = CC / (I + (1/gamma) * (D - I));

    % ----- Objective value (same structure as your code) -----
    % C = ||x||^2 - 2x'M + ||mu||^2, vectorized with 'sd' and 'mu2'
    C    = sd - 2 * (X.' * M) + ones(N,1) * (mu2.');
    alpha = 1/(gamma - 1);
    wQE   = ( trace(PI_next * C) + alpha * norm(X - M * PI_next, 'fro')^2 ) / N;
end

% --- Projection onto the probability simplex {p >= 0, sum(p)=1} ---
function p = proj_simplex(v)
    K    = numel(v);
    u    = sort(v, 'descend');
    cssv = cumsum(u);
    rho  = find(u + (1 - cssv) ./ (1:K)' > 0, 1, 'last');
    theta= (cssv(rho) - 1) / rho;
    p    = max(v - theta, 0);
end
