function [M_next, PI_next, wQE, r_solved] = findNextM_v6_mnist(M, PI, X, gamma, L, beta, d)
    K = size(M,3);
    N = size(X,2);

    M = reshape(M, d, K);

    C = sum(X.*X, 1)' - 2 * X' * M + sum(M.*M, 1);
    alpha = 1/(gamma-1);
    gamma_inv = 1/gamma;
    
    Aeq = ones(1, K);
    beq = 1;

    lb = zeros(K,1);
    ub = ones(K,1);

    % H = (2/(gamma - 1))*(M'*M);
    % H = 2*alpha*(M'*M + 1e-8*eye(K));
    H = 2*(M'*M + 1e-2*eye(K));
    % eigvals = eig(H);
    % fprintf('min λ(H) = %.3e   max λ(H) = %.3e\n',min(eigvals),max(eigvals));
        % 'MaxIter', 3000, ...

    options = optimoptions('quadprog', ...
        'Display','off', ...
        'MaxIter', 3000, ...,
        'Algorithm', 'interior-point-convex', ...
        'ConstraintTolerance', 1e-8, ...
       'OptimalityTolerance', 1e-8, ...
        'StepTolerance', 1e-10);


    PI_next = zeros(K, N);

    % parfor i = 1:N
    for i = 1:N
        xn = X(:, i);
        dists = vecnorm(xn - M, 2, 1).^2;
        % f = dists - (2/(gamma - 1))*(xn'*M);
        f = (gamma - 1)*dists - 2*(xn'*M);

        [pi_col, ~, exitflag] = quadprog(H, f', [], [], Aeq, beq, lb, ub, [], options);

        if exitflag<=0          % -2 = infeasible, 0/-3 = exceeded iters, etc.
            warning('quadprog failed for sample %d (exitflag %d).  Re-projecting.',i,exitflag)
            % fall back to projection onto the simplex
            disp(rank(M'*M));
            disp("===")
            pi_col = max(pi_col,0);
            s      = sum(pi_col);      % should be >0 even if infeasible
            if s==0
                pi_col(:)=1/K;                       % degenerate
            else
                pi_col = pi_col / s;
            end
        end

        PI_next(:, i) = pi_col;

        % PI_next(:, i) = PI_next(:, i)/sum(PI_next(:, i));
    end


    b_temp = Aeq*PI_next;
    am = max(b_temp(1, :));
    pm = min(b_temp(1, :));

    if abs(am - 1) > 1e-2 || abs(pm - 1) > 1e-2
        disp(am)
        disp(pm)
    end

    PiPit = PI_next * PI_next'; %N x N
    Pi = sum(PI_next, 2)'; %1 x N
    
    D = PiPit ./ Pi;
    CC = (X * PI_next') ./ Pi;
    I = eye(K);

    M_next  = CC / (I + gamma_inv * (D-I));

    wQE =  (trace(PI_next*C) + alpha * norm(X-M*PI_next,'fro')^2)/N;
    r_solved = norm(X-M*PI_next,'fro')/sqrt(N)/(gamma - 1);
end




function wQE = wcefff(M, X, gamma, d)
%  M:    d x K data matrix
%  X:    d x N 

    K = size(M,2);
    N = size(X,2);
    
    M = reshape(M, d, K);

    C = sum(X.*X, 1)' - 2 * X' * M + sum(M.*M, 1);
    alpha = 1/(gamma-1);
    gamma_inv = 1/gamma;
    
    %onesK = ones(1,K);
    %onesN = ones(1,N);

    PI = zeros(K, N);

    % Aeq = [ones(1, K); zeros(K-1, K)];
    Aeq = [ones(1, K)];

    % beq = [1; zeros(K-1, 1)];
    beq = 1;
    lb = zeros(K,1);
    H = (2/(gamma - 1))*(M'*M);
    % options = optimoptions('quadprog','Display','off');
    options = optimoptions('quadprog','Display','off', 'MaxIter', 1000);


    for i = 1:N
        xn = X(:, i);
        dists = vecnorm(xn - M, 2, 1).^2;
        f = dists - (2/(gamma - 1))*(xn'*M);
        PI_next(:, i) = quadprog(H, f, [], [], Aeq, beq, lb, [], [], options);
    end
        
    PiPit = PI_next * PI_next'; %N x N
    Pi = sum(PI_next, 2)'; %1 x N
    
    D = PiPit ./ Pi;
    CC = (X * PI_next') ./ Pi;
    I = eye(K);

    M_next  = CC / (I + gamma_inv * (D-I));

    wQE =  (trace(PI_next*C) + alpha * norm(X-M*PI_next,'fro')^2)/N;
end

