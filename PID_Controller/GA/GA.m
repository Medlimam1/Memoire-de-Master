function [z_iter, z_final, pos_final] = GA(popSize, nVar, maxIter, crossRate, mutRate, VarMax, VarMin, objfun)
% GA  Genetic Algorithm for PID tuning using IAE objective
%
% Signature:
%   [z_iter, z_final, pos_final] = GA( ...
%       popSize, nVar, maxIter, crossRate, mutRate, VarMax, VarMin, objfun)
%
% Inputs:
%   popSize   – number of individuals per generation
%   nVar      – number of variables (3 for PID: [Kp,Ki,Kd])
%   maxIter   – number of generations
%   crossRate – crossover probability
%   mutRate   – mutation probability per gene
%   VarMax    – 1×nVar vector of upper bounds
%   VarMin    – 1×nVar vector of lower bounds
%   objfun    – function handle to the IAE objective (e.g. @tracklsq)
%
% Outputs:
%   z_iter    – maxIter×1 vector of best IAE at each generation
%   z_final   – best IAE found over all generations
%   pos_final – 1×nVar best PID gains [Kp, Ki, Kd]

    %--- Initialize population within [VarMin, VarMax] ---
    pop  = repmat(VarMin, popSize, 1) + rand(popSize, nVar) .* (repmat((VarMax - VarMin), popSize, 1));
    cost = zeros(popSize,1);

    %--- Evaluate initial population ---
    for i = 1:popSize
        cost(i) = objfun(pop(i,:));
    end

    %--- Record initial best ---
    [bestCost, idx] = min(cost);
    bestSol = pop(idx, :);

    z_iter = zeros(maxIter,1);

    %--- Evolution loop ---
    for gen = 1:maxIter
        newpop = zeros(popSize, nVar);

        % Elitism: carry over best individual
        newpop(1, :) = bestSol;

        % Create rest of new generation
        for j = 2:popSize
            % ---- Tournament selection for parent1 ----
            i1 = randi(popSize); i2 = randi(popSize);
            if cost(i1) < cost(i2)
                parent1 = pop(i1,:);
            else
                parent1 = pop(i2,:);
            end

            % ---- Tournament selection for parent2 ----
            i1 = randi(popSize); i2 = randi(popSize);
            if cost(i1) < cost(i2)
                parent2 = pop(i1,:);
            else
                parent2 = pop(i2,:);
            end

            % ---- Crossover ----
            if rand < crossRate
                alpha = rand(1, nVar);
                child = alpha .* parent1 + (1 - alpha) .* parent2;
            else
                child = parent1;
            end

            % ---- Mutation ----
            for k = 1:nVar
                if rand < mutRate
                    child(k) = child(k) + 0.1 * (VarMax(k) - VarMin(k)) * randn;
                end
            end

            % ---- Enforce bounds ----
            child = max(child, VarMin);
            child = min(child, VarMax);

            newpop(j, :) = child;
        end

        %--- Replace old population and re-evaluate costs ---
        pop = newpop;
        for j = 1:popSize
            cost(j) = objfun(pop(j,:));
        end

        %--- Update global best ---
        [bestCost, idx] = min(cost);
        bestSol = pop(idx, :);

        z_iter(gen) = bestCost;
        fprintf('GA Generation %2d: Best IAE = %.6f\n', gen, bestCost);
    end

    %--- Outputs ---
    z_final   = bestCost;
    pos_final = bestSol;
end
