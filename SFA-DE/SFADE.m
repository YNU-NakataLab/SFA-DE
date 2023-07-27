classdef SFADE < ALGORITHM
% <multi/many> <real/integer> <expensive>
% Scaralization Function Approximation based MOEA/D
% F     --- 0.5 --- Scale facter for inner DE
% CR    --- 0.9 --- Crossover rate for inner DE
% omega ---  20 --- Number of generations for DE-based search

%------------------------------- Reference --------------------------------
% Y.Horaguchi, K.Nishihara and M. Nakata, Evolutionary Multiobjective 
% Optimization Assisted by Scalarization Function Approximation for 
% High-Dimensional Expensive Problems, TechRxiv, 2023.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2023 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Yuma Horaguchi

    methods
        function main(Algorithm, Problem)
            %% Parameter setting
            [F, CR, omega] = Algorithm.ParameterSet(0.5, 0.9, 20);

            %% Generate the weight vectors
            [W, Problem.N] = UniformPoint(Problem.N, Problem.M);

            %% Detect the neighbours of each solution
            T      = ceil(Problem.N / 10);
            B      = pdist2(W, W);
            [~, B] = sort(B, 2);
            B      = B(:, 1 : T);

            %% Initialize population
            PopDec     = UniformPoint(Problem.N, Problem.D, 'Latin');
            Population = Problem.Evaluation(repmat(Problem.upper - Problem.lower, Problem.N, 1) .* PopDec + repmat(Problem.lower, Problem.N, 1));
            Arc        = Population;
            Z          = min(Population.objs, [], 1);

            %% Optimization
            while Algorithm.NotTerminated(Arc)
                %% Solution-generation
                for i = 1 : Problem.N
                    %% Model Construction
                    [~, UniqueID] = unique(Arc.decs, 'stable', 'rows');
                    ArcUnique     = Arc(UniqueID);
                    ArcTch        = max(abs(ArcUnique.objs - repmat(Z, length(ArcUnique), 1)) .* W(i, :), [], 2);
                    [tr_y, rankA] = sort(ArcTch);
                    tr_x          = ArcUnique(rankA(1 : Problem.N)).decs;
                    tr_y          = tr_y(1 : Problem.N);
                    pair          = pdist2(tr_x, tr_x);
                    D_max         = max(max(pair, [], 2));
                    spread        = D_max * (Problem.D * Problem.N) ^ (-1 / Problem.D);
                    RBFModel      = newrbe(transpose(tr_x), transpose(tr_y), spread);

                    %% DE-based search
                    P        = B(i, randperm(end));
                    PopDEDec = Population(P).decs;
                    PopDEObj = transpose(sim(RBFModel, transpose(PopDEDec)));
                    for gen = 1 : omega
                        [~, BestID] = min(PopDEObj);
                        BestDec     = repmat(PopDEDec(BestID, :), T, 1);
                        indices     = zeros(T, 2);
                        for j = 1 : T
                            indices(j, :) = datasample([1 : j - 1, j + 1 : T], 2, 'Replace', false);
                        end
                        CandidateDec = OperatorDEctb1(Problem, PopDEDec, BestDec, PopDEDec(indices(:, 1), :), PopDEDec(indices(:, 2), :), {CR, F});   % proM=0,disM=0 for SOPs
                        CandidateObj = transpose(sim(RBFModel, transpose(CandidateDec)));
                        Replace      = PopDEObj >= CandidateObj;
                        PopDEDec(Replace, :) = CandidateDec(Replace, :);
                        PopDEObj(Replace, :) = CandidateObj(Replace, :);
                    end
                    [~, BestID] = min(PopDEObj);
                    OffDec      = PopDEDec(BestID, :);
        
                    %% Evaluate offspring
                    Offspring = Problem.Evaluation(OffDec);
        
                    %% Update the reference point
                    Z = min(Z, Offspring.obj);
        
                    %% Update population and archive
                    g_old = max(abs(Population(P).objs - repmat(Z, T, 1)) .* W(P, :), [], 2);
                    g_new = max(repmat(abs(Offspring.obj - Z), T, 1) .* W(P, :), [], 2);
                    Population(P(g_old >= g_new)) = Offspring;
                    Arc   = [Arc, Offspring];
                    
                    %% Check termination criteria
                    Algorithm.NotTerminated(Arc);
                end
            end
        end
    end
end
