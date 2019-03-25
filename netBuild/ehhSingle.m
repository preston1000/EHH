function [netBest, weightsBest, statisticsBest, lambdaBest] = ehhSingle(x, y, parameters)
%ehhSingle runs the forward procedure and the backward pruning once, and
%return the constructed EHH net.
% input
%       x          ---------------- the sample x used for the forward
%                   procedure, with size N x dim, part of the whole samples
%       y          ---------------- the sample y used for the forward
%                   procedure, with size N x 1, part of the whole samples
%       parameters
% output
%       netBest: the generated net, struct with fields
%           B: node information, cell, each is [index, index of x, value of beta]
%           id_layer: node position information, array, the index of layer
%                   that it lies in
%           stemB: N*2 matrix, indicating the indices of previous nodes 
%           nx: length of input
%           nLayer: number of layers
%           nNode: number of nodes
%       weightsBest: the weights of nodes
%       statisticsBest: struct with
%           timeTrain:
%           timeForward: time need for forward procedures
%           timePrune: time need for backward procedures
%           err: 
%           lof
%           stds
%       lambdaBest : best lambda
%


% Parameter initilization
structure_parameter = parameters.structure;  % the parameters for structure definition
start_time = tic;
%% the first layer
net = netInitiation(x, parameters.shares);

id_layer = net.id_layer;
id_var_bb = getIndexOfX( net.B );
%% forward process
num_layer = size(structure_parameter, 2);  % the first layer is not taking into consideration
for layer_index = 2:num_layer+1  % the neurons are added layerwisely
    num_neurons = structure_parameter(layer_index-1);
    % All possible combinations of neurons
    candidate_combinations = []; %the index of id_layer
    layer_index_1 = 1;
    while layer_index_1 < layer_index   %possible combinations yielding layer nl
        layer_index_2 = layer_index-layer_index_1;
        index_in_layer_1 = find(id_layer==layer_index_1);  % in the k1-th layer
        index_in_layer_2 = find(id_layer==layer_index_2);
        [index_in_layer_1, index_in_layer_2] = meshgrid(index_in_layer_1,index_in_layer_2);
        index_in_layer_1 = index_in_layer_1(:);
        index_in_layer_2 = index_in_layer_2(:);
        index_combinations = [];
        for index_combination = 1:length(index_in_layer_1)
            index_x_1 = id_var_bb{index_in_layer_1(index_combination)};
            index_x_2 = id_var_bb{index_in_layer_2(index_combination)};
            if ~isempty( intersect(index_x_1,index_x_2)) % if they have common x_i, they will not be combined
                continue;
            end
            index_combinations = [index_combinations; index_combination];
        end
        candidate_combinations = [candidate_combinations; sort([index_in_layer_1(index_combinations),index_in_layer_2(index_combinations)], 2)];
        layer_index_1 = layer_index_1+1;
    end
    % Choose suitable combinations from all possible combinations (random procedure)   
    net = generate_neurons_rand(candidate_combinations, net, num_neurons);
end
timeForward = toc(start_time);
if checkNodeValidity( net ) || checkDuplicateNode( net ) || checkNodeRelation( net ) % check identical x_i in each node & check identitical nodes
    throw(MException('Error:Train', 'duplication occurs in the training process'))
end %
%% Backward process
lof = 10*norm(y-mean(y))^2;
lambda = parameters.lambda;

netBest = net;
weightsBest = [];
statisticsBest = struct();
lambdaBest = -1;
for k = 1:length(lambda)
    [netPruned, weights, statistics] = prune_node(net, x, y, lambda(k), parameters);

    execute_prune = 'X';
    if statistics.lof < lof
        netBest = netPruned;
        weightsBest = weights;
        statisticsBest = statistics;
        execute_prune = 'ok';
        lambdaBest = lambda(k);
        lof = statistics.lof;
    end
    fprintf('lambda: %2.2f, error: %6.4f, lof: %6.4f, std: %6.4f, prune? %s \n', lambda(k), statistics.err, statistics.lof, statistics.stds, execute_prune);
end

%% The output
time = toc(start_time);
statisticsBest.timeTrain = time;
statisticsBest.timeForward = timeForward;
statisticsBest.timePrune = time - timeForward;
fprintf('Final results: lambda: %2.2f, error: %6.4f, lof: %6.4f, std: %6.4f, ellapsed time: %f \n', lambdaBest, statisticsBest.err, statisticsBest.lof, statisticsBest.stds, time);
