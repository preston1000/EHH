function [netPruned, weights, statistics] = prune_node(net, x, y, lambda, parameters)
%PRUNE_NODE prunes the net 
%
% syntax:
% 
% input:
%       net: struct, the net to be pruned
%       x, y: training sample
%       lambda: 
%       parameters: struct
% output:
%       netPruned: struct, net after prune
%       weights: array, weigths of each nodes, 
%       statistics: struct, lof, err, stds, timePrune

%% parameters
PRECISION = parameters.precision;
%% generate the adjacency matrix A in the net,  where A(i, j) = 1 indicates that node i is connected to node j
adjacency_matrix = stem_to_adjacency( net.stemB );
if checkAdjacentMatrix( adjacency_matrix )
    throw(MException('Error:Train', 'transformation from stem to adjacent matrix is not working'))
end
%% compute the output values of each neuron, using the sample sata
node_values = cal_node_value(net, x);
%% cpmpute the weights of each neuron to the output
lambda = lambda * sqrt(2*log10(net.nNode + 1));   % parameters determined by the number of neurons in the net
weights = lasso(node_values, y, lambda, parameters.rho, parameters.gamma, parameters.quiet); % weights are trained using lasso
weights_of_constant = weights(1);
weights_of_nodes = weights(2:end);
%% delete nodes that have no successors and have no contribute to the output
timeStart = tic;
weightAbs = abs(weights_of_nodes);
num_nodes = net.nNode;
B = net.B;
stemB = net.stemB;
id_layer = net.id_layer;
if sum(weightAbs) > PRECISION
    index_active_node = weightAbs > PRECISION;
    adjacency_matrix(index_active_node, net.nNode + 1) = 1;   % add one column column
    % find useless neurons based on out-fan
    out_fan = sum(adjacency_matrix, 2);
    rem_index = find(out_fan > 0);  % indices of node to be remained
    
    num_node_remained = length(rem_index);
    while num_node_remained < num_nodes
        num_nodes = num_node_remained;
        % check if any two columns in adjacency matrix is the same
        if checkAdjacentMatrix( adjacency_matrix )
            disp('Duplications in the adjacency matrix--2')
        end
        % prune matrix
        adjacency_matrix = adjacency_matrix(rem_index, [rem_index',end]);
        % update network
        if iscell(B)
            B = B(rem_index);
        else
            rem_1=intersect(1:length( B ), rem_index);
            B = B(rem_1, :);
        end
        id_layer = id_layer(rem_index);
        weights_of_nodes = weights_of_nodes(rem_index);
        stemB = zeros(num_node_remained, 2);
        for nn = 1:num_node_remained
            tmp_id = find(adjacency_matrix(:, nn) > 0)';
            if ~isempty(tmp_id)
                if length(tmp_id) ~= 2
                    throw(MException('Error:Trainging', ' when computing adjacency matrix: number of inputs is larger than 2'))
                end
                stemB(nn, :) = tmp_id;
            end
        end
        out_fan = sum(adjacency_matrix, 2);
        rem_index = find(out_fan > 0);  % indices of neurons to be remained
        num_node_remained=length(rem_index);
    end
    
    netPruned = struct('stemB', stemB, 'id_layer', id_layer, 'nx', net.nx);
    netPruned.nNode = length(id_layer);
    netPruned.nLayer = max(netPruned.id_layer);
    netPruned.B = B;
    
    if checkNodeRelation( netPruned )
        throw(MException('Error:Train', 'pruning is not working properly'))
    end
    % calculate the output of each neuron
    node_values = cal_node_value(netPruned, x);
    weights = [weights_of_constant; weights_of_nodes];
    % statistics
    hat_y = node_values * weights;
    err = norm(hat_y - y)^2 / norm(y - mean(y))^2;
    stds =  std(hat_y - y);
    lof = err*norm(y - mean(y))^2 / ( 1 - (  num_nodes + 1 + parameters.penalty * num_nodes ) / size(x, 1) )^2;
else % if all neurons contribute nothing to the output, then delete all nodes
    lof = 10;
    err = norm(y)^2 / norm(y - mean(y))^2;
    stds =  std(y);
    netPruned = net;
end
statistics = struct('err', err, 'stds', stds, 'lof', lof, 'timePrune', toc(timeStart));
