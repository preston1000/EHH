function [ netExpended ] = generate_neurons_rand(candidate_combinations, net, num_node_to_add)
% GENERATE_NEURONS_RAND generates new neurons randomly
%
% syntax:
% 
% input:
%       net: struct, the existing EHH net
%       candidate_combinations: matrix, possible combination of neurons, each row
%               is a candidate
%       num_node_to_add, scalar
% output:
%       netExpended: struct, new net

    % remove duplication in the candidates
    candidate_combinations = unique(candidate_combinations, 'rows');
    numCandidate = length(candidate_combinations);
    % permutate and choose from the candidates
    rand_comb = randperm(numCandidate);
    num_node_to_add = min( numCandidate , num_node_to_add);
    comb_choose = candidate_combinations(rand_comb(1:num_node_to_add), :);
    comb_choose = sortrows(comb_choose);
    
    B = cell(num_node_to_add, 1);
    stemB = zeros(num_node_to_add, 2);
    id_layer = zeros(num_node_to_add, 1);
    
    for k = 1:num_node_to_add
        n1 = comb_choose(k, 1);
        n2 = comb_choose(k, 2);
        
        B{k} = [net.B{n1}; net.B{n2}];
        stemB(k,:) = sort([n1, n2]); % sort the constitute neurons 
        id_layer(k) = net.id_layer(n1) + net.id_layer(n2);
    end
    
    netExpended = struct('stemB', [net.stemB; stemB], 'id_layer', [net.id_layer ; id_layer], 'nx', net.nx, 'nNode', net.nNode + num_node_to_add);
    netExpended.nLayer = max(netExpended.id_layer);
    netExpended.B = [net.B; B];