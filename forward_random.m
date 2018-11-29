function layers=forward_random(first_layer, parameters)
%
% input
%       x          ---------------- the sample x used for the forward
%                   procedure, with size N x dim, part of the whole samples
%       y          ---------------- the sample y used for the forward
%                   procedure, with size N x 1, part of the whole samples
%       x2        ---------------- 
%       y2        ------- 
% output
%       BBf      --- the basis function evaluated at the points
%       Bf       --- the parameters of the basis function
%       coe      --- the coefficient matrix, dim x 1
%--forward growing of the network-----
%--random strategy---

% configuration
structure_parameter = parameters.structure;  % the parameters for structure definition

% iteration to generate new layers
layers = first_layer;
num_layers = size(structure_parameter, 2);  % the first layer is not taking into consideration
for layer_index = 2:num_layers+1  % the neurons are added layerwisely
    % preparation
    id_layer = layers(:, 2);
    id_layer = cell2mat(id_layer);
    num_neurons = structure_parameter(layer_index-1); % #nodes in layer #layer_index
    % find all possible combinations 
    candidate_combinations = []; %存储的是id_layer的下标
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
            index_x_1 = layers{index_in_layer_1(index_combination), 1}(:, 2);
            index_x_2 = layers{index_in_layer_2(index_combination), 1}(:, 2);
            if ~isempty( intersect(index_x_1,index_x_2)) % 有共同的x_i，就不进行组合了
                continue;
            end
            index_combinations = [index_combinations; index_combination];
        end
        candidate_combinations = [candidate_combinations; index_in_layer_1(index_combinations),index_in_layer_2(index_combinations)];
        layer_index_1 = layer_index_1+1;
    end
    % select from candidate combinations to form the new layer
    layer = generate_neurons_rand(candidate_combinations, num_neurons, layers);
    layers = [layers; layer];
end


