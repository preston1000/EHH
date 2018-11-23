function [ adjacency, layer_index, B_new ] = merge_net( adjacency_matrices, layer_indices, Bs, weights_all )
%MERGE_NET 将生成的多个分片线性网络合并在一起
%   此处显示详细说明

%% 检查数据
if ~iscell(adjacency_matrices) || ~iscell(layer_indices) || ~iscell(weights_all) || ~iscell(Bs)
    fprintf('input arguments should be cells.\n')
end

num_nets = length(adjacency_matrices);
if length(layer_indices) ~= num_nets || length(weights_all) ~= num_nets || length(Bs) ~= num_nets
    fprintf('input arguments are not of the same dimension.\n')
end

%% 获取每个子网络的数据
num_nodes_separate = zeros(num_nets, 1); % 每个网络中节点个数
for i = 1:num_nets
    num_nodes_separate(i) = size(adjacency_matrices{i}, 1);
end
num_nodes = sum(num_nodes_separate);
num_accumulated = cumsum(num_nodes_separate);

containing_x_in_node = cell(num_nets, 1); % containing x in each node of each net
for i = 1:num_nets
    tmp_containing = cell(num_nodes_separate(i), 1);
    for j = 1:num_nodes_separate(i)
        tmp_containing{j} = Bs{i}{j}(:, 2);
    end
    containing_x_in_node{i} = tmp_containing;
end

corresponding_ids = [zeros(num_nodes, 3) (1:num_nodes)' zeros(num_nodes, 1)]; % [net id， node id， layer index, sequential ids, id after transformation]
for i = 1:num_nets
    if i == 1
        range = 1:num_accumulated(i);
    else
        range = (num_accumulated(i-1) + 1):num_accumulated(i);
    end
    corresponding_ids(range, 1) = i; % to set the net id
    corresponding_ids(range, 2) = (1:num_nodes_separate(i) )'; % to set the node id in each net
    
    belonging_layer = zeros(num_nodes_separate(i), 1);
    for j = 1:num_nodes_separate(i)
        belonging_layer(j) = length(containing_x_in_node{i}{j});
    end
    corresponding_ids(range, 3) = belonging_layer; % to set the layer index where a node lies
end
%% 初步合并所有网络
adjacency = sparse(num_nets);
for i = 1:num_nets
    if i == 1
        range = 1:num_accumulated(i);
    else
        range = (num_accumulated(i-1) + 1):num_accumulated(i);
    end
    adjacency(range, range) = adjacency_matrices{i};
end

%%  rearrange nodes so that the adjacent nodes are of the same layer, except the nodes on the boundary
num_layers = max(corresponding_ids(:,3)); % number of layers
num_nodes_layerwise = zeros(num_layers, 1); % number of nodes in each layer
start = 0;
for i = 1:num_layers
    range = find(corresponding_ids(:,3) == i); 
    num_nodes_layerwise(i) = length(range);
    corresponding_ids(range, 5) = start + (1:num_nodes_layerwise(i))';
    start = start + num_nodes_layerwise(i);
end

%% find duplicate nodes, layerwise
index_to_be_deleted = [];  % index in the fourth column of corresponding_ids
for i = 1:num_layers
    range = find(corresponding_ids(:,3) == i); 
    reference = zeros(num_nodes_layerwise(i), i*2); % each row is the indices of x in each node & the values of beta in each node(sorted)
    for j = 1:num_nodes_layerwise(i) % find corresponding x and beta
        tmp_B = Bs{corresponding_ids(range(j), 1)};
        tmp_matrix_B = tmp_B{corresponding_ids(range(j), 2)};
        reference(j, :) = [sort(tmp_matrix_B(:, 2))'   sort(tmp_matrix_B(:, 3))'];
    end
    unique_reference = zeros(size(reference));
    correspondence = zeros(num_nodes_layerwise(i), 2);
    counter = 1;
    for j = 1:num_nodes_layerwise(i)
        [C, ia, ~] = intersect(unique_reference, reference(j, :), 'rows');
        if isempty(C)
            unique_reference(counter, :) = reference(j, :);
            counter = counter + 1;
            correspondence(j, :) = [j, j];
        else
            correspondence(j, :) = [j, correspondence(ia, 2)];
            index_to_be_deleted = [index_to_be_deleted; range(j)];
        end
    end
    if counter == 1 % no duplicated nodes
        continue
    end
    correspondence = correspondence(1:(counter - 1), :);
    corresponding_ids(range(correspondence(:, 1)), 5) = correspondence(:, 2);
end
 %% delete duplicated nodes in adjacency matrix
 if isempty(index_to_be_deleted)
     fprintf('All nodes are unique.\n')
     return
 end
 for i = 1:length(index_to_be_deleted)
     indices = find(adjacency(:, index_to_be_deleted(i)));
     adjacency(indices, corresponding_ids(indices, 5)) = 1;
     
     indices = find(adjacency(index_to_be_deleted(i), :));
     adjacency(corresponding_ids(indices, 5), indices) = 1;
 end
 
 for i = 1:length(index_to_be_deleted)
     if sum(adjacency(:,  index_to_be_deleted(i))) > 0 || sum(adjacency( index_to_be_deleted(i), :)) > 0
         fprintf('转换错误，删除后还是有连接。\n')
     end
 end
 % generate net parameters
adjacency(index_to_be_deleted, :) = [];
adjacency(:, index_to_be_deleted) = [];
layer_index = corresponding_ids(:, 3);
layer_index(index_to_be_deleted) = [];

index_remained = setdiff(corresponding_ids(:, 4), index_to_be_deleted);
num_remained = length(index_remained);
B_new = cell(num_remained, 1);
for i = 1:num_remained
    B_new{i} = Bs{corresponding_ids(index_remained(i), 1)}{corresponding_ids(index_remained(i), 2)};
end

