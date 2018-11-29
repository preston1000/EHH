function [LAYERS,  WEIGHTS, stats]=prune_node(layers, x, y, parameters)
%prune_node 后向剪枝过程 
%

%% parameters
% lasso parameters
gamma = parameters.gamma;
rho = parameters.rho;
precision = parameters.precision;
quiet = parameters.quiet;
LAMBDA = parameters.lambda;
lof = parameters.lof;

%% 计算网络的连接矩阵， 第ij个元素表示有连接i->j
num_nodes = size(layers,1);
%% 由训练数据，计算每个节点的值，
LAYERS = layers;
WEIGHTS = [];
err = 0;
stds = 0;
for ii = 1:length(LAMBDA)
    adjacency_matrix = stem_to_adjacency( cell2mat(layers(:, 3)) );
    node_values = cal_node_value(layers, x);
    % 计算每个节点到输出层的权重
    lambda = LAMBDA(ii) * sqrt(2*log10(num_nodes + 1));   % 由网络中所有节点个数决定
    weights = lasso(node_values, y, lambda, rho, gamma, quiet); % 这里是训练每个节点到输出节点的权重
    weights_of_constant = weights(1);
    weights_of_nodes = weights(2:end);

    % 删除节点
    if sum(abs(weights_of_nodes)) > precision
        index_active_node = abs(weights_of_nodes) > precision;
        adjacency_matrix(index_active_node, num_nodes + 1) = 1;   % the last column is the weights to output
        % 先删除出度==0的节点
        out_fan = sum(adjacency_matrix, 2);
        rem_index = find(out_fan > precision);  % 要保留的节点编号

        num_node_remained = length(rem_index);
        while num_node_remained < num_nodes
            num_nodes = num_node_remained;
            adjacency_matrix = adjacency_matrix(rem_index, [rem_index',end]);
            % 更新网络参数
            layers = layers(rem_index, :);
            weights_of_nodes = weights_of_nodes(rem_index);

            out_fan = sum(adjacency_matrix, 2);
            rem_index = find(out_fan > 0);  % 要保留的节点编号
            num_node_remained=length(rem_index);
        end
        % 重新确定节点输出值
        stem = adjacency_to_stem( adjacency_matrix(:, 1:end-1) );
        layers(:, end) = num2cell(stem, 2);

        weights = [weights_of_constant; weights_of_nodes];

    else % do not delete any node

    end
    
    [ errk, stdsk, lofk ] = statistics( layers, weights, x, y, parameters );
        
    execute_prune = 'X';
    if lofk < lof
        execute_prune = 'ok';
        LAYERS = layers;
        WEIGHTS = weights;
        lof = lofk;
        err = errk;
        stds = stdsk;
    end
    fprintf('lambda: %2.2f, error: %6.4f, lof: %6.4f, std: %6.4f, prune? %s \n', LAMBDA(ii), errk, lofk, stdsk, execute_prune);

end
stats = struct('err', err, 'stds', stds, 'lof', lof);