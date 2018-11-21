function [B, BB, stem_B, adjacency_matrix, id_layer, id_var_bb, weights, lof, err, stds]=prune_node(B, stem_B, id_layer, id_var_bb, x, y, lambda, parameters)
% 后向剪枝过程 
%

%% parameters
%  complexity penalty
penalty = parameters.penalty;
% lasso parameters
gamma = parameters.gamma;
rho = parameters.rho;
precision = parameters.precision;
quiet = parameters.quiet;

%% 计算网络的连接矩阵， 第ij个元素表示有连接i->j
num_nodes = size(stem_B,1);

num_connection = nnz(stem_B);
row_indices = zeros(num_connection, 1);
col_indices = zeros(num_connection, 1);
counter = 1;
for kk=1:num_nodes
    for jj = 1:2
        vkk=stem_B(kk,jj);
        if vkk > 0
            row_indices(counter) =  vkk;
            col_indices(counter) =  kk;
            counter = counter + 1;
        end
    end
end
adjacency_matrix = sparse(row_indices, col_indices, 1, num_nodes, num_nodes+1);
%% 由训练数据，计算每个节点的值，
node_values = cal_node_value(B, stem_B, x);

%% 计算每个节点到输出层的权重
lambda = lambda * sqrt(2*log10(num_nodes + 1));   % 由网络中所有节点个数决定
weights = lasso(node_values, y, lambda, rho, gamma, quiet); % 这里是训练每个节点到输出节点的权重
weights_of_constant = weights(1);
weights_of_nodes = weights(2:end);

%% 删除节点
if sum(abs(weights_of_nodes)) > precision
    index_active_node = abs(weights_of_nodes) > precision;
    adjacency_matrix(index_active_node, num_nodes + 1) = 1;   % the last column
    % 先删除出度==0的节点
    out_fan = sum(adjacency_matrix, 2);
    rem_index = find(out_fan > 0);  % 要保留的节点编号
    
    num_node_remained = length(rem_index);
    while num_node_remained < num_nodes
        num_nodes = num_node_remained;
        adjacency_matrix = adjacency_matrix(rem_index, [rem_index',end]);
        % 更新网络参数
        B = B(rem_index);
        id_layer = id_layer(rem_index);
        id_var_bb = id_var_bb(rem_index);
        weights_of_nodes = weights_of_nodes(rem_index);
        stem_B = zeros(num_node_remained, 2);
        for nn = 1:num_node_remained
            tmp_id = find(adjacency_matrix(:,nn) > 0)';
            if ~isempty(tmp_id)
                if length(tmp_id) ~= 2
                    fprintf('连接矩阵计算错误，输入节点数不为2\n')
                    quit()
                end
                stem_B(nn, :) = tmp_id;
            end
        end
        
        out_fan = sum(adjacency_matrix, 2);
        rem_index = find(out_fan > 0);  % 要保留的节点编号
        num_node_remained=length(rem_index);
    end
    % 重新确定节点输出值
    node_values = cal_node_value(B, stem_B, x);
    weights = [weights_of_constant; weights_of_nodes];
    BB = node_values(:,2:end);
    
    hat_y = node_values * weights;
    err = norm(hat_y - y)^2 / norm(y - mean(y))^2;
    stds =  std(hat_y-y);
    lof = err / ( 1 - ( num_nodes + 2 + penalty * (num_nodes+1) ) / size(x, 1) )^2;
else % 所有节点到输出层的权重都是零，删除所有节点
    lof = 10;
    err = norm(y)^2 / norm(y - mean(y))^2;
    stds =  std(y);
    if weights_of_constant == 0
        BB=[];
    else
        BB = node_values(:, 2:end);
    end
end

