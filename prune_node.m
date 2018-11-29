function [LAYERS,  WEIGHTS, stats]=prune_node(layers, x, y, parameters)
%prune_node �����֦���� 
%

%% parameters
% lasso parameters
gamma = parameters.gamma;
rho = parameters.rho;
precision = parameters.precision;
quiet = parameters.quiet;
LAMBDA = parameters.lambda;
lof = parameters.lof;

%% ������������Ӿ��� ��ij��Ԫ�ر�ʾ������i->j
num_nodes = size(layers,1);
%% ��ѵ�����ݣ�����ÿ���ڵ��ֵ��
LAYERS = layers;
WEIGHTS = [];
err = 0;
stds = 0;
for ii = 1:length(LAMBDA)
    adjacency_matrix = stem_to_adjacency( cell2mat(layers(:, 3)) );
    node_values = cal_node_value(layers, x);
    % ����ÿ���ڵ㵽������Ȩ��
    lambda = LAMBDA(ii) * sqrt(2*log10(num_nodes + 1));   % �����������нڵ��������
    weights = lasso(node_values, y, lambda, rho, gamma, quiet); % ������ѵ��ÿ���ڵ㵽����ڵ��Ȩ��
    weights_of_constant = weights(1);
    weights_of_nodes = weights(2:end);

    % ɾ���ڵ�
    if sum(abs(weights_of_nodes)) > precision
        index_active_node = abs(weights_of_nodes) > precision;
        adjacency_matrix(index_active_node, num_nodes + 1) = 1;   % the last column is the weights to output
        % ��ɾ������==0�Ľڵ�
        out_fan = sum(adjacency_matrix, 2);
        rem_index = find(out_fan > precision);  % Ҫ�����Ľڵ���

        num_node_remained = length(rem_index);
        while num_node_remained < num_nodes
            num_nodes = num_node_remained;
            adjacency_matrix = adjacency_matrix(rem_index, [rem_index',end]);
            % �����������
            layers = layers(rem_index, :);
            weights_of_nodes = weights_of_nodes(rem_index);

            out_fan = sum(adjacency_matrix, 2);
            rem_index = find(out_fan > 0);  % Ҫ�����Ľڵ���
            num_node_remained=length(rem_index);
        end
        % ����ȷ���ڵ����ֵ
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