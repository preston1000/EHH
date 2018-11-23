function [B, weights, id_var_bb, stem_B, adjacency_matrix, id_layer, lof, err, stds]=forward(x, y, x2,y2,parameters)
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

% 参数配置
shares = parameters.shares;  % quantile number of each coordinate, the number of points interpolate in the interval
structure_parameter = parameters.structure;  % the parameters for structure definition
lambda = parameters.lambda;% the parameter for the Lasso regression 

start_time = tic;
%% the first layer
[B0, BB0, id_var_bb0, stem_B0, id_layer0] = ini_basis(x,shares);  % not containing the constant basis

B = B0;
BB = BB0';
stem_B = stem_B0;
id_layer = id_layer0;
id_var_bb = id_var_bb0;

%% 前向过程
num_layer = size(structure_parameter, 2);  % the first layer is not taking into consideration
for layer_index = 2:num_layer+1  % the neurons are added layerwisely
    num_neurons = structure_parameter(layer_index-1);
    % 先将所有可能进行结合的neuron的组合找出来
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
            index_x_1 = id_var_bb{index_in_layer_1(index_combination)};
            index_x_2 = id_var_bb{index_in_layer_2(index_combination)};
            if ~isempty( intersect(index_x_1,index_x_2)) % 有共同的x_i，就不进行组合了
                continue;
            end
            index_combinations = [index_combinations; index_combination];
        end
        candidate_combinations = [candidate_combinations; index_in_layer_1(index_combinations),index_in_layer_2(index_combinations)];
        layer_index_1 = layer_index_1+1;
    end
    % 从可能的组合中选择合适的组合，加入到网络中(随机方法)   
    [B_new, BB_new, stem_B_new, id_var_bb_new, id_layer_new ] = generate_neurons_rand(candidate_combinations,  num_neurons, B, BB, id_var_bb, id_layer);
    B = [B; B_new];
    BB = [BB BB_new];
    stem_B = [stem_B; stem_B_new];
    id_var_bb = [id_var_bb; id_var_bb_new];
    id_layer = [id_layer; id_layer_new];
end
 
%% 后向过程

lof = 10;
for k = 1:length(lambda)
    [Bk, BBk, stem_Bk, Adjak, id_layerk, id_var_bbk, coefk, lofk, errk, stdsk] = prune_node(B, stem_B, id_layer, id_var_bb, x2, y2,lambda(k), parameters);
    execute_prune = 'X';
    if lofk < lof
        B = Bk;
        BB = BBk;
        stem_B = stem_Bk;
        adjacency_matrix = Adjak;
        id_layer = id_layerk;
        id_var_bb = id_var_bbk;
        weights = coefk;
        lof = lofk;
        execute_prune = 'ok';
    end
    fprintf('lambda: %2.2f, error: %6.4f, lof: %6.4f, std: %6.4f, prune? %s \n', lambda(k), errk, lofk, stdsk, execute_prune);
end

%% 输出结果
node_values = cal_node_value(B,stem_B,x);
hat_y = node_values*weights;
err = norm(hat_y - y)^2/norm(y-mean(y))^2;
stds = std(hat_y - y);
time = toc(start_time);

fprintf('Final results: lambda: %2.2f, error: %6.4f, lof: %6.4f, std: %6.4f, ellapsed time: %f \n', lambda(k), err, lof, stds, time);

