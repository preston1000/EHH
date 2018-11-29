function [ output_args ] = backward( x2, y2, parameters )
%BACKWARD 后向过程
%   此处显示详细说明

% 参数配置
lambda = parameters.lambda;% the parameter for the Lasso regression 

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

