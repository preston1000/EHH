function [ err, stds, lof ] = statistics( layers, weights, x, y, parameters )
%STATISTICS to compute errors and lof and stds
%   

penalty = parameters.penalty;  % complexity penalty

num_nodes = size(layers, 1);
node_values = cal_node_value(layers, x);
hat_y = node_values * weights;

err = norm(hat_y - y)^2 / norm(y - mean(y))^2;
stds =  std(hat_y-y);
lof = err / ( 1 - ( num_nodes + 2 + penalty * (num_nodes+1) ) / size(x, 1) )^2;
