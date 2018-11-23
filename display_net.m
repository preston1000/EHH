function  display_net( B, layer_index, adjacency )
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明

if nargin == 0
    fprintf('at least one argument should be given.\n')
    return 
elseif nargin == 1
        num_nodes = length(B);
        layer_index = zeros(num_nodes, 1);
        for i = 1:num_nodes
            layer_index(i) = size(B{i}, 1);
        end
        adjacency = [];
elseif nargin == 2
    num_nodes = length(B);
    if length(layer_index) ~= num_nodes
        fprintf('the arguments are not consistent in dimension.\n')
        return
    end
    adjacency = [];
elseif nargin > 3
    fprintf('the many arguments.\n')
    return 
else
    num_nodes = length(B);
    if length(layer_index) ~= num_nodes || size(adjacency, 1) ~= num_nodes || size(adjacency, 2) ~= num_nodes
        fprintf('the arguments are not consistent in dimension.\n')
        return
    end
end

MAX_IN_LINE = 4;

fprintf('  layer     |  nodes \n ');
fprintf(strcat(repmat('-', 1, 100), '\n'));
num_layer = max(layer_index);
for i = 1:num_layer
    context = sprintf('   %2d    |', i);
    range = find(layer_index == i);
    
    for j = 1:length(range)
        B_sub = B{range(j)};
        for k = 1:size(B_sub, 1)
            context = strcat(context, sprintf('(%3d, %6.4f)\t', B_sub(k, 2), B_sub(k, 3)));
        end
        context = strcat(context, '||||');
        if mod(j, MAX_IN_LINE) == 0
            context = strcat(context, '\n         |');
        end
    end
    context = strcat(context, '\n');
    fprintf(context)
end


