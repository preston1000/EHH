function [ stem ] = adjacency_to_stem( adjacency )
%UNTITLED2 �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��

if isempty(adjacency) 
    stem = [];
    return
end

num_nodes = size(adjacency, 1);
stem = zeros(num_nodes, 2);
for i = 1:num_nodes
    previous = find(adjacency(:, num_nodes));
    if isempty(previous) 
        stem(i, :) = [0 0];
    elseif length(previous) ~= 2
        fprintf('one node has not exactly two inputs\n')
        stem = [];
        return
    else
        stem(i, :) = previous(:)';
    end
end

