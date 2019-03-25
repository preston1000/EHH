function [ stem ] = adjacency_to_stem( adjacency )
%adjacency_to_stem transforms the adjacency matrix into the node relation
%matrix
% 
% syntax:
% 
% input:
%       adjacency: n*n matrix, n is the number of nodes
% output:
%       stem: n*2 matrix, indicating the indices of previous nodes and n is the number of nodes
% written by X. Xi

if isempty(adjacency) 
    stem = [];
    return
end

num_nodes = size(adjacency, 1);
stem = zeros(num_nodes, 2);
for i = 1:num_nodes
    previous = find(adjacency(:, i));
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

