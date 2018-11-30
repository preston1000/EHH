function [ adjacency ] = stem_to_adjacency( stem )
%STEM_TO_ADJACENCY 此处显示有关此函数的摘要
%   stem 是n*2矩阵，第ij元素>0, 表示stem(i, j)节点的后续节点是i节点
    if isempty(stem) || ~ismatrix(stem) || size(stem, 2) > 2
        fprintf('Invalid input!\n')
        return
    end
    
    num_nodes = size(stem, 1);
    
    [col, ~, value] = find(stem);
    
    adjacency = sparse(value, col, ones(size(col)), num_nodes, num_nodes );


