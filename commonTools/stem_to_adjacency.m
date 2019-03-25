function [ adjacency ] = stem_to_adjacency( stem )
%STEM_TO_ADJACENCY generate the adjacency matrix A in the net,  
% where A(i, j) = 1 indicates that node i is connected to node j
% syntax:
% 
% input:
%   stem: stem: n*2 matrix, indicating the indices of previous nodes and n is the number of nodes
% output:
%   adjacency: n*n matrix, n is the number of nodes
% 
% written by X. Xi

    if isempty(stem) || ~ismatrix(stem) || size(stem, 2) > 2
        fprintf('Invalid input!\n')
        return
    end
    
    num_nodes = size(stem, 1);
    
    [col, ~, value] = find(stem);
    
    adjacency = sparse(value, col, ones(size(col)), num_nodes, num_nodes );


