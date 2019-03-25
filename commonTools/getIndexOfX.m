function [ idX ] = getIndexOfX( B )
%GETINDEXOFX This function is to extract the information of the indices of
%x in each node of the EHH net
%   Input:
%       B: nodes information
%   Output:
%       idX: cell, the indices of X in each node

    numNodes = length(B);
    idX = cell(numNodes, 1);
    for i = 1:numNodes
        idX{i} = B{i}(:, 2);
    end
    
end