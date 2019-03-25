function [ flag ] = checkDuplicateNode( net )
%CHECKDUPLICATENODE checks if there are same nodes in the list
%   

    flag = false;
    nodes = net.stemB;
    indicator = (nodes(:, 1) == 0) & (nodes(:, 2) == 0);
    nodes = nodes(~indicator, :);
    tmp = unique(nodes, 'rows');
    if size(tmp, 1) ~= size(nodes, 1)
        flag = true;
    end

end

