function [ flag ] = checkAdjacentMatrix( adjacency )
%CHECKADJACENTMATRIX checks if any two columns in adjacency matrix is the same

    tmp_index = find(sum(adjacency) > 0);
    bb1 = adjacency(:, tmp_index)' ;
    aa = unique(bb1, 'rows');
    tmp = size(aa, 1);
    flag = false;
    if length(tmp_index) ~= tmp
        flag = true;
    end

end

