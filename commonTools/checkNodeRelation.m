function [ flag ] = checkNodeRelation( net )
%CHECKNODERELATION checks if the node relations are correct

    flag = false;
    tmp = net.stemB(:, 1) > 0;
    tmp = net.stemB(tmp, :);
    tmp1 = unique(tmp, 'rows');
    if size(tmp1, 1) ~= size(tmp, 1)
        flag = true;
    end

end

