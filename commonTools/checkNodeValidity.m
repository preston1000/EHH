function [ flag ] = checkNodeValidity( net )
%CHECKNODEVALIDITY checks if there are same indices of x in each node of
%the net.
    flag = false;
    for iii = 1:size(net.B, 1)
        tmp = net.B{iii};
        if length(unique(tmp(:,2))) ~= size(tmp, 1)
            flag = true;
            return
        end
    end

end

