function node =first_layer(x,shares)
% inputs:
%           x        :  the sample x used for the forward procedure, with size num_data x dim
%           shares:  the number of points interpolate in the interval
% outputs:
%           node       :  n*3 cell, {x and beta in each node, layer id, previous node id}
%           B            : (dim + num_knot)*1元胞向量，每个元素是一个1*3的向量，分别是[1（固定值）,  所对应x的下标, beta值]

[num_sample, num_variable] = size(x);

% count #samples between i/s~(i+1)/s
beta_candi = (-0.5 / shares) : (1 / shares) : (10.5 / shares); 
[counts,  ~] = hist(x, beta_candi ); % (], head & tail rows are useless (-0.5/s~0) & (1~10.5/s) 
counts = counts(2:end-1, :);

Beta = [ones(num_variable, 1), (1: num_variable)' zeros(num_variable, 1)];

for i=1:num_variable
    index_major = find(counts(:,i) >= 0.2*num_sample);
    index_normal = intersect(find(counts(:,i) < 0.2*num_sample), find( counts(:,i) >= 0.1*num_sample));
    knots = [index_normal / shares;(index_normal - 1) / shares; index_major / shares; (index_major - 1) / shares; (index_major - 0.5) / shares];
    knots = unique(knots);
    Beta = [Beta;ones(length(knots),1)  i*ones(length(knots),1) knots];
end

num_nodes = size(Beta, 1);

Beta = num2cell(Beta, 2);
stem = num2cell(zeros(num_nodes, 2), 2);
id_layer = num2cell(ones(num_nodes,1), 2);

node = [Beta id_layer stem];
