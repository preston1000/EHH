function [B, BB, id_var_bb, stem_B, id_layer]=ini_basis(x,shares)
% inputs:
%           x        :  the sample x used for the forward procedure, with size num_data x dim
%           shares:  the number of points interpolate in the interval
% outputs:
%           B            : (dim + num_knot)*1元胞向量，每个元素是一个1*3的向量，分别是[1（固定值）,  所对应x的下标, beta值]
%           id_var_bb :(dim + num_knot)向量，每一个是x的下标
%           BB          : (dim + num_knot) *num_data的矩阵，第i行为id_var_bb{i}对应的x的所有
%                           样本与相应beta相减后与0取max的值
%           stem_B    :
%           id_layer    :  the index of layer in which the basis functions located
%           Beta         : num_knot*2 矩阵，存储了所有的\beta_{km}，其中第2列是所有的\beta_{km}，第1列存储的是对应的x的下标
%                            is the candidate bias matrix, dynamically changed


%----------- data preprocessing----------
[num_data, dim] = size(x);
beta_candi = linspace(1/shares, 1-1/shares , shares-1);%x_i的等分点

Beta = [];
BB = x';

tmp_id_var_bb = 1:dim;
% basis
for i=1:dim
    % 选取x_i上的beta_i
    all_samples_i = x(:,i); % x_i的所有样本值
    knot=[];
    num_0=0;
    kn0=0;
    for ii = 1:length(beta_candi)
        num_ii = length(find(all_samples_i<=beta_candi(ii)));
        samples_num = num_ii - num_0;
        if samples_num >= 0.2*num_data  % 落在区间中点的个数大于样本数的20%时，增加区间的中点
            knot = [knot;(kn0+beta_candi(ii))/2;beta_candi(ii)];
        elseif samples_num >= 0.1*num_data
            knot = [knot;beta_candi(ii)];
        end
        num_0 = num_ii;
        kn0 = beta_candi(ii);
    end
    Beta = [Beta;  i*ones(length(knot),1) knot];
    tmp_BB = max(repmat(x(:,i), 1, length(knot)) - repmat(knot', num_data, 1), 0)';
    BB = [BB; tmp_BB];
    tmp_id_var_bb = [tmp_id_var_bb  ones(1, length(knot))];
end

id_var_bb = num2cell(tmp_id_var_bb');
affines = [ones(dim,1) [1:dim]' zeros(dim, 1); 
               ones( size(Beta,1), 1), Beta];
B = num2cell(affines, 2);

l_bf = length(id_var_bb);
stem_B = zeros(l_bf, 2);
id_layer = ones(l_bf,1);
