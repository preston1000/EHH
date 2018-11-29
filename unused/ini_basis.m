function [B, BB, id_var_bb, stem_B, id_layer]=ini_basis(x,shares)
% inputs:
%           x        :  the sample x used for the forward procedure, with size num_data x dim
%           shares:  the number of points interpolate in the interval
% outputs:
%           B            : (dim + num_knot)*1Ԫ��������ÿ��Ԫ����һ��1*3���������ֱ���[1���̶�ֵ��,  ����Ӧx���±�, betaֵ]
%           id_var_bb :(dim + num_knot)������ÿһ����x���±�
%           BB          : (dim + num_knot) *num_data�ľ��󣬵�i��Ϊid_var_bb{i}��Ӧ��x������
%                           ��������Ӧbeta�������0ȡmax��ֵ
%           stem_B    :
%           id_layer    :  the index of layer in which the basis functions located
%           Beta         : num_knot*2 ���󣬴洢�����е�\beta_{km}�����е�2�������е�\beta_{km}����1�д洢���Ƕ�Ӧ��x���±�
%                            is the candidate bias matrix, dynamically changed


%----------- data preprocessing----------
[num_data, dim] = size(x);
beta_candi = linspace(1/shares, 1-1/shares , shares-1);%x_i�ĵȷֵ�

Beta = [];
BB = x';

tmp_id_var_bb = 1:dim;
% basis
for i=1:dim
    % ѡȡx_i�ϵ�beta_i
    all_samples_i = x(:,i); % x_i����������ֵ
    knot=[];
    num_0=0;
    kn0=0;
    for ii = 1:length(beta_candi)
        num_ii = length(find(all_samples_i<=beta_candi(ii)));
        samples_num = num_ii - num_0;
        if samples_num >= 0.2*num_data  % ���������е�ĸ���������������20%ʱ������������е�
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
