function node_values = cal_node_value(layers, x)
% ������x���������к󣬼��������и��ڵ�����ֵ�ĺ�������ʱ��ֻ��Ҫ�����һ�����к�����ֵ������ȷ���������нڵ��ֵ
% input: 
%           B         :  ��ʾ��һ�����num_knots*1άcell������ÿ��Ԫ����һ��1*3������[1���̶�ֵ��,  ����Ӧx���±�, betaֵ]
%           stem_B:  num_node*2ά����ÿһ�б�ʾһ���ڵ����������ڵ���±�
%           x         :  num_data*dim,�������ݣ�ÿһ����һ��������ÿһ����һ������������ȡֵ
% output: 
%           node_values: num_data*(num_nodes+1)ά���󣬵�ij��Ԫ�ر�ʾ��i����������֮�������е�j���ڵ�����ֵ


num_nodes=size(layers,1);
num_data = size(x,1);

layer_id = cell2mat(layers(:, 2));
pos_row_id = find(layer_id > 1);  %positive row index, the rows for the first hidden layer are zero
if isempty(pos_row_id)   % all the neurons are in the first hidden layer
    num1layer = num_nodes;
else
    num1layer = num_nodes - length(pos_row_id);  % number of nodes in the first hidden layer
end
B = cell2mat(layers(1:num1layer, 1));  % basis function matrix in the first hidden layer

node_values(:,1) = ones(num_data,1);  %constant basis

%% the first hidden layer
for i = 1:num1layer  
    index_x = B(i, 2);
    beta = B(i, 3);
    node_values(:, i+1) = max(x(:, index_x) - beta, 0);
end
%% subsequent layers
for i = num1layer+1:num_nodes  
    input_1 = layers{i, 3}(1);
    input_2 = layers{i, 3}(2);
    node_values(:,i+1) = min(node_values(:,input_1+1), node_values(:,input_2+1));
end
    

