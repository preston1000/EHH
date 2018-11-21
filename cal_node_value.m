function node_values = cal_node_value(B, stem_B, x)
% 将变量x代入网络中后，计算网络中各节点的输出值的函数。这时，只需要计算第一隐层中函数的值，即可确定后面所有节点的值
% input: 
%           B         :  表示第一隐层的num_knots*1维cell向量，每个元素是一个1*3向量，[1（固定值）,  所对应x的下标, beta值]
%           stem_B:  num_node*2维矩阵，每一行表示一个节点的两个输入节点的下标
%           x         :  num_data*dim,样本数据，每一行是一个样本，每一列是一个变量的所有取值
% output: 
%           node_values: num_data*(num_nodes+1)维矩阵，第ij个元素表示第i个样本代入之后，网络中第j个节点的输出值

num_nodes=size(stem_B,1);
num_data = size(x,1);

pos_row_id = find(stem_B(:,1)>0);  %positive row index, the rows for the first hidden layer are zero
if isempty(pos_row_id)   % all the neurons are in the first hidden layer
    num1layer = num_nodes;
else
    num1layer = num_nodes - length(pos_row_id);  % number of nodes in the first hidden layer
end
B = cell2mat(B(1:num1layer));  % basis function matrix in the first hidden layer

node_values(:,1) = ones(num_data,1);  %constant basis

%% the first hidden layer
for i = 1:num1layer  
    index_x = B(i, 2);
    beta = B(i, 3);
    node_values(:, i+1) = max(x(:, index_x) - beta, 0);
end
%% subsequent layers
for i = num1layer+1:num_nodes  
    input_1 = stem_B(i, 1);
    input_2 = stem_B(i, 2);
    node_values(:,i+1) = min(node_values(:,input_1+1), node_values(:,input_2+1));
end
    

