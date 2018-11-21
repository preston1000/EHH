%dmain2.m

clear

dbstop if error
%% load the data
data_files = { 'fried_delve_std.data'; 'abalone_std.data'; 'cal_housing_std.data'; 'cart_delve_std.data'; 
    'cpusmall_std.data'; 'kin8nm_std.data'; 'r_wpbc_std.data'; 'space_ga_std.data'; };
data_file = data_files{1};
config_file = 'config.ini';

A = textread(strcat('./data/' , data_file));  % 每一行是一个样本，最后一列是y，前面是x

% miny = min(A(:,end));
% maxy = max(A(:,end));
% A(:,end) = (A(:,end)-miny)/(maxy-miny);  %you can finish this step by modifyling the data file


%% 参数配置
parameters = init_par(config_file);
penalty = parameters.penalty;  % complexity penalty
num_train = parameters.num_train;  % 训练次数
percent_train = parameters.percent_train; % 训练样本占总样本的比例
%% 选择训练集和测试集
[x_train, y_train, x_test, y_test] = validate_data(A, percent_train);

dim = size(x_train,2);
dim_y = size(y_train,2);
%%
ns = 2000;%size(x_train,1);%floor(size(x_train,1)/2);%2000;
x_batch = x_train(1:ns,:);
y_batch = y_train(1:ns,:);
x_left = x_train(ns+1:end,:);
y_left = y_train(ns+1:end,:);


adjacency_matrices = cell(num_train, 1);
stem_BBs = cell(num_train, 1);
layer_indices = cell(num_train, 1);
weights_all = cell(num_train, 1);
lofs = zeros(num_train, 1);
errs = zeros(num_train, 1);
stds_all = zeros(num_train, 1);
yahh = zeros(length(y_test), num_train);
err_test = zeros(num_train, 1);
std_test =  zeros(num_train, 1);
for TT = 1:num_train

% [B1,B,coef,id_var_bb, stem_B,Adja,id_layer,t] = forward_batch(x_train, y_train, shares, epsilon,lambda);
% [B1,B,coef,id_var_bb, stem_B,Adja,id_layer,lof,t] = forward_v4(x_train, y_train, shares, structure_parameter);
% [B1,B,coef,id_var_bb, stem_B,Adja,id_layer,lof,rt] = forward_v4(x_struct, y_struct, shares, structure_parameter);
[B, weights, id_var_bb, stem_B, adjacency_matrix, id_layer, lof, err, stds] = forward(x_batch, y_batch, x_left, y_left, parameters);

adjacency_matrices{TT} = adjacency_matrix;
stem_BBs{TT} = stem_B;
layer_indices{TT} = id_layer;
weights_all{TT} = weights;

lofs(TT)  = lof;
errs(TT) = err;
stds_all(TT) = stds;

node_values = cal_node_value(B,stem_B, x_test);
yahh(:,TT) = node_values*weights;
err_test(TT) = norm( yahh(:,TT) - y_test )^2 / norm( y_test - mean( y_test ) )^2;
std_test(TT) = std( yahh(:,TT) - y_test);



end
