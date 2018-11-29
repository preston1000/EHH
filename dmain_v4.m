%dmain2.m

clear

dbstop if error
%% load the data
data_files = { 'fried_delve_std.data'; 'abalone_std.data'; 'cal_housing_std.data'; 'cart_delve_std.data'; 
    'cpusmall_std.data'; 'kin8nm_std.data'; 'r_wpbc_std.data'; 'space_ga_std.data'; };
data_file = data_files{1};
config_file = 'config.ini';

fprintf('开始读取数据文件：%s\n', strcat('./data/' , data_file))
try
    A = textread(strcat('./data/' , data_file));  % 每一行是一个样本，最后一列是y，前面是x
    if ismatrix(A)
        fprintf('数据文件读取成功\n')
    else
        fprintf('数据文件读取失败\n')
        return
    end
catch ME
    fprintf('数据文件读取失败\n')
    return
end
%% 参数配置
fprintf('开始读取配置文件：%s\n', config_file)
parameters = init_par(config_file);
if isstruct(parameters)
    fprintf('配置文件读取完成\n')
else
    fprintf('配置文件读取失败\n')
    return
end
penalty = parameters.penalty;  % complexity penalty
num_train = parameters.num_train;  % 训练次数
percent_train = parameters.percent_train; % 训练样本占总样本的比例
shares = parameters.shares;  % quantile number of each coordinate, the number of points interpolate in the interval
lambda = parameters.lambda;% the parameter for the Lasso regression 
%% choose training set and test set
[x_train, y_train, x_test, y_test] = validate_data(A, percent_train);
n_train = size(x_train, 1);
n_tol = size(A, 1);
fprintf('训练集：%d \t 测试集：%d \t 变量数：%d\n', n_train, n_tol - n_train, size(x_train, 2));
fprintf(strcat(repmat('-', 1, 100), '\n'));
%% train
% forward and backward training samples
ns = 2000;%size(x_train,1);%floor(size(x_train,1)/2);%2000;
x_batch = x_train(1:ns,:);
y_batch = y_train(1:ns,:);
x_left = x_train(ns+1:end,:);
y_left = y_train(ns+1:end,:);

fprintf('前向训练集：%d \t 后向训练集：%d \t 训练集：%d\n', ns, n_train - ns, n_train);
fprintf(strcat(repmat('-', 1, 100), '\n'));
% record results
LAYERS = cell(num_train, 1);
WEIGHTS = cell(num_train, 1);
LOFS = zeros(num_train, 1);
ERRS_TRAIN = zeros(num_train, 1);
STDS_TRAIN = zeros(num_train, 1);

yahh = zeros(length(y_test), num_train);
ERRS_TEST = zeros(num_train, 1);
STDS_TEST =  zeros(num_train, 1);
LOFS_TEST = zeros(num_train, 1);

% the first layer
first_layer = first_layer(x_batch,shares);  %  containing the constant basis

% use (x_batch, y_batch) to train 
for TT = 1:num_train
    % forward procedures to generate hidden layers
    layers = forward_random(first_layer, parameters); % generate new hidden layers
    % backward procedure to prune net, tuning prune parameters
    [layers,  weights, stats] = prune_node(layers, x_left, y_left, parameters);
    % record results
    LAYERS{TT} = layers;
    WEIGHTS{TT} = weights;

    LOFS(TT)  = stats.lof;
    ERRS_TRAIN(TT) = stats.err;
    STDS_TRAIN(TT) = stats.stds;
    % test performances
    [ err_test, stds_test, lof_test ] = statistics( layers, weights, x_test, y_test, parameters );
    ERRS_TEST(TT) = err_test;
    STDS_TEST(TT) = stds_test;
    LOFS_TEST(TT) = lof_test;
    fprintf('#%2d training finished: training error: %6.4f, training lof: %6.4f, training std: %6.4f, test error: %6.4f, test lof: %6.4f, test std: %6.4f \n', TT, stats.err, stats.lof, stats.stds, err_test, lof_test, stds_test);
    fprintf(strcat(repmat('-', 1, 100), '\n'));
end

%% merge
ratio = ones(num_train, 1) / num_train;
[layers, weights] = merge_net(LAYERS, WEIGHTS, ratio );
[ err_test, stds_test, lof_test ] = statistics( layers, weights, x_test, y_test, parameters );
fprintf('Merge performance:  test error: %6.4f, test lof: %6.4f, test std: %6.4f \n', err_test, lof_test, stds_test);
fprintf(strcat(repmat('-', 1, 100), '\n'));

%% prune after merge
[layers,  weights, stats] = prune_node(layers, x_left, y_left, parameters);

[ err_test, stds_test, lof_test ] = statistics( layers, weights, x_test, y_test, parameters );
fprintf('Merge performance(after pruning):  test error: %6.4f, test lof: %6.4f, test std: %6.4f \n', err_test, lof_test, stds_test);
fprintf(strcat(repmat('-', 1, 100), '\n'));


