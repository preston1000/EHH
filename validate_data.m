function [x_train, y_train, x_test, y_test] = validate_data(A, percent_train)

    n_tol = size(A,1);  %样本总数
    n_train = floor(n_tol*percent_train);  %训练集样本数（总样本数的70%）

    id = randperm(n_tol);% 随机选择训练样本，余下的是测试样本
    id_train = id(1:n_train);
    id_test = id(n_train+1:end);

    data_train = A(id_train,:);
    data_test = A(id_test,:);

    x_train = data_train(:,1:end-1);
    y_train = data_train(:,end);

    x_test = data_test(:,1:end-1);
    y_test = data_test(:,end);
    