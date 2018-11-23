function [x_train, y_train, x_test, y_test] = validate_data(A, percent_train)
% ������������ѡȡѵ�������Ͳ�������
% input:
%       A                :   ������������ÿһ����һ������
%       percent_train:  ѵ������ռ��

    n_tol = size(A,1);  %��������
    n_train = floor(n_tol*percent_train);  %ѵ����������������������70%��

    id = randperm(n_tol);% ���ѡ��ѵ�����������µ��ǲ�������
    id_train = id(1:n_train);
    id_test = id(n_train+1:end);

    data_train = A(id_train,:);
    data_test = A(id_test,:);

    x_train = data_train(:,1:end-1);
    y_train = data_train(:,end);

    x_test = data_test(:,1:end-1);
    y_test = data_test(:,end);
    