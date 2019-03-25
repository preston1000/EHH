function [ netModels, subNets, weightPerSubNet, statisticsPerNet, statisticTrain, lambdaBest, testResult  ] = netTrainTest( xTrain, yTrain, testSample, parameters )
%NETTRAINTEST trains and predicts the EHH net with given data
% 
% syntax:
% 
% input:
%           xTrain, yTrain: 训练数据，分别是n*dim, n*1 array,其中n是样本数，dim是变量数
%           testSample: 测试数据，m*1 struct array,其中m是测试集个数，with field, x and y 
%           parameters: struct，参数，见配置文件说明
% output:
%         netModels: n *1 cell of struct, the trained nets, n is the number of training time
%                   B: node information, cell, each is [index, index of x, value of beta]
%                   id_layer: node position information, array, the index of layer that it lies in
%                   stemB: nNode*2 matrix, indicating the indices of previous nodes 
%                   nx: length of input
%                   nLayer: number of layers
%                   nNode: number of nodes
%                   weights: array
%         subNets: n *1 cell of struct, the sub nets in each training iteration
%         weightPerSubNet: n *1 cell of struct, weights of each sub nets
%         statisticsPerNet: n *1 cell of struct, statistics of each sub nets
%                   timeTrain:
%                   timeForward: time need for forward procedures
%                   timePrune: time need for backward procedures
%                   err: 
%                   lof
%                   stds
%         statisticTrain: n *1 cell of struct, statistics of each training iteration
%                   timeTrain: the whole training time including the sub net and  the merging
%                   timeSubNet: time for training all the sub nets
%         lambdaBest: n *1 cell of struct, best lambda in each training  iteration
%         testResult: : n*k cell of struct, k is the number of training samples
%                     predictWithHistory: using the test sample y to predict
%                     predictWithPredict: start from 0 to predict
%                     errH
%                     errP
%                     stdH
%                     stdP

    %% Parameters 
    trainTimes = parameters.num_train;  % number of training

    %% main body
    % pre-allocation
    netModels = cell(trainTimes, 1);
    subNets = cell(trainTimes, 1);
    weightPerSubNet = cell(trainTimes, 1);
    statisticsPerNet = cell(trainTimes, 1);
    statisticTrain = cell(trainTimes, 1);
    lambdaBest = cell(trainTimes, 1);
    if strcmpi(parameters.dataProcessFunction, 'none')
        testResult = struct( 'predictWithPredict', [], 'errP', 0, 'stdP', 0);
    else
        testResult = struct('predictWithHistory', [], 'predictWithPredict', [], 'errH', 0, 'errP', 0, 'stdH', 0, 'stdP', 0);
    end
    for trainIter = 1:trainTimes
        % train
        [ netMerged, result2, result3, result4, result5, result6 ] = ehhTrain( xTrain, yTrain, parameters);
        netModels{trainIter} = netMerged;
        subNets{trainIter}  = result2;
        weightPerSubNet{trainIter}  = result3;
        statisticsPerNet{trainIter}  = result4;
        statisticTrain{trainIter}  = result5;
        lambdaBest{trainIter}  = result6;
        % test
        netMerged.evalFunc = @cal_node_value;
%         resultTest = struct('predictWithHistory', [], 'predictWithPredict', [], 'errH', 0, 'errP', 0, 'stdH', 0, 'stdP', 0);
        for i = 1:length(testSample)
            testResult(i) = ehhPredict( netMerged, testSample(i).x, testSample(i).y, parameters );
        end
%         testResult{trainIter}  = resultTest;
    end

end

