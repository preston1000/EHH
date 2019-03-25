function [ netMerged, subNets, weightPerSubNet, statisticsPerNet, statisticTrain, lambdaBest ] = ehhTrain( xTrain, yTrain, parameters)
%EHHTRAIN This function is to train the ehh net a few times and merge all
%the resulted nets into one net.
%   
% syntax:
% 
% input:
%       xTrain, yTrain: training samples
%       parameters:
% Output:
%       netMerged: struct, net resulted from merging sub nets
%               weightMerged: array, weights for the merged net
%       subNets: struct array, sub models
%       weightPerSubNet: cell, weights for each sub net
%       statisticsPerNet: struct array
%           timeTrain: time of the whole training in each sub net
%           timeForward: time for forward in each sub net
%           timePrune: time for pruning in each sub net
%           lof:
%           err
%           stds:
%       statisticTrain£ºstruct
%           timeTrain: the whole training time including the sub net and
%                   the merging
%           timeSubNet: time for training all the sub nets
%       lambdaBest: array, best lambda in each sub net

    numSample = size(xTrain, 1);
    numSubNets = parameters.num_sub_net;
    
    ns = parameters.ns;
    
    subNets = struct('stemB', [], 'nx', 0, 'nNode', 0, 'nLayer', 0, 'id_layer', []);
    subNets.B = cell(0);
    weightPerSubNet = cell(numSubNets, 1);
    lambdaBest = zeros(numSubNets, 1);
    statisticsPerNet = struct('timeTrain', 0, 'timeForward', 0, 'timePrune', 0, 'err', 0, 'lof', 0, 'stds', 0);

    nstructure = length(parameters.structure_candidate);
    
    f_ehh_TT = zeros(numSample, numSubNets);
    timeStart = tic;
    % training sub nets
    for kk = 1 : numSubNets   
        xTrainSub = xTrain(1:ns(kk), :);
        yTrainSub = yTrain(1:ns(kk));
        parameters.structure = parameters.structure_candidate{randi(nstructure)};
        
        [result1, result2, result3, result4] = ehhSingle(xTrainSub, yTrainSub, parameters);
        subNets(kk) = result1;
        weightPerSubNet{kk} = result2;
        statisticsPerNet(kk) = result3;
        lambdaBest(kk) = result4;
        f_ehh_TT(:, kk) = cal_node_value(subNets(kk), xTrain) * weightPerSubNet{kk};
    end
    timeSubNet = toc(timeStart);
    % training weights for subnets
    P = 2 * (f_ehh_TT' *f_ehh_TT);
    P = (P + P')/2;
    q = -2 * f_ehh_TT' * yTrain; %y_validate;
    lb = zeros(parameters.num_train, 1);
    [ratio, ~] = quadprog(P, q, [], [], [],[], lb, []);
    validIndicator = ratio < parameters.THRESHOLD_FOR_SUBNETS; % delete those redundant layers with too low weights
    ratio(validIndicator) = 0;
    % merge all sub nets
    [netMerged, weightMerged] = merge_net2(subNets, weightPerSubNet, ratio );
    netMerged.weights = weightMerged;
    statisticTrain = struct('timeTrain', toc(timeStart), 'timeSubNet', timeSubNet);
end

