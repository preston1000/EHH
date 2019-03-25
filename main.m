%% preparation
clear
clc
close all
addpath(genpath(pwd))
dbstop if error
%% test boucWen 
config_file = './configurations/bouc-wen-config.ini';
dataName = './data/bouc-wen.mat';
labels.xTrain = 'u';
labels.yTrain = 'y';
labels.xTest = {'uval_multisine', 'uval_sinesweep'};
labels.yTest = {'yval_multisine', 'yval_sinesweep'};
% ns = (numSample - numSubNets):(numSample - 1);
[ xTrain, yTrain, testSample, parameters ] = prepareData( dataName, config_file, labels );

[ netModels, subNets, weightPerSubNet, statisticsPerNet, statisticTrain, lambdaBest, testResult  ] = netTrainTest( xTrain, yTrain, testSample, parameters );
a =  testSample(1).y;
b = testResult(1).predictWithPredict;
plotResult( 3, a(7001:end), b(7001:end) )

% plotResult( 2, testSample(1).x, testResult(1).predictWithHistory )

return
%% test narx
dataName = './data/narx_1200_800_1_1.mat';
config_file = './configurations/narx-config.ini';
labels.xTrain = 'xTrain';
labels.yTrain = 'yTrain';
labels.xTest = {'xTest'};
labels.yTest = {'yTest'};

[ xTrain, yTrain, testSample, parameters ] = prepareData( dataName, config_file, labels );
[ netModels, subNets, weightPerSubNet, statisticsPerNet, statisticTrain, lambdaBest, testResult  ] = netTrainTest( xTrain, yTrain, testSample, parameters );
%% test narx1


dataName = './data/narx_8000_800_2_3.mat';

%% plot
plotResult( 1, testSample(1).x, testResult(1).predictWithPredict )
plotResult( 2, testSample(1).x, testResult(1).predictWithHistory )



