function [ xTrain, yTrain, testSample, parameters ] = prepareData( dataFile, configFile, labels )
%PREPAREDATA data preparation
% input:
%       dataFile: path of the data file, string
%       configFile: path of the configuration file, string
%       labels: struct, labels of the training and testing data
%               x, y: name of the train data, string
%               xTest, yTest: cell of string, same length, name of the test
%                       data.
% output:
 
    parameters = init_par(configFile);
    
    data = load(dataFile);
    x = data.(labels.xTrain)(:);
    y = data.(labels.yTrain)(:);
    testSample = struct('x', [], 'y', []);
    numTest = length(labels.xTest);
    for i = 1:numTest
        testSample(i).x = data.(labels.xTest{i})(:);
        testSample(i).y = data.(labels.yTest{i})(:);
    end

    if ~isfield(parameters, 'x_interval') || ~isfield(parameters, 'y_interval')
        parameters.x_interval = [min(x(:)), max(x(:))];
        parameters.y_interval = [min(y(:)), max(y(:))];
    end
    
    if isfield(parameters, 'dataProcessFunction')
        if strcmpi(parameters.dataProcessFunction, 'none')
            xTrain = x;
            yTrain = y;
        else
            [xTrain, yTrain] = feval(parameters.dataProcessFunction, x, y, parameters);
        end
    end

end

