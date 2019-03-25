function [ xTrain, yTrain, xTest, yTest ] = generateData( numTrain, numTest )
%GENERATEDATA generates example for "narx"
%   
% input:
%       numTrain:  scalar, length for train data
%       numTest: scalar, length for test data
%       nu : the actual lag for u
%       ny : the actual lag for y
% output:
%       xTrain, yTrain: training example
%       xTest, yTest: testing example
% 

    nu = 1;
    ny = 1;
    %% train data generation
    yTrain = zeros(numTrain, 1);
    ns1 = max(nu, ny);

    t = 1 : numTrain;
    xTrain = sin(pi * t / 50) + sin(pi * t / 20);
    xTrain = xTrain(:);
    
    for t = ns1+1 : numTrain
        yTrain( t ) = yTrain(t - 1) / (1 + (yTrain( t - 1)) ^ 2)+ (xTrain( t - 1))^3;
    end

    %% test data generation
    yTest = zeros(numTest, 1);

    t = 1 : numTest;
    xTest = 0.9 * sin(pi * t / 50) + 1.1 * sin(pi * t / 20);
    xTest = xTest';
    
    for t = ns1+1 : numTest
        yTest( t ) = yTest( t - 1)/(1 + (yTest(t - 1))^2)+ (xTest(t - 1))^3;
    end

end

