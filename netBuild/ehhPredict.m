function [ resultTest  ] = ehhPredict( net, xTest, yTest, parameters )
%EHHPREDICT predicts with the given net and test sample
    
    nTest = length(yTest);
    if strcmpi(parameters.dataProcessFunction, 'none')
        y1 = zeros(nTest, 1);
        for i = 1:length(yTest)
            y1(i) = feval(model.evalFunc, model, xTest(i, :)) * model.weights;
        end
        err1 = sqrt(norm(y1 - yTest)^2 / nTest);
        std1 = std(y1 - yTest);
        
        resultTest = struct( 'predictWithPredict', y1, 'errP', err1, 'stdP', std1);
    else
        [~, y1] = feval(parameters.dataProcessFunction, xTest, net, parameters);   % only use x
        [~, y2] = feval(parameters.dataProcessFunction, xTest, yTest, net, parameters); % use x and historical y
        
        err1 = sqrt(norm(y1 - yTest)^2 / nTest);
        std1 = std(y1 - yTest);
        err2 = sqrt(norm(y2 - yTest)^2 / nTest);
        std2 = std(y2 - yTest);
        
        resultTest = struct('predictWithHistory', y2, 'predictWithPredict', y1, 'errH', err2, 'errP', err1, 'stdH', std2, 'stdP', std1);
    end
    
end

