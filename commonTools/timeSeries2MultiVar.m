function [trainX, output] = timeSeries2MultiVar(varargin )
%This function is to transform the time series into a multi-variable
%training data. It means that if the given time seris are
%               u(1) u(2)  ....  u(N), y(1) y(2) ... y(N)
% then the training data will be
%       [u(t-na) ... u(t-ulag) y(t-nb) ...y(t-1) ] ---(input)
%                                   y(t)                         ---(output)
% syntax:
%       [trainX, trainY] = timeSeries2MultiVar(x, y, parameters)
%   or
%       [trainX, trainY] = timeSeries2MultiVar(x, model, parameters)
%   or
%       [trainX, trainY] = timeSeries2MultiVar(x, y, model, parameters)
% Input:
%       x: output of time series, vector
%       y: if vector, it is the output of time series (in syntax #1, to prepare the train samples, 
%           in syntax #3, to predict with the true history output)
%       model: the net, with fields
%           B: n*1cell, n is the number of nodes
%           stemB: n*2 matrix, indicating the indices of previous nodes 
%           weights:
%           evalFunc: function to evaluate the values in each node of the net
%       parameters: struct with fields
%           nx, ny: length of historical data for input and output, could be integer or vector, 
%               when any one is vector, it means the specified position of past
%               moments, when both scalar, it means the past sequential moments
%           xDelay: length of delay, integer, only valid when nx and nx are both
%               integer
%           x_interval, y_interval: ranges of x and y for scale 


if nargin > 3 % syntax #3
    x = varargin{1}(:);
    y = varargin{2}(:);
    model = varargin{3};
    parameters = varargin{4};
    predictWithHistory = true;
    trainMode = false;
elseif nargin < 3
    throw(MException('Argument:Invlid', 'too many'))
else
    x = varargin{1}(:);
    if isstruct(varargin{2})
        model = varargin{2};
        predictWithHistory = false;
        trainMode = false;
    elseif isvector(varargin{2})
        y = varargin{2}(:);
        predictWithHistory = false;
        trainMode = true;
    else
        throw(MException('Argument:InvlidType', 'the second should either be vector or struct'))
    end
    parameters = varargin{3};
end

if ~isfield(parameters, 'x_interval') || ~isfield(parameters, 'y_interval')...
    || ~isfield(parameters, 'nx') || ~isfield(parameters, 'ny') ...
    || ~isfield(parameters, 'xDelay')
    throw(MException('Argument:MissingFields', 'parameters should have 5 fields.'))
end


xMin = parameters.x_interval(1);
xWidth = parameters.x_interval(2) - parameters.x_interval(1);
yMin = parameters.y_interval(1);
yWidth = parameters.y_interval(2) - parameters.y_interval(1);
nx = parameters.nx;
ny = parameters.ny;
xDelay = parameters.xDelay;
posStart = max([ny(:) ; nx(:)]);
len = length(x);

if trainMode || predictWithHistory
    inputY = y;
else
    inputY = zeros(len, 1);
end
output = zeros(size(x));

if isscalar(nx) && isscalar(ny)
    trainX = zeros(len - posStart, ny + nx - xDelay + 1);
    for t = (posStart+1) : len
        rangeX = (t - xDelay) : -1 : (t - nx);
        rangeY = (t - 1) : -1 : (t - ny);
        partX = (x(rangeX) - xMin) / xWidth; % extract part of u
        partY = (inputY(rangeY) - yMin) / yWidth; % extract part of y
        trainX(t - posStart, :) = [partY ; partX]';
        if predictWithHistory
            output(t) = feval(model.evalFunc, model, trainX(t - posStart, :)) * model.weights;
        elseif trainMode
            output(t) = inputY(t);
        else
            inputY(t) = feval(model.evalFunc, model, trainX(t - posStart, :)) * model.weights;
        end
    end
else
    trainX = zeros(len - posStart, length(ny) + length(nx) + 1);
    for t = (posStart + 1):len
        partY = (inputY(t - ny) - yMin) / yWidth;
        partX = [x(t - 2); (x(t - nx) - xMin) / xWidth];
        trainX(t-posStart, :) = [partY; partX]';
        if predictWithHistory
            output(t) = feval(model.evalFunc, model.B, model.stemB, trainX(t - posStart, :)) * model.weights;
        elseif trainMode
            output(t) = inputY(t);
        else
            inputY(t) = feval(model.evalFunc, model.B, model.stemB, trainX(t - posStart, :)) * model.weights;
        end
    end
end
if trainMode
    output = output((posStart+1) : len);
end
if ~(trainMode || predictWithHistory)
    output = inputY;
end