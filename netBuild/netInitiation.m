function net = netInitiation(x, shares)
% inputs:
%           x: the sample x used for the forward procedure, with size num_data * dim
%           shares: the number of points interpolate in the interval
% syntax:
%               net = ini_basis(x, shares)
% outputs:
%       x: the input of the net, numSample*nx matrix
%       net: the generated EHH net
%           B: nNode*1 cell vector, each element contains a matrix, k*3, i.e.,[1(fixed value), subscript of x, beta]
%           stemB: nNode*2 matrix, the i-th row is the indices of two
%               previous nodes
%           id_layer: the index of layer in which the basis functions located
%           nx: number of variables
%           nLayer: number of layers
%           nNode: number of nodes


%----------- data preprocessing----------
nx = size(x, 2);

knot = linspace(0, 1-1/shares , shares)'; %Equipartition point in x_i

B = zeros(nx * shares, 3);
B(:, 1) = 1;
B(:, 2) = repmat((1:nx)', shares, 1);
tmp = repmat(knot, 1, nx)';
B(:, 3) = tmp(:);
B = num2cell(B, 2);

stemB = zeros(nx * shares, 2);

id_layer = ones(nx * shares, 1);

net = struct('stemB', stemB, 'id_layer', id_layer, 'nx', nx, 'nLayer', 1, 'nNode', nx * shares);
net.B = B;


