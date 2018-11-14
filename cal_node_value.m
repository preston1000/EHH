function node_values=cal_node_value(B,stem_B,x)
% B-basis function matrix in the first hidden layer
% stem_B- from which the current basis function is minimized
% x-num*dim
% node_values is the node value (including the constant basis)
num_bf=size(stem_B,1);


if ~iscell(B)
    B1=B;
else
    pos_row_id=find(stem_B(:,1)>0);  %positive row index, the rows for the first hidden layer are zero
    if isempty(pos_row_id)   % all the neurons are in the first hidden layer
        num1layer=num_bf;
    else
        num1layer=pos_row_id(1)-1;  % number of nodes in the first hidden layer
    end
    B1=cell2mat(B(1:num1layer)');  % basis function matrix in the first hidden layer
end

num_x=size(x,1);
num1layer=size(B1,1);
M=num_bf;

z(:,1)=repmat(1,num_x,1);  %constant basis
% z=[];  % not including the constant basis

for i=1:num1layer  %the first hidden layer
    vec=B1(i,:);
    vm=vec(2);
    tm=vec(3);
    z(:,i+1)=max(x(:,vm)-tm,0);
end

for i=num1layer+1:M  %subsequent layers
    vec=stem_B(i,:);
    i1=vec(1);
    i2=vec(2);
    z(:,i+1)=min(z(:,i1+1),z(:,i2+1));
end

node_values=z;
    

