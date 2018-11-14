function [B,BB,id_var_bb, stem_B, id_layer]=ini_basis5(x,shares)
% S_B is the candidate bias matrix, dynamically changed
% lm is the upper limit of initial neurons in the first hidden layer


%----------- data preprocessing----------
N=size(x,1);
% t=1:round((N-1)/shares):N;  % index in sorted a
dim=size(x,2);
num_data=size(x,1);
S_B=[];
kn_candi=linspace(1/shares,1-1/shares,shares-1);%0.1:0.1:0.9;
% num_bias=repmat(num_ini,dim,1);

%constant Basis
BB=[]; B=[];
% BB(1,:)=ones(1,N);
% B=[];B{1}=[0 0 0];
id_var_bb=[];
% id_var_bb{1}=0;
num_bf=0;  

% x_i basis
for i=1:dim
    B{i}=[1 i 0];
    id_var_bb{i}=i;
%     stem_B{i+1}=i+1;
    num_bf=num_bf+1;
end
BB=[BB;x'];


for dd=1:dim
    a=x(:,dd);
    knot=[];
    num_0=0;
    kn0=0;
    for ii=1:length(kn_candi)
        num_ii=length(find(a<=kn_candi(ii)));
        if num_ii-num_0>=0.2*num_data
            knot=[knot;(kn0+kn_candi(ii))/2;kn_candi(ii)];
        elseif num_ii-num_0>=0.1*num_data
            knot=[knot;kn_candi(ii)];
        end
        num_0=num_ii;
        kn0=kn_candi(ii);
    end
    S_B=[S_B;knot,repmat(dd,length(knot),1)];
end

num_bits=size(S_B,1);

for ii=1:num_bits
    vm=S_B(ii,2);
    tm=S_B(ii,1);
    num_bf=num_bf+1;
    B{num_bf}=[1 vm tm];
    BB(num_bf,:)=max(x(:,vm)-tm,0)';
    id_var_bb{num_bf}=vm;
end

l_bf = num_bf;
stem_B = zeros(l_bf, 2);
id_layer=repmat(1,l_bf,1);
