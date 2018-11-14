function [B,BB,stem_B,Adja,id_layer,id_var_bb,alpha,lof]=prune_node(B,stem_B,id_layer,id_var_bb,x,y,lambda)

% there is one more output BB

rho=1;gamma=1;
d=2;  %%%%%%%parameters to be tuned, complexity penalty%%%%%%%%%%%

num_bf=size(stem_B,1);
Adja=sparse(num_bf,num_bf+1);
for kk=1:num_bf
    vkk=stem_B(kk,:);
    if sum(vkk)==0
        continue
    end
    Adja(vkk,kk)=1;
end
A=cal_node_value(B,stem_B,x);
na=size(A,2);
lambda=lambda*sqrt(2*log10(na));   %%%%%%% here lambda can be tuned%%%%%%%%%
[alpha,~]=lasso(A,y,lambda,rho,gamma);
alpha_b=alpha(2:end);


if sum(abs(alpha_b))>0
    valpha=find(alpha_b~=0);
    Adja(valpha,num_bf+1)=1;   % the last column
    rem_index=[];del_index=[];
    for kk=1:num_bf
        if sum(Adja(kk,:))~=0   %for the kk-th neuron
            rem_index=[rem_index;kk];
        else
            del_index=[del_index;kk];
        end
    end
    num_bf2=length(rem_index);
    while num_bf2<num_bf
        num_bf=num_bf2;
        Adja=Adja(rem_index,[rem_index',end]);
        if iscell(B)
            B=B(:,rem_index);
        else
            rem_1=intersect(1:length(B), rem_index);
            B=B(rem_1,:);
        end
        id_layer=id_layer(rem_index);
        id_var_bb=id_var_bb(:,rem_index);
        alpha_b=alpha_b(rem_index);
        stem_B=[];
        for nn=1:length(rem_index)
            st_id=find(Adja(:,nn)~=0)';
            if isempty(st_id)
                stem_B(nn,:)=[0 0];
            else
                stem_B(nn,:)=st_id;
            end
        end
        
        rem_index=[];
        for kk=1:num_bf2
            if sum(Adja(kk,:))~=0
                rem_index=[rem_index;kk];
            end
        end
        num_bf2=length(rem_index);
    end
    
    A=cal_node_value(B,stem_B,x);
    alpha=[alpha(1);alpha_b];
    BB=A(:,2:end);
    
    yhat=A*alpha;
    err=norm(yhat-y)^2/norm(y-mean(y))^2
    N=size(yhat,1);
    lof = err / ( 1 - ( num_bf + 2 + d * (num_bf+1) ) / N )^2
else
    lof=10;
    if alpha(1)==0
        BB=[];
    else
        BB=A(:,2:end);
    end
end

