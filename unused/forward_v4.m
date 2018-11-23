function [B1,B,coef,id_var_bb,stem_B,Adja,id_layer,lof,rt]=forward_v4(x, y, shares, structure_parameter)
%
% input
%       b_left   --- the left endpoint of the interval
%       b_right  --- the right endpoint of the interval
%       x        --- the sample x used for the forward procedure, with size N x dim
%       y        --- the sample y used for the forward procedure, with size N x 1
%       shares   --- the number of points interpolate in the interval
%       Mmax     --- the maximum number of basis functions, larger Mmax
%                    leads to smaller approximation error
% output
%       BBf      --- the basis function evaluated at the points
%       Bf       --- the parameters of the basis function
%       coe      --- the coefficient matrix, dim x 1
%--forward growing of the network-----
%--random strategy---


% lm=40;  % upper limit of # initial basis functions
tic
lambda=[0.1,0.5,1,5,10,100];%1;%lambda*sqrt(2*log10(num_bf));   %%%%%%%% this is the parameter for the Lasso regression to be tuned%%%%%%%%%

%----the first layer-------
[B0,BB0,id_var_bb0,stem_B0,id_layer0]=ini_basis5(x,shares);  %id_layer the the index of layer the basis functions located in
% not containing the constant basis


lof=10;
for k=1:length(lambda)
    [Bk,BBk,stem_Bk,Adjak,id_layerk,id_var_bbk,coefk,lofk]=prune_node(B0,stem_B0,id_layer0,id_var_bb0,x,y,lambda(k));
    if lofk<lof
        B=Bk;
        BB=BBk;
        stem_B=stem_Bk;
        Adja=Adjak;
        id_layer=id_layerk;
        id_var_bb=id_var_bbk;
        coef=coefk;
        lof=lofk;
        lambda0=lambda(k);
    end
end
% A=cal_node_value(B,stem_B,x);  %contain the constant basis

% BB=A(:,2:end);

% B=B0;
% BB=BB0';
% stem_B=stem_B0;
% id_layer=id_layer0;
% id_var_bb=id_var_bb0;

n0=size(stem_B,1);
Adja=zeros(n0,n0+1);
Adja(:,end)=1;

num_layer=size(structure_parameter,2);  % the first layer is not taking into consideration





num_bf=n0;



% BB=A(:,2:end);


for nl=2:num_layer+1  % the neurons are added layerwisely
    B0=B;
    BB0=BB;
    stem_B0=stem_B;
    id_layer0=id_layer;
    id_var_bb0=id_var_bb;
    Adja0=Adja;
    
    k1=1;
    
    candi_comb=[];
    num_neurons=structure_parameter(nl-1);

    while k1<nl
        k2=nl-k1;
        l1=find(id_layer==k1);
        l2=find(id_layer==k2);
        [c1,c2]=meshgrid(l1,l2);
        candi_comb=[candi_comb;c1(:),c2(:)];
        k1=k1+1;
    end
    
    rho=zeros(length(candi_comb),1);
    for k=1:length(candi_comb)
        ca=candi_comb(k,:);
        n1=ca(1);
        n2=ca(2);
        if ismember([n1,n2],stem_B,'rows')
            rho(k)=Inf;
            continue;
        end
        vn_common=intersect(id_var_bb{n1},id_var_bb{n2});
        if ~isempty(vn_common)
            rho(k)=Inf;
            continue;
        end
        vec1=BB(:,n1);
        vec2=BB(:,n2);
        vec_min=min(vec1,vec2);
        rho1=vec1'*vec_min/norm(vec1)/norm(vec_min);
        rho2=vec2'*vec_min/norm(vec2)/norm(vec_min);
        rho(k)=max(rho1,rho2);%cond([vec1,vec2,vec_min]);
    end
    [rho_sort, k_sort]=sort(rho);
    comb_choose=candi_comb(k_sort(1:num_neurons),:);
    for k=1:num_neurons
        num_bf=num_bf+1;
        n1=comb_choose(k,1);
        n2=comb_choose(k,2);
        B{num_bf}=[B{n1};B{n2}];
        stem_B(num_bf,:)=[n1,n2];
        id_var_bb{num_bf}=union(id_var_bb{n1},id_var_bb{n2});
        id_layer(num_bf)=id_layer(n1)+id_layer(n2);
        BB(:,num_bf)=min(BB(:,n1),BB(:,n2));
    end
    [Bk,BBk,stem_Bk,Adjak,id_layerk,id_var_bbk,coefk,lofk]=prune_node(B,stem_B,id_layer,id_var_bb,x,y,lambda0);
    if lofk<lof
        B=Bk;
        BB=BBk;
        stem_B=stem_Bk;
        id_layer=id_layerk;
        id_var_bb=id_var_bbk;
        Adja=Adjak;
        coef=coefk;
        lof=lofk;

    else
        B=B0;
        BB=BB0;
        stem_B=stem_B0;
        id_layer=id_layer0;
        id_var_bb=id_var_bb0;
        Adja=Adja0;
        break;
    end

end
 


% B0=B;
% stem_B0=stem_B;
% id_layer0=id_layer;
% id_var_bb0=id_var_bb;

% lof=10;
% for k=1:length(lambda)
%     [Bk,stem_Bk,Adjak,id_layerk,id_var_bbk,coefk,lofk]=prune_node(B0,stem_B0,id_layer0,id_var_bb0,x,y,lambda(k));
%     if lofk<lof
%         B=Bk;
%         stem_B=stem_Bk;
%         Adja=Adjak;
%         id_layer=id_layerk;
%         id_var_bb=id_var_bbk;
%         coef=coefk;
%         lof=lofk;
%     end
% end

% [B,stem_B,Adja,id_layer,id_var_bb,coef,lof]=prune_node(B0,stem_B0,id_layer0,id_var_bb0,x,y,lambda0);


if max(id_layer)>1
    
    [~,order] = sort(id_layer);
    Adja=Adja(order,[order',end]);
    for nn=1:length(order)
        st_id=find(Adja(:,nn)~=0)';
        if isempty(st_id)
            stem_B(nn,:)=[0 0];
        else
            stem_B(nn,:)=st_id;
        end
    end
    B=B(:,order);
    id_var_bb=id_var_bb(:,order);
    
    id_layer = id_layer(order);%id_layer0(order);
    coef=[coef(1);coef(order+1)];
    
    %--the B1 matrix for the 1st hidden layer neurons---
    pos_row_id=find(stem_B(:,1)>0);  %positive row index, the rows for the first hidden layer are zero
    if isempty(pos_row_id)   % all the neurons are in the first hidden layer
        num1layer=num_bf;
    else
        num1layer=pos_row_id(1)-1;  % number of nodes in the first hidden layer
    end
    B1=cell2mat(B(1:num1layer)');  % basis function matrix in the first hidden layer, not containing the constant basis
else
    B1=cell2mat(B');
end

A=cal_node_value(B1,stem_B,x);
ya=A*coef;
norm(ya-y)^2/norm(y-mean(y))^2

rt=std(ya-y);


t = toc;
