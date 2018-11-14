function [B1,B,coef,id_var_bb,stem_B,Adja,id_layer,lof,rt]=forward_1022(x, y, x2,y2,shares, structure_parameter)
%
% input
%       x          ---------------- the sample x used for the forward procedure, with size N x dim
%       y          ---------------- the sample y used for the forward procedure, with size N x 1
%       shares     ---------------- the number of points interpolate in the interval
%       structure_parameter ------- the parameters for structure definition
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




n0=size(stem_B0,1);
Adja0=zeros(n0,n0+1);
Adja0(:,end)=1;
or_ba0=orth(BB0');

num_layer=size(structure_parameter,2);  % the first layer is not taking into consideration

B=B0;
BB=BB0';
stem_B=stem_B0;
id_layer=id_layer0;
id_var_bb=id_var_bb0;



num_bf=n0;



% BB=A(:,2:end);


for nl=2:num_layer+1  % the neurons are added layerwisely
    
    k1=1;
    
    candi_comb=[];
    num_neurons=structure_parameter(nl-1);

    while k1<nl   %possible combinations yielding layer nl
        k2=nl-k1;
        l1=find(id_layer==k1);  % in the k1-th layer
        l2=find(id_layer==k2);
        [c1,c2]=meshgrid(l1,l2);
        c1=c1(:);
        c2=c2(:);
        id_rem=[];
        for kk=1:size(c1,1)
            vn1=id_var_bb{c1(kk)};
            vn2=id_var_bb{c2(kk)};
            vn_common=intersect(vn1,vn2);
            if ~isempty(vn_common)
                continue;
            end
            id_rem=[id_rem;kk];
        end
        candi_comb=[candi_comb;c1(id_rem),c2(id_rem)];
        k1=k1+1;
    end
    
    rho=zeros(length(candi_comb),1);
    N1=candi_comb(:,1);
    N2=candi_comb(:,2);
    vec_min=min(BB(:,N1),BB(:,N2));  %column vector
    rho=(vec_min'*or_ba0)*or_ba0';%sum(vec_min'*or_ba0,2);  %<vec_min,or_ba0(:,i)>
    rho=rho'; % projection of rho onto the subspace spanned by or_ba0
    num_layer=num_bf+num_neurons;
   
%     while num_bf<num_layer
%         rho_eff=sqrt(sum(rho.^2,1))./sqrt(sum(vec_min.^2,1)); %the angle between vec_min and its projection
%         [~,kmin]=min(rho_eff);
%         num_bf=num_bf+1;
%         n1=candi_comb(kmin,1);
%         n2=candi_comb(kmin,2);
%         vec_choose=vec_min(:,kmin);
%         B{num_bf}=[B{n1};B{n2}];
%         stem_B(num_bf,:)=[n1,n2];
%         id_var_bb{num_bf}=union(id_var_bb{n1},id_var_bb{n2});
%         id_layer(num_bf)=id_layer(n1)+id_layer(n2);
%         BB(:,num_bf)=vec_choose;
%         % new basis
%         beta=vec_choose-or_ba0*(or_ba0'*vec_choose);
%         eta_new=beta/norm(beta);
%         or_ba0=[or_ba0,eta_new];
%         rho=rho+((vec_min'*eta_new)*eta_new')';
%     end
    % An alternative approach
%     rho_eff=sqrt(sum(rho.^2,1))./sqrt(sum(vec_min.^2,1)); %
%     [rho_sort, k_sort]=sort(rho_eff);
%     comb_choose=candi_comb(k_sort(1:num_neurons),:);
%     for k=1:num_neurons
%         num_bf=num_bf+1;
%         n1=comb_choose(k,1);
%         n2=comb_choose(k,2);
%         B{num_bf}=[B{n1};B{n2}];
%         stem_B(num_bf,:)=[n1,n2];
%         id_var_bb{num_bf}=union(id_var_bb{n1},id_var_bb{n2});
%         id_layer(num_bf)=id_layer(n1)+id_layer(n2);
%         BB(:,num_bf)=min(BB(:,n1),BB(:,n2));
%     end
    % random approach
    rand_comb=randperm(length(candi_comb));
    comb_choose=candi_comb(rand_comb(1:num_neurons),:);
    comb_choose=sortrows(comb_choose);
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
end
 


lof=10;
for k=1:length(lambda)
    [Bk,BBk,stem_Bk,Adjak,id_layerk,id_var_bbk,coefk,lofk]=prune_node(B,stem_B,id_layer,id_var_bb,x2,y2,lambda(k));
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

num_bf=size(Adja,1);

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



% if max(id_layer)>1
%     
%     [~,order] = sort(id_layer);
%     Adja=Adja(order,[order',end]);
%     for nn=1:length(order)
%         st_id=find(Adja(:,nn)~=0)';
%         if isempty(st_id)
%             stem_B(nn,:)=[0 0];
%         else
%             stem_B(nn,:)=st_id;
%         end
%     end
%     B=B(:,order);
%     id_var_bb=id_var_bb(:,order);
%     
%     id_layer = id_layer(order);%id_layer0(order);
%     coef=[coef(1);coef(order+1)];
    
    %--the B1 matrix for the 1st hidden layer neurons---
    pos_row_id=find(stem_B(:,1)>0);  %positive row index, the rows for the first hidden layer are zero
    if isempty(pos_row_id)   % all the neurons are in the first hidden layer
        num1layer=num_bf;
    else
        num1layer=pos_row_id(1)-1;  % number of nodes in the first hidden layer
    end
    B1=cell2mat(B(1:num1layer)');  % basis function matrix in the first hidden layer, not containing the constant basis
% else
%     B1=cell2mat(B');
% end

A=cal_node_value(B1,stem_B,x);
ya=A*coef;
norm(ya-y)^2/norm(y-mean(y))^2

rt=std(ya-y);


t = toc;
