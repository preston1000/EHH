%dmain2.m

clear

% load the data
% A=textread('abalone_std.data');
% A=textread('cal_housing_std.data');
% A=textread('cart_delve_std.data');
% A=textread('cpusmall_std.data');
A=textread('fried_delve_std.data');
% A=textread('kin8nm_std.data');
% A=textread('r_wpbc_std.data');
% A=textread('space_ga_std.data');

% miny=min(A(:,end));
% maxy=max(A(:,end));
% A(:,end)=(A(:,end)-miny)/(maxy-miny);  %you can finish this step by modifyling the data file

dbstop if error

n_tol=size(A,1);
n_train=floor(n_tol*0.7);

id=randperm(n_tol);%1:n_tol;%
id_train=id(1:n_train);
id_test=id(n_train+1:end);

data_train=A(id_train,:);
data_test=A(id_test,:);

x_train=data_train(:,1:end-1);
dim=size(x_train,2);
y_train=data_train(:,end);
dim_y=size(y_train,2);

x_test=data_test(:,1:end-1);
y_test=data_test(:,end);

shares=10;
structure_parameter=[20 20];  %!!!here you can also tune

ns=2000;%size(x_train,1);%floor(size(x_train,1)/2);%2000;
x_struct=x_train(1:ns,:);
y_struct=y_train(1:ns,:);
x2=x_train(ns+1:end,:);
y2=y_train(ns+1:end,:);

for TT=1:1

% [B1,B,coef,id_var_bb, stem_B,Adja,id_layer,t]=forward_batch(x_train, y_train, shares, epsilon,lambda);
% [B1,B,coef,id_var_bb, stem_B,Adja,id_layer,lof,t]=forward_v4(x_train, y_train, shares, structure_parameter);
% [B1,B,coef,id_var_bb, stem_B,Adja,id_layer,lof,rt]=forward_v4(x_struct, y_struct, shares, structure_parameter);
[B1,B,coef,id_var_bb, stem_B,Adja,id_layer,lof,rt]=forward_1022(x_struct, y_struct, x2,y2,shares, structure_parameter);

matrix_adja{TT}=Adja;
stem_BB{TT}=stem_B;
layer_inform{TT}=id_layer;
first_layer{TT}=B1;
coefficient{TT}=coef;
% shares is the number of preset points
% epsilon is the error threshold
node_values=cal_node_value(B1,stem_B,x_test);
yahh(:,TT)=node_values*coef;
err_test_before(TT)=norm( yahh(:,TT) - y_test )^2 / norm( y_test - mean( y_test ) )^2;

aa=yahh-y_test;


% lof_forward(TT)=lof;
% 
% len_alpha=size(coef,1);
% len_beta=size(B1,1);
% len_para=len_alpha+len_beta;
% alpha=coef;
% beta=B1(:,3);
% R=eye(dim_y)*5;%0.01;%*rt^2;%4.5;%2  %!!!this parameter should be tuned
% Q1=eye(len_alpha)*1e-6;
% Q2=eye(len_beta)*1e-6;
% P1=eye(len_alpha);
% P2=eye(len_beta);
% 
% for k=1:size(x_train,1)
%     xk=x_train(k,:);
%     yk=y_train(k);  %y_train(k,:)'
%     node_values=cal_node_value(B1,stem_B,xk);
%     yhat=node_values*alpha;
%     B1_values=xk(B1(:,2))'-B1(:,3);
%     num_node=length(node_values);
%     Tkx=zeros(num_node-1, dim);
%     Tkb=zeros(num_node-1,len_beta);
%     for i=2:num_node
%         id=find(B1_values==node_values(i));
%         v=B1(id,2);
%         Tkx(i-1,v)=1;
%         Tkb(i-1,id)=1;
%     end
%     H1k=[1;Tkx*xk'-Tkb*beta];  %H1k=kron(eye(dim_y),[1;Tkx*xk'-Tkb*beta]);
%     H2k=-Tkb'*([zeros(num_node-1,1),eye(num_node-1)]*alpha); %H2k=-Tkb'*[zeros(num_node-1,1),eye(num_node-1)]*reshape(alpha,num_node,dim_y);
%     %Kalman filter prediction
%     Ak=inv(R+H1k'*P1*H1k+H2k'*P2*H2k);
%     K1k=P1*H1k*Ak;
%     alpha=alpha+K1k*(yk-yhat);
%     P1=P1-K1k*H1k'*P1+Q1;
%     K2k=P2*H2k*Ak;
%     beta=beta+K2k*(yk-yhat);
%     beta=min(0.99,max(0,beta));
%     P2=P2-K2k*H2k'*P2+Q2;
%     
%     B1(:,3)=beta;
%     cov1_err(k)=trace(P1);
%     cov2_err(k)=trace(P2);
% end

% lambda=[1e-4,1e-3,0.1,1];%1;%lambda*sqrt(2*log10(num_bf));   %%%%%%%% this is the parameter for the Lasso regression to be tuned%%%%%%%%%
% lof=10;
% for k=1:length(lambda)
%     [Bk,BBk,stem_Bk,Adjak,id_layerk,id_var_bbk,coefk,lofk]=prune_node(B1,stem_B,id_layer,id_var_bb,x_train,y_train,lambda(k));
%     if lofk<lof
%         B=Bk;
%         BB=BBk;
%         stem_B=stem_Bk;
%         Adja=Adjak;
%         id_layer=id_layerk;
%         id_var_bb=id_var_bbk;
%         coef=coefk;
%         lof=lofk;
%     end
% end
% [B,BB,stem_B,Adja,id_layer,id_var_bb,alpha,lof]=prune_node(B1,stem_B,id_layer,id_var_bb,x_train,y_train,0.1);%lambda

% tf_B=B1;
% coef=alpha;
% node_values=cal_node_value(tf_B,stem_B,x_test);
% ya=node_values*coef;
% E_test_deep = norm( ya - y_test )^2 / norm( y_test - mean( y_test ) )^2
% err_test(TT)=E_test_deep;
% ET(TT)=10*log10(norm(yahh-y_test)^2/norm(y_test)^2);
end
