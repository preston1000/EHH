function [B_new, BB_new, stem_B_new, id_var_bb_new, id_layer_new ] = generate_neurons_iter(candidate_combinations,  num_layer, num_neurons, B, BB, id_var_bb, id_layer, or_ba0)
% 生成第i层节点的程序（迭代法）。

    N1 = candidate_combinations(:,1);
    N2 = candidate_combinations(:,2);
    vec_min = min(BB(:,N1),BB(:,N2));  %column vector
    rho = (vec_min'*or_ba0)*or_ba0'; %sum(vec_min'*or_ba0,2);  %<vec_min,or_ba0(:,i)>
    rho = rho'; % projection of rho onto the subspace spanned by or_ba0
    num_layer = num_bf+num_neurons;
   
    while num_bf<num_layer
        rho_eff = sqrt(sum(rho.^2,1))./sqrt(sum(vec_min.^2,1)); %the angle between vec_min and its projection
        [~,kmin] = min(rho_eff);
        num_bf = num_bf+1;
        n1 = candi_comb(kmin,1);
        n2 = candi_comb(kmin,2);
        vec_choose = vec_min(:,kmin);
        B{num_bf} = [B{n1};B{n2}];
        stem_B(num_bf,:) = [n1,n2];
        id_var_bb{num_bf} = union(id_var_bb{n1},id_var_bb{n2});
        id_layer(num_bf) = id_layer(n1)+id_layer(n2);
        BB(:,num_bf) = vec_choose;
        % new basis
        beta = vec_choose-or_ba0*(or_ba0'*vec_choose);
        eta_new = beta/norm(beta);
        or_ba0 = [or_ba0,eta_new];
        rho = rho+((vec_min'*eta_new)*eta_new')';
    end
    % An alternative approach
    rho_eff = sqrt(sum(rho.^2,1))./sqrt(sum(vec_min.^2,1)); %
    [rho_sort, k_sort] = sort(rho_eff);
    comb_choose = candi_comb(k_sort(1:num_neurons),:);
    for k = 1:num_neurons
        num_bf = num_bf+1;
        n1 = comb_choose(k,1);
        n2 = comb_choose(k,2);
        B{num_bf} = [B{n1};B{n2}];
        stem_B(num_bf,:) = [n1,n2];
        id_var_bb{num_bf} = union(id_var_bb{n1},id_var_bb{n2});
        id_layer(num_bf) = id_layer(n1)+id_layer(n2);
        BB(:,num_bf) = min(BB(:,n1),BB(:,n2));
    end