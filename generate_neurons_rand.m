function layer = generate_neurons_rand(candidate_combinations, num_neurons, first_layer)
% generate_neurons_rand 生成第i层节点的程序（随机法）。
%

    % 从可能的组合中选择合适的组合，加入到网络中(随机方法)
    rand_comb = randperm(length(candidate_combinations));
    comb_choose = candidate_combinations(rand_comb(1:num_neurons),:);
    comb_choose = sortrows(comb_choose);
    
    Beta = cell(num_neurons, 1);
    stem = zeros(num_neurons, 2);
    id_layer = zeros(num_neurons, 1);
    
    for k = 1:num_neurons
        n1 = comb_choose(k,1);
        n2 = comb_choose(k,2);
        
        Beta{k} = [first_layer{n1, 1}; first_layer{n2, 1}];
        stem(k,:) = [n1, n2];
        id_layer(k) = first_layer{n1, 2} + first_layer{n2, 2};
    end

    layer = [Beta num2cell(id_layer, 2) num2cell(stem, 2) ];