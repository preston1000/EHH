
Bs = {{[1 1 .2]; [1 1 .3]; [1,2,.4]; [1 2 .5]; [1,2,.6]; [1,3,.7]; [1 3 .8]; [1, 3 .9];[1 3 1];[1 1 .2; 1 2 .4];[1 1 .3;1 2 .6]; [1 2 .4;1 3 .7]; [1 2 .6;1 3 .8];[1 2 .6;1 3 .9]; [1 1 .2;1 3 1]};
    {[1 1 .4]; [1 1 .5]; [1,2,.4]; [1 2 .8]; [1,2,.9]; [1,3,.1]; [1 3 .2]; [1, 3 .9];[1 1 4;1 2 4];[1 1 5; 1 2 8];[1 2 9; 1 3 1];[1 2 8 ; 1 3 2];[1 2 4; 1 3 7]}};
layer_indices = {[ones(9, 1);2*ones(6,1)];[ones(8, 1); 2*ones(5, 1)]};
ind = [1,10;3,10;2,11;5,11;3,12;6,12;5,13;7,13;5,14;8,14;1,15;9,15];
m1 = sparse(ind(:,1), ind(:,2), 1, 15, 15);
ind = [1 9;3 9;2 10;4 10;5 11;6 11;4 12 ;7 12;8 13 ;3 13];
m2  = sparse(ind(:,1), ind(:,2), 1, 13, 13);
adjacency_matrices = {m1;m2};
weights_all = {[];[]};
containing_x_in_node = {{1;1;2;2;2;3;3;3;3;[1 2];[1 2];[2 3];[2 3];[2 3];[1 3]}, {1;1;2;2;2;3;3;3;[1 2];[1 2];[2 3];[2 3];[2 3]}};

[ adjacency, layer_index, B_new ] = merge_net( adjacency_matrices, layer_indices, Bs, weights_all );
display_net( B_new, layer_index, adjacency )
% [Bt,stem_Bt]=aggregate_adja(stem_B1,B11,id_layer1,coef1,stem_B2,B12,id_layer2,coef2);