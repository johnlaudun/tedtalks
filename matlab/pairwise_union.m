function [D_union] = pairwise_union(dt_mat)

nr = size(dt_mat,1);

union_mat = zeros(nr);

for i = 1:nr-1
    for j = i:nr
        row_i = dt_mat(i,:);
        row_j = dt_mat(j,:);
        union_ij = sum((row_i + row_j) > 0);
        union_mat(i,j) = union_ij;
        union_mat(j,i) = union_ij;
    end

end

D_union = union_mat;

end