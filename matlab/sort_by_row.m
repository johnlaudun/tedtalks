function [dt_row_sort, dt_sort_inds] = sort_by_row(dt_mat, lh_flag)

% SORT_BY_ROW takes in a matrix with each row is a document and sorts that
% row either such that the most used topic is first or such that the least 
% used topic is first. 
%
% INPUT: DT_MAT -- Matrix with each row as a document and each column as a
%                  topic
%        LH_FLAG -- Flag so that we can either sort low to high ("L") or 
%                   high to low. Default is high ("H")
%
% OUTPUT: DT_ROW_SORT -- Matrix with each row as a document and with
%                        entries corresponding to the amount of the sorted 
%                        topic that is used.
%         DT_ROW_INDS -- Matrix that tells you which topic corresponds to
%                        which entry of the row. 

if nargin < 2
    lh_flag = 'H';
elseif (lh_flag ~= 'H') && (lh_flag ~= 'L')
        lh_flag = 'H';
end

DT = dt_mat;
[nr,nc] = size(DT);

DT_out = zeros(nr,nc);
DT_ind_out = zeros(nr,nc);

if lh_flag == 'H'
    for i = 1:nr
        [DT_out(i,:), DT_ind_out(i,:)] = sort(DT(i,:),'descend');
    end
    
else
    for i = 1:nr
        [DT_out(i,:), DT_ind_out(i,:)] = sort(DT(i,:),'ascend');
    end
end

dt_row_sort = DT_out;
dt_sort_inds = DT_ind_out;

end

    
