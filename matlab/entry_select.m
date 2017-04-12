function [dt_select, dt_select_binary] ...
    = entry_select(dt_mat, dt_inds, min_thresh)

% ENTRY_SELECT takes in a matrix with each row is a document and selects 
% the given indices, turning all others in the row to 0.
%
% INPUT: DT_MAT -- Matrix with each row as a document and each column as a
%                  topic. The entries are the amount of document that is
%                  covered by each topic. 
%        DT_INDS -- The list of top k-topics where k is the number of
%                   columns in DT_INDS. We do not explicitly set k in this
%                   file, but it is implicit given the shape of DT_INDS
%        MIN_THRESH -- This third parameter is an optional parameter. This
%                      parameter allows us to set the minimum contribution
%                      of a topic to a document in order to count as a
%                      contribution. If one does not use three parameters,
%                      then we assume that any contribution above 0 (even
%                      if it is under machine tolerance) is considered a
%                      contribution. 
%
% OUTPUT: DT_SELECT -- Matrix with each row as a document and with the top
%                      used topics being selected. All other entries are 
%                      set to 0. If MIN_THRESH is set, then entries are
%                      both the top used topics and have contributions
%                      greater than or equal to MIN_THRESH. 
%         DT_SELECT_BINARY -- Matrix that is the binary version of 
%                             DT_SELECT. We have other code that does this,
%                             but it was a natural and straight-forward to
%                             include this as well. 


DT = dt_mat;
[nr,nc] = size(DT);

DT_sb = zeros(nr,nc);

% Just select the given indices for each row
for i = 1:nr
    row_inds = dt_inds(i,:);
    DT_sb(i,row_inds) = 1;
end


% We can set a minimum, but the default is to use all the entrys from the
% input matrix. 
if nargin == 3
    DT_min = (DT >= min_thresh);
    DT_sb = DT_sb.*DT_min;
end

% Leverage entrywise multiplication to add nuance to our thresholded matrix
dt_select_binary = DT_sb;
dt_select = DT.*dt_select_binary;

end