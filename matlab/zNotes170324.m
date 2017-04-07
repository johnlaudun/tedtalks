% Notes from 170324

%Step 0) Load the data and see what we?re working with - 

% Working with the test file of just documents and topic values. Adjusting 
% the name as to be not so annoying. 
load('../data/dt_KK_test.csv')
DT = dt_KK_test;

% Just plotting what the topic values for all the documents are to get a 
% sense of the general trends. This gives a plot line for each document 
% with the x-axis as the topic numbers and the y-axis as the amount of 
% topic present in that document. 
%
% Note - the ? takes the transpose of the matrix. Otherwise, we would have 
% the x-axis as the documents and the y as something nonsensical 
plot(DT')

% Instead of using plot lines, we can also just ?peek? at the matrix using
% imagesc()
imagesc(DT)

% I wanted to see what the distribution of values are for the matrix
% Note - The : turns the matrix into a vector which we can sort. 
plot(sort(DT(:)))

% Getting a sense of the density of the data...
imagesc(DT==0)
imagesc((DT>0).*DT)
imagesc((DT>0))

% Now I wanted to see if our data was normalized and if not, to try playing
% with it in a normalized form...
plot(sum(DT,2),'*')
total_topics = sum(DT,2);
tt_mats = repmat(total_topics,1,40);
dt_normed = DT./tt_mats;
%imagesc(dt_normed)

% Check that our normalization ?worked? (i.e. that each row sums to one). 
plot(sum(dt_normed,2))

% Built SORT_BY_ROW and checked out the output. 
[doc_sort, doc_inds] = sort_by_row(DT);
figure();plot(doc_sort','*')

[doc_sort, doc_inds] = sort_by_row(dt_normed);
figure();plot(doc_sort','*')

% Notes from 170406
% Diff mat
diff_mat = zeros(size(DT,1),5);
for i = 1:5
    diff_mat(:,i) = doc_sort(:,i) - doc_sort(:,i+1);
end

imagesc(diff_mat)
plot(sort(diff_mat(:,1)))
plot(sort(diff_mat(:,2)))
plot(sort(diff_mat(:,1)))
plot(diff_mat)