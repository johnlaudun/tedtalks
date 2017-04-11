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
%plot(sum(dt_normed,2))

% Built SORT_BY_ROW and checked out the output. 
[doc_sort, doc_inds] = sort_by_row(DT);
figure();plot(doc_sort','*')

[doc_sort, doc_inds] = sort_by_row(dt_normed);
figure();plot(doc_sort','*')

% Notes from 170406
% Diff mat compares the differences between how much the topics are used in
% order of how much they are used. 
diff_mat = zeros(size(DT,1),5);
for i = 1:5
    diff_mat(:,i) = doc_sort(:,i) - doc_sort(:,i+1);
end

% Just making some pictures to see what the results look like
imagesc(diff_mat)
plot(sort(diff_mat(:,1)))
plot(sort(diff_mat(:,2)))
plot(sort(diff_mat(:,1)))
figure();plot(diff_mat)

% The above images are fairly "fuzzy" and it is hard to tell what the
% trends are. Instead, we will sort the documents based on how much they
% use their most used topics. 
[~,inds] = sort(doc_sort(:,1));
test_mat = [doc_sort(inds,1), sum(doc_sort(inds,1:2),2), ...
    sum(doc_sort(inds,1:3),2),sum(doc_sort(inds,1:4),2), ...
    sum(doc_sort(inds,1:5),2), sum(doc_sort(inds,1:6),2)];
% This makes the 6 waves plot
plot(test_mat,'*')

% These plots allow you see just the first 6 topics in order of te most
% used topics. Plot() shows the values in order, while imagesc() shows a
% heat map of the values: 
% figure(); plot(doc_sort(inds,1:6))
% figure(); imagesc(doc_sort(inds,1:6))

% To think critically about the number of topics that we should be using,
% we need to examine how much of each document is explained by the top used
% documents, the top 2 documents, the top 3 documents, ... to the top 6
% documents. 
test = doc_sort(:,1);
test2 = sum(doc_sort(:,1:2),2);
test3 = sum(doc_sort(:,1:3),2);
test4 = sum(doc_sort(:,1:4),2);
test5 = sum(doc_sort(:,1:5),2);
test6 = sum(doc_sort(:,1:6),2);

% We are counting the number of documents that have at least 90% covered by
% the first N most used topics (for N <= 6). 
high_tc_vec = [sum(test >= 0.9), sum(test2 >= 0.9), sum(test3 >= 0.9), ...
 sum(test4 >= 0.9), sum(test5 >= 0.9), sum(test6 >= 0.9)];


