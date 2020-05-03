function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

for k = 1:K
    % find out which row is marked with k-th centroid, mark with 1, others with 0
    idx_k = (idx == k); 
    
    % set example which is not marked with k-th centroid to 0
    X_k = idx_k.*X;     
    
    % number of example which marked with k-th centroid
    count = sum(idx_k);  
    
    % what if count == 0 ?
    if (count ~= 0)
        centroids(k,:) = sum(X_k)/count;
    endif

endfor


% =============================================================


end

