function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% m examples
m = size(X, 1);

% n dimension
n = size(X, 2);

% You need to return the following variables correctly.
idx = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

## Input:
##          X,          m X n, m examples, n dimension
##          centroid,   K X n, K centroid, n dimension 
## Output:
##          idx,        m X 1, m examples, the i-th row store the index of the closest centroid to this i-th example 

% init dist for every example to centroids
dist_vec = zeros(K, 1);

for i = 1:m
    for j = 1:K
        dist_vec(j) = sum((X(i,:) - centroids(j,:)).^2);
    endfor
    
    idx(i) = find(dist_vec == min(dist_vec),1);
endfor





% =============================================================

end

