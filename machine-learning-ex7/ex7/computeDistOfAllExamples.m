function dist = computeDistOfAllExamples(X, centroids, idx)
## This function compute the sum of all distance from every example to its centroid
## Input:
##          X,          m X n, m examples, n dimension
##          centroid,   K X n, K centroid, n dimension 
##          idx,        m X 1, m examples, the i-th row store the index of the closest centroid to this i-th example 
## Output:
##          dist,       sum of all distance

##centroid_idx = zeros(size(X));
##centroid_idx_k = zeros(size(X));

##dist = 0;
##for k = 1:size(centroids, 1)
##    idx_k = (idx == k);
##    
##    centroid_idx = idx_k.*repmat(centroids(k,:), size(X,1), 1);
##    
##    dist = dist + sum(sum((idx_k.*X - centroid_idx).^2,2),1);
##end
 
 
 
dist = 0;
K = size(centroids,1);
for i = 1:size(X,1)
    for k = 1:K
        if idx(i) == k
            dist = dist + sum((X(i) - centroids(k,:)).^2);   
        endif
    endfor
endfor
 
 
 
endfunction
