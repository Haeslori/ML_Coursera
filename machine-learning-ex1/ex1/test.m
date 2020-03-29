# Basic Operations

# + - * / == ~= && || xor()
a = 3;  # no print 
a = 2   # print 

a = pi;
disp(a)
disp(sprintf('2 decimals: %0.2f', a))

format long
a = pi;
format short
a = pi;

A = [1 2;3 4;5 6];
v = [1 2 3];

v = 1:0.1:2;
v = 1:6;

C = ones(1,3);
C = 2*ones(1,3);

w = zeros(1,3);
w = rand(3,3);

w = randn(1,3);
w = -6 + sqrt(10)*(randn(1, 10000));

#hist(w);         # histogram
#hist(w, 50);

e = eye(4);

A = [1 2; 3 4; 5 6];
size(A);
size(A,1);
size(A,2);

v = [1 2 3 4];
length(v);

pwd; #current working path 

A(:,2);
A([1,3], :);
A(:,2) = [10;11;12]
A = [A, [100, 101, 102]];


# Moving Data Around

#load featuresX.dat
#load priceY.dat
#load('featuresX.dat')

#who    # show parameter
#whos   # show more detail about parameter 

# save hello.mat v;
# save hello.mat v -askii 










