%<-- Part 2.B Read the matrix into the Matlab
clear; clc; close all; 
M = dlmread('../data/ml_ratings.txt');
% M = dlmread('../data/sampletensor.txt');
% M = dlmread('../data/tiny.txt');

M_subs = M(: , 1:3);
M_vals = M(: , 4:4);
rank = 2;

T = sptensor(M_subs, M_vals);

comp_start = tic;
P = cp_als(T, rank, 'maxiters', 20);
comp_time = toc(comp_start);

% diff = 0;
% 
% for n = 1:nnz(T)
%     i = M_subs(n, 1);
%     j = M_subs(n, 2);
%     k = M_subs(n, 3);
%     
%     tmp = 0;
%     for r = 1:rank
%        tmp = tmp + P.lambda(r) * P.U{1}(i, r) * P.U{2}(j, r) * P.U{3}(k, r);
%     end
% 
%     diff = diff + abs(M_vals(n) - tmp) / abs(M_vals(n));
%   
% end
% 
% diff = diff / nnz(T);

% acolumn1 = P.U{1}(:,1:1);
% acolumnsize = size(P.U{1}(:,1:1), 1);
% aindex = transpose([1:acolumnsize]);
% plot1 = [aindex, acolumn1];
% % scatter(plot1(:,1),plot1(:,2));
% 
% acolumn2 = P.U{1}(:,2:2);
% plot2 = [aindex, acolumn2];
% % scatter(plot2(:,1),plot2(:,2));
% 
% bcolumn1 = P.U{2}(:,1:1);
% bcolumnsize = size(P.U{2}(:,1:1), 1);
% bindex = transpose([1:bcolumnsize]);
% plot3 = [bindex, bcolumn1];
% % scatter(plot3(:,1),plot3(:,2));
% 
% bcolumn2 = P.U{2}(:,2:2);
% plot4 = [bindex, bcolumn2];
% % scatter(plot4(:,1),plot4(:,2));
% 
% ccolumn1 = P.U{3}(:,1:1);
% ccolumnsize = size(P.U{3}(:,1:1), 1);
% cindex = transpose([1:ccolumnsize]);
% plot5 = [cindex, ccolumn1];
% % scatter(plot5(:,1),plot5(:,2));
% 
% ccolumn2 = P.U{3}(:,2:2);
% plot6 = [cindex, ccolumn2];
% scatter(plot6(:,1),plot6(:,2));
