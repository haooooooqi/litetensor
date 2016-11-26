clear; clc; close all;

num_threads = [1, 4, 16, 64, 128];
times = [12.8531
12.613
11.3805
9.18253
7.89663];
speedup = times(1) ./ times;

plot(num_threads, speedup, 'LineWidth', 3);
hold on;
% plot(num_threads, num_threads, 'LineWidth', 3);

title('Computation Speedup');
xlabel('Number of Threads');
ylabel('Speedup');
set(get(gca,'XLabel'),'FontSize',15);
set(get(gca,'YLabel'),'FontSize',15);
set(get(gca,'TITLE'),'FontSize',15);
set(gca,'FontSize',15);
% len = legend('Actual Speedup', 'Ideal Speedup');
saveas(gcf, '../../report/checkpoint/speedup.jpg')
