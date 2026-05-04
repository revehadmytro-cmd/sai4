clc;
clear;
close all;

% Початкові дані варіанту 13
x1_const = 1/5;
x2_const = 1/6;
x3_const = 1/7;

m1 = 3;
m2 = 1;

% Формування 10 наборів вхідних даних
x1 = linspace(x1_const, x1_const + 0.9, 10);
x2 = linspace(x2_const, x2_const + 0.9, 10);
x3 = linspace(x3_const, x3_const + 0.9, 10);

% Матриця вхідних параметрів
P = [x1; x2; x3];

% Еталонні значення функції
T = cos(x1) + cos(x2) - sin(x3);

fprintf('Початкові дані варіанту 13:\n');
fprintf('x1 = %.4f, x2 = %.4f, x3 = %.4f\n', x1_const, x2_const, x3_const);
fprintf('m1 = %d, m2 = %d\n\n', m1, m2);

fprintf('Таблиця значень:\n');
fprintf('   x1        x2        x3        T\n');

for i = 1:length(T)
    fprintf('%7.4f  %7.4f  %7.4f  %7.4f\n', x1(i), x2(i), x3(i), T(i));
end

% Створення нейронної мережі
% Для старих версій MATLAB використовується newff.
% Для нових версій, якщо newff недоступна, використовується feedforwardnet.

if exist('newff', 'file') == 2
    net = newff(P, T, [m1 m2], {'tansig', 'purelin'}, 'traingd');
else
    net = feedforwardnet([m1 m2], 'traingd');
    net.layers{1}.transferFcn = 'tansig';
    net.layers{2}.transferFcn = 'purelin';
end

% Налаштування параметрів навчання
net.trainParam.show = 50;
net.trainParam.lr = 0.05;
net.trainParam.epochs = 1000;
net.trainParam.goal = 1e-5;

% Навчання нейронної мережі
[net, tr] = train(net, P, T);

% Моделювання результатів
Y = sim(net, P);
error = T - Y;
mse_value = mean(error .^ 2);

fprintf('\nРезультати навчання:\n');
fprintf('MSE = %.8f\n\n', mse_value);

fprintf('Порівняння реальних і отриманих значень:\n');
fprintf('   №      Реальне T     Прогноз Y     Помилка\n');

for i = 1:length(T)
    fprintf('%4d    %9.4f    %9.4f    %9.4f\n', i, T(i), Y(i), error(i));
end

% Побудова графіків
figure;
plot(1:length(T), T, 'o-', 'LineWidth', 1.5);
hold on;
plot(1:length(Y), Y, '*-', 'LineWidth', 1.5);
grid on;
xlabel('Номер набору даних');
ylabel('Значення функції');
title('Порівняння реальних значень та результатів нейронної мережі');
legend('Реальні значення', 'Прогноз нейронної мережі');

figure;
plotperform(tr);

gensim(net);
