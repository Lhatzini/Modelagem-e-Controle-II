clc;  
clear all;
close all;

%% Input data
data = readtable('modelagemdados.csv');
u1 = data.Temp;  
y1 = data.PWM;  


window_size = 5; % Tamanho da janela para a média móvel
u1m = u1 - movmean(u1, window_size, 'omitnan'); 
y1m = y1 - movmean(y1, window_size, 'omitnan'); 

% Normalizar os dados
u1m = (u1m - mean(u1m)) / std(u1m);
y1m = (y1m - mean(y1m)) / std(y1m);

%% Treat data
sampling_time = 1; % tempo de amostragem
dataset = iddata(y1m, u1m, sampling_time);
figure(1);
plot(dataset);
title('Dados de Entrada e Saída');

% Ajuste o tamanho do conjunto de treinamento e validação
training = dataset(1:350); % Conjunto de treinamento
validation = dataset(351:end); % Conjunto de validação

%% Model prediction for ARX and ARMAX
coef_arx = [];
ajustes_arx = [];
aic_arx = [];

coef_armax = [];
ajustes_armax = [];
aic_armax = [];

% Foco no 'prediction'
opt_arx = arxOptions('Focus', 'prediction');
opt_armax = armaxOptions('Focus', 'prediction');

% Ajustes de ordens para capturar mais dinâmica
for na = 1:3  % Ordem de autoregressão
    for nb = 1:3  % Ordem de entrada
        for nk = 0:1  % Defasagem
            % ARX model
            coef_arx = [coef_arx; [na, nb, nk]];
            try
                Mtest_arx = arx(training, [na, nb, nk], opt_arx);
                [~, fit_arx] = compare(Mtest_arx, validation);
                ajustes_arx = [ajustes_arx; fit_arx];
                aic_arx = [aic_arx; aic(Mtest_arx)];
            catch
                ajustes_arx = [ajustes_arx; NaN];
                aic_arx = [aic_arx; NaN];
            end
        end
    end
end

for na = 1:4  
    for nb = 1:4  % Aumentando as ordens de entrada
        for nc = 1:4  % Aumentando a ordem do ruído
            for nk = 0:1  % Defasagem
                % ARMAX model
                coef_armax = [coef_armax; [na, nb, nc, nk]];
                try
                    Mtest_armax = armax(training, [na, nb, nc, nk], opt_armax);
                    [~, fit_armax] = compare(Mtest_armax, validation);
                    ajustes_armax = [ajustes_armax; fit_armax];
                    aic_armax = [aic_armax; aic(Mtest_armax)];
                catch
                    ajustes_armax = [ajustes_armax; NaN];
                    aic_armax = [aic_armax; NaN];
                end
            end
        end
    end
end

%% Organize ARX results
data_arx = table(coef_arx, ajustes_arx, aic_arx, 'VariableNames', {'coef', 'ajuste', 'AIC'});
sorted_data_arx = sortrows(data_arx, 'ajuste', 'descend');

fprintf("Melhores coeficientes ARX: ");
best_coef_arx = sorted_data_arx.coef(1,:);

fprintf("Com o ajuste ARX de: ");
best_adj_arx = sorted_data_arx.ajuste(1);

%% Organize ARMAX results
data_armax = table(coef_armax, ajustes_armax, aic_armax, 'VariableNames', {'coef', 'ajuste', 'AIC'});
sorted_data_armax = sortrows(data_armax, 'ajuste', 'descend');

fprintf("\nMelhores coeficientes ARMAX: ");
best_coef_armax = sorted_data_armax.coef(1,:);

fprintf("Com o ajuste ARMAX de: ");
best_adj_armax = sorted_data_armax.ajuste(1);

%% Comparação dos melhores modelos
fprintf("\nDesempenho ARX: %.2f%%\n", best_adj_arx);
fprintf("Desempenho ARMAX: %.2f%%\n", best_adj_armax);

%% Plotting ARX and ARMAX results
best_arx_model = arx(training, best_coef_arx, opt_arx);
figure(2);
compare(best_arx_model, validation);
title('Comparação ARX com Dados de Validação');

best_armax_model = armax(training, best_coef_armax, opt_armax);
figure(3);
compare(best_armax_model, validation);
title('Comparação ARMAX com Dados de Validação');
