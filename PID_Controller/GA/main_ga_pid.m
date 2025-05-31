%% main_ga_pid.m
% تحسين PID باستخدام Genetic Algorithm (GA)

clear; clc; rng(0);

% 1) Search bounds
VarMin = [0 0 0];
VarMax = [20 10 10];
nVar   = numel(VarMin);

% 2) إعدادات GA (من Table II) :contentReference[oaicite:8]{index=8}:contentReference[oaicite:9]{index=9}
popSize      = 100;
maxIter      = 50;
crossRate    = 0.8;    % crossover probability
mutRate      = 0.02;   % mutation probability

% 3) استدعاء GA
[z_iter, z_final, pos_final] = GA( ...
    popSize, nVar, maxIter, crossRate, mutRate, VarMax, VarMin, @tracklsq);

BestSol  = pos_final;
BestCost = z_final;

fprintf('\n=== GA Optimization ===\n');
fprintf('Best Kp = %.4f, Ki = %.4f, Kd = %.4f, IAE = %.6f\n\n', ...
        BestSol(1), BestSol(2), BestSol(3), BestCost);

% 4) Convergence Curve
figure('Name','GA Convergence');
plot(1:maxIter, z_iter, '-bd','LineWidth',1.5);
xlabel('Generation'); ylabel('Best IAE so far');
title('GA Convergence Curve'); grid on;

% 5) Closed‑loop Analysis
G    = tf(15, [1.08 6.1 1.63]);
Cpid = pid(BestSol(1), BestSol(2), BestSol(3));
CL   = feedback(Cpid*G, 1);

t10  = linspace(0, 10, 1000);
y10  = step(CL, t10);
IAEcurve = cumtrapz(t10, abs(1 - y10));

figure('Name','GA Step Response');
plot(t10, y10, 'LineWidth',2); grid on;
title('GA‑PID Step Response (0–10 s)');
xlabel('Time (s)'); ylabel('y(t)');

figure('Name','GA Bode Plot');
bode(CL); grid on;
title('GA‑PID Bode Plot');

figure('Name','GA IAE Curve');
plot(t10, IAEcurve, 'LineWidth',2); grid on;
title('GA‑PID Accumulated IAE');
xlabel('Time (s)'); ylabel('IAE(t)');
