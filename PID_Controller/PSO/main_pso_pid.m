%% main_pso_pid.m
% تحسين PID باستخدام Particle Swarm Optimization (PSO)

clear; clc; rng(0);

% 1) حدود البحث [Kp, Ki, Kd]
VarMin = [0  0  0];
VarMax = [20 10 10];
nVar   = numel(VarMin);

% 2) إعدادات PSO (من Table II) :contentReference[oaicite:4]{index=4}:contentReference[oaicite:5]{index=5}
swarmSize = 100;
maxIter   = 50;
C1 = 0.5;    % cognitive weight 
C2 = 0.5;    % social weight 
w  = 0.1;    % inertia weight 

% 3) استدعاء PSO
[z_iter, z_final, pos_final] = PSO( ...
    swarmSize, nVar, maxIter, C1, C2, w, VarMax, VarMin, @tracklsq);

% 4) أفضل حل
BestSol  = pos_final;   % [Kp, Ki, Kd]
BestCost = z_final;     % أفضل IAE

fprintf('\n=== PSO Optimization ===\n');
fprintf('Best Kp = %.4f, Ki = %.4f, Kd = %.4f, IAE = %.6f\n\n', ...
        BestSol(1), BestSol(2), BestSol(3), BestCost);

% 5) منحنى التقارب
figure('Name','PSO Convergence');
plot(1:maxIter, z_iter, '-ok','LineWidth',1.5);
xlabel('Iteration'); ylabel('Best IAE so far');
title('PSO Convergence Curve'); grid on;

% 6) تحليل الحلقة المغلقة ورسم النتائج
G    = tf(15, [1.08 6.1 1.63]);
Cpid = pid(BestSol(1), BestSol(2), BestSol(3));
CL   = feedback(Cpid*G, 1);

t10  = linspace(0, 10, 1000);
y10  = step(CL, t10);
IAEcurve = cumtrapz(t10, abs(1 - y10));

figure('Name','PSO Step Response');
plot(t10, y10, 'LineWidth',2); grid on;
xlabel('Time (s)'); ylabel('y(t)');
title('PSO‑PID Step Response (0–10 s)');

figure('Name','PSO Bode Plot');
bode(CL); grid on;
title('PSO‑PID Bode Plot');

figure('Name','PSO IAE Curve');
plot(t10, IAEcurve, 'LineWidth',2); grid on;
xlabel('Time (s)'); ylabel('IAE(t)');
title('PSO‑PID Accumulated IAE');
