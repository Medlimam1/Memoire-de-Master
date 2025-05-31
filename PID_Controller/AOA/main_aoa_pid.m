%% main_aoa_pid.m
% تحسين PID باستخدام Arithmetic Optimization Algorithm (AOA)

clear; clc; rng(0);

% 1) Search bounds for PID gains
VarMin = [0  0  0];
VarMax = [20 10 10];
nVar   = numel(VarMin);

% 2) AOA parameters (from Abualigah et al. :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1})
popSize = 100;    % N
maxIter = 50;     % M_Iter
LB      = VarMin; % lower bound
UB      = VarMax; % upper bound
Dim     = nVar;   % dimensionality
F_obj   = @tracklsq;

% 3) Run AOA (signature: AOA(N,M_Iter,LB,UB,Dim,F_obj))
[Best_FF, Best_P, Conv_curve] = AOA( ...
    popSize, maxIter, LB, UB, Dim, F_obj);

% unpack results
BestSol  = Best_P;    % [Kp, Ki, Kd]
BestCost = Best_FF;   % best IAE

fprintf('\n=== AOA Optimization ===\n');
fprintf('Best Kp = %.4f, Ki = %.4f, Kd = %.4f, IAE = %.6f\n\n', ...
        BestSol(1), BestSol(2), BestSol(3), BestCost);

% 4) Plot convergence
figure('Name','AOA Convergence');
plot(1:maxIter, Conv_curve, '-sr','LineWidth',1.5);
xlabel('Iteration'); ylabel('Best IAE so far');
title('AOA Convergence Curve'); grid on;

% 5) Closed‑loop analysis
G    = tf(15, [1.08 6.1 1.63]);            % plant transfer function
Cpid = pid(BestSol(1), BestSol(2), BestSol(3)); 
CL   = feedback(Cpid * G, 1);

% 6) Step response over 0–10 s
t10  = linspace(0, 10, 1000);
y10  = step(CL, t10);

figure('Name','AOA Step Response');
plot(t10, y10, 'LineWidth',2); grid on;
title('AOA‑PID Step Response (0–10 s)');
xlabel('Time (s)'); ylabel('y(t)');

% 7) Bode plot
figure('Name','AOA Bode Plot');
bode(CL); grid on;
title('AOA‑PID Bode Plot');

% 8) Accumulated IAE curve
IAEcurve = cumtrapz(t10, abs(1 - y10));
figure('Name','AOA IAE Curve');
plot(t10, IAEcurve, 'LineWidth',2); grid on;
title('AOA‑PID Accumulated IAE');
xlabel('Time (s)'); ylabel('IAE(t)');
