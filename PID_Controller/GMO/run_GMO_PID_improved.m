%% run_GMO_PID_improved.m
% سكريبت محسن لضبط PID لمحرك DC باستخدام GMO مع إحصائيات متعددة ورسم boxplot لـ IAE
%
% الوصف:
%   هذا السكريبت يقوم بتنفيذ خوارزمية GMO لضبط معاملات متحكم PID لمحرك تيار مستمر.
%   يتم تنفيذ الخوارزمية لعدة مرات مستقلة (20 مرة) لجمع إحصائيات موثوقة.
%   يتم توليد رسوم بيانية متعددة لتحليل أداء النظام والخوارزمية.
%
% البريد الإلكتروني: elimam_mohamed@univ-biskra.dz                      %
%                            medlimame9@gmail.com                               %
%                                                                   %
% الصفحة الشخصية: https://www.linkedin.com/in/mohamed-el-imam-434321335/ %
%                                                                   %
% المراجع:
%   Rezaei, F., Safavi, H.R., Abd Elaziz, M. et al. GMO: geometric mean optimizer 
%   for solving engineering problems. Soft Comput (2023).
%   https://doi.org/10.1007/s00500-023-08202-z

clear; clc; close all; rng(0);  % إعادة تعيين المولد العشوائي لإمكانية تكرار النتائج

%% 1) نموذج المحرك
% نموذج دالة النقل لمحرك التيار المستمر
% G(s) = 15 / (1.08s^2 + 6.1s + 1.63)
G = tf(15, [1.08 6.1 1.63]);

% عرض معلومات النموذج
disp('=== نموذج محرك التيار المستمر ===');
disp(G);

%% 2) حدود متغيرات PID
% تحديد الحدود الدنيا والعليا لمعاملات PID [Kp, Ki, Kd]
VarMin = [0 0 0];      % الحدود الدنيا (لا يمكن أن تكون المعاملات سالبة)
VarMax = [20 10 10];   % الحدود العليا (قيم مناسبة لنموذج المحرك)
nVar   = numel(VarMin); % عدد المتغيرات (3 لـ PID)

% حدود السرعة مشتقة من حدود المتغيرات
velmax = VarMax - VarMin;  % الحد الأعلى للسرعة
velmin = -velmax;          % الحد الأدنى للسرعة

%% 3) إعدادات GMO المحسنة
np      = 100;    % حجم السكان (عدد الأفراد)
maxit   = 100;    % عدد الأجيال (زيادة من 50 إلى 100 لتحسين النتائج)
epsilon = 1e-10;  % ثابت صغير لتجنب القسمة على صفر (تحسين من 0 إلى قيمة صغيرة جداً)

%% 4) إعداد زمن التكامل ودالة الهدف
Tsim = 5;                                % مدة المحاكاة (ثوانٍ)
tvec = linspace(0, Tsim, 1000);          % متجه الزمن (1000 نقطة)
objFcn = @(sol) IAE_penalty_improved(sol, G, tvec);  % دالة الهدف المحسنة

%% 5) تشغيل GMO لمرات متعددة لجمع الإحصائيات
numRuns = 20;     % عدد التجارب المستقلة
IAEs    = zeros(numRuns, 1);  % مصفوفة لتخزين قيم IAE
Sols    = zeros(numRuns, 3);  % مصفوفة لتخزين معاملات PID
cellZ   = cell(numRuns, 1);   % مصفوفة خلوية لتخزين منحنيات التقارب

fprintf('\n=== بدء تنفيذ %d تجربة مستقلة ===\n\n', numRuns);

% تنفيذ التجارب المستقلة
for i = 1:numRuns
    fprintf('تنفيذ التجربة %d من %d...\n', i, numRuns);
    
    % استدعاء خوارزمية GMO المحسنة
    [z_iter_i, J, sol_i] = GMO_improved( ...
        np, nVar, maxit, VarMax, VarMin, velmax, velmin, epsilon, objFcn);
    
    % تخزين النتائج
    IAEs(i)     = J;              % قيمة IAE النهائية
    Sols(i, :)  = sol_i;          % معاملات PID المثلى
    cellZ{i}    = z_iter_i;       % منحنى التقارب
    
    % عرض نتائج التجربة الحالية
    fprintf('التجربة %02d → IAE = %.6f, [Kp,Ki,Kd] = [%.4f, %.4f, %.4f]\n', ...
        i, J, sol_i(1), sol_i(2), sol_i(3));
end

%% 6) احتساب الإحصائيات واستخلاص أفضل تجربة
[bestIAE, idxBest] = min(IAEs);    % أفضل قيمة IAE ومؤشرها
meanIAE    = mean(IAEs);           % متوسط قيم IAE
worstIAE   = max(IAEs);            % أسوأ قيمة IAE
stdIAE     = std(IAEs);            % الانحراف المعياري لقيم IAE
bestSol    = Sols(idxBest, :);     % أفضل معاملات PID
z_iter     = cellZ{idxBest};       % منحنى التقارب للتجربة الأفضل

% استخراج معاملات PID المثلى
Kp = bestSol(1);
Ki = bestSol(2);
Kd = bestSol(3);

% عرض إحصائيات IAE
fprintf('\n=== إحصائيات IAE عبر %d تجربة ===\n', numRuns);
Tstats = table(bestIAE, meanIAE, worstIAE, stdIAE, ...
    'VariableNames', {'BestIAE','MeanIAE','WorstIAE','StdIAE'});
disp(Tstats);

% عرض أفضل معاملات PID
fprintf('=== أفضل معاملات PID (التجربة %d) ===\n', idxBest);
fprintf('Kp = %.4f   Ki = %.4f   Kd = %.4f   IAE = %.6f\n\n', ...
    Kp, Ki, Kd, bestIAE);

%% 7) تقييم الاستجابة الخطوية للمعاملات المثلى
% بناء متحكم PID باستخدام المعاملات المثلى
C_opt  = pid(Kp, Ki, Kd);
% النظام المغلق (وحدة التغذية الراجعة)
CL_opt = feedback(C_opt * G, 1);
% حساب الاستجابة الخطوية
[y_cl, ~] = step(CL_opt, tvec);

% حساب مؤشرات الأداء
e_cl      = 1 - y_cl;                  % الخطأ
IAE_curve = cumtrapz(tvec, abs(e_cl)); % منحنى IAE التراكمي

% حساب مؤشرات الأداء الزمنية باستخدام دالة stepinfo
S = stepinfo(y_cl, tvec, ...
    'RiseTimeLimits',       [0.1 0.9], ...
    'SettlingTimeThreshold', 0.02, ...
    'StepAmplitude',         1);

% عرض مؤشرات الأداء في جدول
Tperf = table(Kp, Ki, Kd, bestIAE, S.RiseTime, S.SettlingTime, S.Overshoot, ...
    'VariableNames', ...
    {'Kp','Ki','Kd','IAE','RiseTime_s','SettlingTime_s','Overshoot_pct'});
disp('=== أداء الاستجابة الخطوية للنظام المغلق ===');
disp(Tperf);

%% 8) منحنى التقارب للتجربة الأفضل
figure('Name', 'منحنى التقارب', 'NumberTitle', 'off');
semilogy(1:maxit, z_iter, '-o', 'LineWidth', 1.5, 'MarkerIndices', 1:5:maxit);
grid on;
xlabel('الجيل (Iteration)', 'FontWeight', 'bold');
ylabel('أفضل IAE حتى الآن', 'FontWeight', 'bold');
title('منحنى التقارب لخوارزمية GMO', 'FontWeight', 'bold');
set(gca, 'FontSize', 12);
saveas(gcf, 'convergence_curve_GMO.png');

%% 9) استجابة خطوة النظام المغلق
figure('Name', 'استجابة النظام المغلق', 'NumberTitle', 'off');
plot(tvec, y_cl, 'LineWidth', 2);
grid on;
xlabel('الزمن (s)', 'FontWeight', 'bold');
ylabel('الاستجابة y(t)', 'FontWeight', 'bold');
title('استجابة النظام المغلق للإشارة الخطوية', 'FontWeight', 'bold');
set(gca, 'FontSize', 12);
saveas(gcf, 'step_response_GMO.png');

%% 10) القيمة التراكمية للخطأ (IAE)
figure('Name', 'منحنى IAE التراكمي', 'NumberTitle', 'off');
plot(tvec, IAE_curve, 'LineWidth', 2);
grid on;
xlabel('الزمن (s)', 'FontWeight', 'bold');
ylabel('IAE التراكمي', 'FontWeight', 'bold');
title('القيمة التراكمية للخطأ المطلق (IAE) مع الزمن', 'FontWeight', 'bold');
set(gca, 'FontSize', 12);
saveas(gcf, 'accumulated_IAE_GMO.png');

%% 11) مخطط Bode للاستجابة الترددية
figure('Name', 'مخطط Bode', 'NumberTitle', 'off');
bode(CL_opt);
grid on;
title('مخطط Bode للنظام المغلق', 'FontWeight', 'bold');
set(findall(gcf, 'type', 'axes'), 'FontSize', 12);
saveas(gcf, 'bode_plot_GMO.png');

%% 12) boxplot لتوزيع IAE عبر جميع التجارب
figure('Name', 'مخطط الصندوق لـ IAE', 'NumberTitle', 'off');
boxplot(IAEs, 'Labels', {'GMO IAE'});
ylabel('IAE', 'FontWeight', 'bold');
title('مخطط الصندوق لقيم IAE عبر 20 تجربة مستقلة', 'FontWeight', 'bold');
set(gca, 'FontSize', 12);
saveas(gcf, 'boxplot_IAE_GMO.png');

%% 13) مقارنة منحنيات التقارب للخوارزميات المختلفة
% هذا الجزء سيتم تنفيذه بعد تشغيل جميع الخوارزميات

% حفظ النتائج للمقارنة اللاحقة
save('GMO_results.mat', 'bestIAE', 'meanIAE', 'worstIAE', 'stdIAE', ...
    'bestSol', 'z_iter', 'IAEs', 'Sols', 'y_cl', 'IAE_curve', 'S');

fprintf('\n=== تم الانتهاء من تنفيذ خوارزمية GMO وحفظ النتائج ===\n');
