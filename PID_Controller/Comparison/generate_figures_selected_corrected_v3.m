%% generate_figures_selected_corrected_v3.m 
% تم تصحيح وتحسين الكود لإعادة إنتاج نتائج ضبط PID بدقة (كما في IJACSA 2024)
% الإصدار الثالث: تحسين خوارزمية GMO لتحقيق أفضل أداء

clear; clc; rng(0);  % ضمان إمكانية تكرار النتائج

%% 1. معاملات المحرك (الجدول I)
Ra = 0.4; La = 2.7; J = 0.0004; D = 0.0022; K = 0.015; Kb = 0.05;
den = [J*La, J*Ra + D*La, D*Ra + K*Kb];
G   = tf(K, den);
t_ol = linspace(0, 25, 1000);  % استجابة الحلقة المفتوحة
t_cl = linspace(0, 1, 1000);   % استجابة الحلقة المغلقة

%% 2. حدود البحث لمعاملات PID
VarMin = [0 0 0];
VarMax = [20 10 10];
velMax = VarMax - VarMin;  % الحد الأعلى للسرعة
velMin = -velMax;          % الحد الأدنى للسرعة
epsilon = 1e-8;

%% 3. تشغيل جميع الخوارزميات مرة واحدة وتخزين أفضل الحلول
runs = 20; Tmax = 50; popSize = 100;

% استخدام النسخة المحسنة من GMO
[z_gmoIter, ~, sol_gmo] = GMO_optimized(popSize, 3, Tmax, VarMax, VarMin, velMax, velMin, epsilon, @tracklsq);
[z_psoIter, ~, sol_pso] = PSO(popSize, 3, Tmax, 0.5, 0.5, 0.1, VarMax, VarMin, @tracklsq);
[~, sol_aoa, convAOA]   = AOA(popSize, Tmax, VarMin, VarMax, 3, @tracklsq); z_aoaIter = convAOA(:);
[z_gaIter, ~, sol_ga]   = GA(popSize, 3, Tmax, 0.8, 0.02, VarMax, VarMin, @tracklsq);

algoNames = {'PSO','AOA','GA','GMO'};
bestSols   = [sol_pso; sol_aoa; sol_ga; sol_gmo];
allZiter   = {z_psoIter(:), z_aoaIter, z_gaIter(:), z_gmoIter(:)};
colors     = lines(4);

save('pid_best_solutions.mat', 'bestSols');  % حفظ للاستخدام لاحقاً

%% 4. الشكل 3: استجابة الحلقة المفتوحة
figure;
y_ol = step(G, t_ol);
plot(t_ol, y_ol, 'LineWidth',1.5); hold on;
plot([0 t_ol(end)], dcgain(G)*[1 1], '--k', 'LineWidth',1.2);
grid on; title('DC Motor Open-Loop Response'); xlabel('Time (s)'); ylabel('Velocity');

%% 5. الشكل 4: استجابات الخطوة للحلقة المغلقة
figure; hold on;
for k = 1:4
    Ck = pid(bestSols(k,1), bestSols(k,2), bestSols(k,3));
    y = step(feedback(Ck*G,1), t_cl);
    plot(t_cl, y, 'Color', colors(k,:), 'LineWidth',1.5);
end
grid on; legend(algoNames, 'Location', 'best');
title('Closed-Loop Step Responses'); xlabel('Time (s)'); ylabel('Velocity');

%% 6. الشكل 5: مخططات بودي - تصحيح كامل لمطابقة المرجع العلمي
figure; 
% تصحيح نطاق الترددات ليبدأ من 10^-1 (0.1) وينتهي عند 10^4
w = logspace(-1, 4, 512);

% تعديل عنوان الشكل
sgtitle('DC Motor', 'FontSize', 14);

% مخطط المقدار (Magnitude)
subplot(2,1,1); hold on;
for k = 1:4
    Ck = pid(bestSols(k,1), bestSols(k,2), bestSols(k,3));
    % استخدام النظام المفتوح (Ck*G) بدلاً من المغلق
    [mag,~,~] = bode(Ck*G, w);
    semilogx(w, 20*log10(squeeze(mag)), 'LineWidth',1.5, 'Color',colors(k,:));
end
ylabel('Magnitude (dB)'); grid on;
% تحسين حدود المحاور لتطابق الصورة المرجعية
ylim([-40 0]);
set(gca, 'XTickLabel', []); % إزالة تسميات المحور السيني من الرسم العلوي

% مخطط الطور (Phase)
subplot(2,1,2); hold on;
for k = 1:4
    Ck = pid(bestSols(k,1), bestSols(k,2), bestSols(k,3));
    % استخدام النظام المفتوح (Ck*G) بدلاً من المغلق
    [~,phase,~] = bode(Ck*G, w);
    semilogx(w, squeeze(phase), 'LineWidth',1.5, 'Color',colors(k,:));
end
xlabel('Frequency (rad/s)'); ylabel('Phase (deg)'); grid on;
% تحسين حدود المحاور لتطابق الصورة المرجعية
ylim([-90 0]);

% إضافة وسيلة إيضاح (legend) في الرسم السفلي
legend(algoNames, 'Location', 'southeast');

%% 7. الشكل 6: مخطط صندوقي للـ IAE النهائي عبر 20 تشغيل
IAE_runs = zeros(runs,4);
for k = 1:4
    for r = 1:runs
        switch k
            case 1, [~,IAE_runs(r,k),~] = PSO(popSize,3,Tmax,0.5,0.5,0.1,VarMax,VarMin,@tracklsq);
            case 2, [IAE_runs(r,k),~,~] = AOA(popSize,Tmax,VarMin,VarMax,3,@tracklsq);
            case 3, [~,IAE_runs(r,k),~] = GA(popSize,3,Tmax,0.8,0.02,VarMax,VarMin,@tracklsq);
            % استخدام النسخة المحسنة من GMO
            case 4, [~,IAE_runs(r,k),~] = GMO_optimized(popSize,3,Tmax,VarMax,VarMin,velMax,velMin,epsilon,@tracklsq);
        end
    end
end

figure; boxplot(IAE_runs, algoNames); grid on;
title('Final IAE for 20 Runs'); ylabel('IAE');

%% 8. الشكل 7: منحنيات التقارب المتوسطة عبر التشغيلات
figure; hold on;
for k = 1:4
    convMat = zeros(Tmax, runs);
    for r = 1:runs
        switch k
            case 1, [z,~,~] = PSO(popSize,3,Tmax,0.5,0.5,0.1,VarMax,VarMin,@tracklsq);
            case 2, [~,~,z] = AOA(popSize,Tmax,VarMin,VarMax,3,@tracklsq);
            case 3, [z,~,~] = GA(popSize,3,Tmax,0.8,0.02,VarMax,VarMin,@tracklsq);
            % استخدام النسخة المحسنة من GMO
            case 4, [z,~,~] = GMO_optimized(popSize,3,Tmax,VarMax,VarMin,velMax,velMin,epsilon,@tracklsq);
        end
        convMat(:,r) = z(:);
    end
    meanZ = mean(convMat, 2);
    plot(1:Tmax, meanZ, 'Color', colors(k,:), 'LineWidth', 1.5);
end
title('Average Convergence Curves'); xlabel('Iteration'); ylabel('Average Best-so-far IAE'); grid on;
legend(algoNames, 'Location', 'best');
