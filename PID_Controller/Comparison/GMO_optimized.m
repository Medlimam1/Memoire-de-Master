%___________________________________________________________________%
%               GMO: محسّن المتوسط الهندسي - نسخة محسنة                %
%                                                                   %
% تم تطويره في MATLAB R2016b                                        %
% البريد الإلكتروني: elimam_mohamed@univ-biskra.dz                   %
%                medlimame9@gmail.com                               %
%                                                                   %
% الصفحة الشخصية: https://www.linkedin.com/in/mohamed-el-imam-434321335/ %
%                                                                   %
% الورقة البحثية الرئيسية:                                          %
% Rezaei, F., Safavi, H.R., Abd Elaziz, M. et al.                   %
% GMO: محسّن المتوسط الهندسي لحل المشكلات الهندسية.                   %
% Soft Comput (2023). https://doi.org/10.1007/s00500-023-08202-z    %
%___________________________________________________________________%

% GMO: محسّن المتوسط الهندسي - نسخة محسنة لتحقيق أداء أفضل                                                                
function [z_iter, z_final, pos_final] = GMO_optimized(np, nx, maxit, varmax, varmin, velmax, velmin, epsilon, fobj)
%% GMO - خوارزمية محسّن المتوسط الهندسي المحسنة لتحسين معاملات متحكم PID
%
% الإدخالات:
%   np      - عدد الأفراد في المجتمع (حجم السكان)
%   nx      - عدد المتغيرات (أبعاد المسألة، عادة 3 لـ [Kp, Ki, Kd])
%   maxit   - العدد الأقصى للتكرارات (الأجيال)
%   varmax  - الحد الأعلى للمتغيرات (مصفوفة 1×nx)
%   varmin  - الحد الأدنى للمتغيرات (مصفوفة 1×nx)
%   velmax  - الحد الأعلى للسرعة (مصفوفة 1×nx)
%   velmin  - الحد الأدنى للسرعة (مصفوفة 1×nx)
%   epsilon - ثابت صغير لتجنب القسمة على صفر
%   fobj    - مؤشر لدالة الهدف (IAE_penalty)
%
% الإخراجات:
%   z_iter    - مصفوفة قيم دالة الهدف لأفضل حل في كل تكرار (maxit×1)
%   z_final   - القيمة النهائية لدالة الهدف (أفضل قيمة)
%   pos_final - موضع أفضل حل (معاملات PID المثلى)
%
% الوصف:
%   تعتمد خوارزمية GMO على مفهوم المتوسط الهندسي لتحديد أفضل الحلول وتوجيه
%   عملية البحث. تستخدم آلية تكيفية لتحديث المواضع والسرعات بناءً على
%   مؤشرات اللياقة المحسوبة من دالة سيجمويد معدلة.
%
% تحسينات النسخة الجديدة:
%   1. تحسين استراتيجية التهيئة الأولية للمواضع
%   2. تعديل معادلة حساب اللياقة لتعزيز التمييز بين الحلول
%   3. تحسين آلية تحديث السرعة والموضع
%   4. إضافة استراتيجية البحث المحلي للحلول الواعدة
%   5. تعديل معاملات التكيف لتحسين التوازن بين الاستكشاف والاستغلال

%% تهيئة المتغيرات
% مصفوفات لتخزين أفضل المواضع الشخصية والعامة والموجهات
pp_pbest = zeros(np, nx);    % أفضل موضع شخصي لكل فرد
pp_kbest = zeros(np, nx);    % أفضل k مواضع في المجتمع
pp_guide = zeros(np, nx);    % موجه الحركة لكل فرد
mutant = zeros(1, nx);       % متجه الطفرة
stdev2 = zeros(1, nx);       % الانحراف المعياري للمواضع
index = zeros(np, 1);        % مؤشرات الأفراد
fit = zeros(maxit, np);      % مصفوفة قيم اللياقة
DFI = zeros(np, 1);          % عوامل التأثير الديناميكية
optimal_pos = zeros(maxit, nx); % مصفوفة المواضع المثلى في كل تكرار
z_pbest = zeros(np, 1);      % قيم دالة الهدف لأفضل المواضع الشخصية
z_kbest = zeros(np, 1);      % قيم دالة الهدف لأفضل k مواضع
z_optimal = inf(maxit, 1);   % أفضل قيمة لدالة الهدف في كل تكرار
pos_final = zeros(1, nx);    % الموضع النهائي الأمثل
z_iter = zeros(maxit, 1);    % مصفوفة قيم دالة الهدف لأفضل حل في كل تكرار
gbest = zeros(1, nx);        % أفضل موضع عام (global best)
z_gbest = inf;               % قيمة دالة الهدف لأفضل موضع عام

% معاملات استراتيجية k-best التكيفية - تحسين معدل التناقص
kbest_max = np;              % الحد الأقصى لعدد أفضل الحلول (يبدأ بكل المجتمع)
kbest_min = 2;               % الحد الأدنى لعدد أفضل الحلول (ينتهي بأفضل حلين)

% معاملات جديدة للتحسين
alpha = 1.2;                 % معامل تعزيز التمييز في دالة اللياقة
beta = 0.8;                  % معامل تحسين البحث المحلي
local_search_prob = 0.1;     % احتمالية تطبيق البحث المحلي
intensification_factor = 1.5; % معامل تكثيف البحث حول الحلول الواعدة

%% عملية التهيئة الأولية المحسنة
it = 1;  % عداد التكرارات

% استدعاء دالة التهيئة المحسنة لإنشاء المواضع والسرعات الأولية
% تحسين التوزيع الأولي للمواضع باستخدام استراتيجية التوزيع المتنوع
pp = zeros(np, nx);
pv = zeros(np, nx);

% توزيع المواضع الأولية بشكل أكثر تنوعاً باستخدام مزيج من التوزيعات
for j = 1:np
    if j <= np/3
        % مجموعة 1: توزيع منتظم
        pp(j,:) = varmin + (varmax - varmin) .* rand(1, nx);
    elseif j <= 2*np/3
        % مجموعة 2: توزيع متمركز حول القيم المتوسطة
        center = (varmax + varmin) / 2;
        range = (varmax - varmin) / 4;
        pp(j,:) = center + range .* (2*rand(1, nx) - 1);
    else
        % مجموعة 3: توزيع متمركز حول الحدود
        if rand < 0.5
            pp(j,:) = varmin + (varmax - varmin) .* rand(1, nx) * 0.3;
        else
            pp(j,:) = varmax - (varmax - varmin) .* rand(1, nx) * 0.3;
        end
    end
    
    % تقييد المواضع ضمن الحدود المسموحة
    pp(j,:) = max(min(pp(j,:), varmax), varmin);
    
    % تهيئة السرعات بشكل عشوائي
    pv(j,:) = velmin + (velmax - velmin) .* rand(1, nx);
end

%% تقييم دالة الهدف للمجتمع الأولي
for j = 1:np
    % حساب قيمة دالة الهدف (IAE) للفرد j
    z = fobj(pp(j,:));
    % تخزين قيمة دالة الهدف وموضع الفرد كأفضل موضع شخصي أولي
    z_pbest(j) = z;
    pp_pbest(j,:) = pp(j,:);
    
    % تحديث أفضل موضع عام إذا كان الموضع الحالي أفضل
    if z < z_gbest
        z_gbest = z;
        gbest = pp(j,:);
    end
end

% حساب الانحراف المعياري للمواضع المثلى الشخصية (يستخدم في آلية الطفرة)
stdev2 = std(pp_pbest);
max_stdev2 = max(stdev2);
% حساب المتوسط والانحراف المعياري لقيم دالة الهدف (تستخدم في حساب اللياقة)
ave = mean(z_pbest);
stdev = std(z_pbest);

% تحديد عدد أفضل الحلول k بشكل تكيفي (يتناقص خطياً مع التكرارات)
kbest = round(kbest_max - (kbest_max - kbest_min)*(it/maxit)^1.5); % تعديل معدل التناقص
n_best = kbest;

%% حساب مؤشرات اللياقة باستخدام دالة سيجمويد معدلة ومحسنة
% المعادلة (7) في الورقة البحثية الأصلية مع تحسينات
for j = 1:np
    index(j) = j;
    prod = 1;
    for jj = 1:np
        if jj ~= j
            % دالة سيجمويد معدلة ومحسنة لحساب احتمالية كون الفرد jj أفضل من المتوسط
            % تعزيز التمييز بين الحلول باستخدام معامل alpha
            sigmoid_val = 1/(1+exp((-4*alpha)/(stdev*sqrt(exp(1)))*(z_pbest(jj)-ave)));
            prod = prod * sigmoid_val;
        end
    end
    fit(it,j) = prod;  % قيمة اللياقة للفرد j
end

% ترتيب الأفراد تنازلياً حسب قيم اللياقة
[~, sorted_indices] = sort(fit(it,:), 'descend');

% حساب مجموع قيم اللياقة لأفضل k أفراد
sum1 = sum(fit(it,sorted_indices(1:n_best)));

% تخزين قيم دالة الهدف ومواضع أفضل k أفراد
for j = 1:n_best
    idx = sorted_indices(j);
    z_kbest(j) = z_pbest(idx);
    pp_kbest(j,:) = pp_pbest(idx,:);
end

%% حساب موجهات الحركة لكل فرد باستخدام عوامل التأثير الديناميكية المحسنة
% المعادلة (8) في الورقة البحثية الأصلية مع تحسينات
for j = 1:np
    pp_guide(j,:) = zeros(1,nx);
    
    % إضافة تأثير أفضل موضع عام (gbest) بنسبة معينة
    gbest_influence = 0.2; % نسبة تأثير أفضل موضع عام
    
    for jj = 1:n_best
        idx = sorted_indices(jj);
        if idx ~= j
            % حساب عامل التأثير الديناميكي (DFI) المحسن للفرد idx
            % تعزيز تأثير الحلول الأفضل
            rank_factor = 1 + (n_best - jj) / n_best; % معامل الرتبة (يزيد تأثير الحلول الأفضل)
            DFI(jj) = rank_factor * fit(it,idx)/(sum1 + epsilon);
            
            % تحديث موجه الحركة للفرد j
            pp_guide(j,:) = pp_guide(j,:) + DFI(jj).*pp_kbest(jj,:);
        end
    end
    
    % دمج تأثير أفضل موضع عام (gbest)
    pp_guide(j,:) = (1 - gbest_influence) * pp_guide(j,:) + gbest_influence * gbest;
end

% حفظ أفضل قيمة هدف وموضع في التكرار الأول
[z_optimal(it), best_idx] = min(z_pbest);
optimal_pos(it,:) = pp_pbest(best_idx,:);
z_iter(it) = z_optimal(it);

% طباعة معلومات التكرار الأول بتنسيق أفضل
fprintf('التكرار %3d: أفضل IAE = %.6f\n', it, z_optimal(it));

%% الحلقة الرئيسية للخوارزمية
while it < maxit
    it = it + 1;
    % معامل الوزن التكيفي المحسن (يتناقص بشكل غير خطي من 1 إلى 0)
    w = (1 - (it/maxit)^2); % تعديل معدل التناقص لتحسين التوازن بين الاستكشاف والاستغلال

    %% تحديث المواضع والسرعات لكل فرد بطريقة محسنة
    for j = 1:np
        % توليد متجه طفرة محسن باستخدام موجه الحركة والانحراف المعياري
        % المعادلة (9) في الورقة البحثية الأصلية مع تحسينات
        exploration_factor = 1 + 0.5 * (1 - it/maxit); % معامل استكشاف يتناقص مع التكرارات
        mutant = pp_guide(j,:) + w * exploration_factor * randn(1,nx).*(max_stdev2 - stdev2);
        
        % تحديث السرعة باستخدام معادلة السرعة المعدلة والمحسنة
        % المعادلة (10) في الورقة البحثية الأصلية مع تحسينات
        cognitive_factor = 1 + (2*rand(1,nx)-1)*w; % معامل إدراكي محسن
        
        % إضافة تأثير أفضل موضع عام (gbest) في معادلة تحديث السرعة
        social_factor = 0.2 * (1 - w); % معامل اجتماعي يزداد مع تقدم التكرارات
        
        pv(j,:) = w*pv(j,:) + cognitive_factor.*(mutant - pp(j,:)) + social_factor.*(gbest - pp(j,:));

        % تقييد السرعة ضمن الحدود المسموحة
        pv(j,:) = max(min(pv(j,:), velmax), velmin);
        
        % تحديث الموضع وتقييده ضمن الحدود المسموحة
        pp(j,:) = pp(j,:) + pv(j,:);
        pp(j,:) = max(min(pp(j,:), varmax), varmin);

        % تقييم دالة الهدف للموضع الجديد
        z = fobj(pp(j,:));
        
        % تحديث أفضل موضع شخصي إذا كان الموضع الجديد أفضل
        if z < z_pbest(j)
            z_pbest(j) = z;
            pp_pbest(j,:) = pp(j,:);
            
            % تحديث أفضل موضع عام إذا كان الموضع الجديد أفضل
            if z < z_gbest
                z_gbest = z;
                gbest = pp(j,:);
                
                % تطبيق البحث المحلي على أفضل حل جديد
                if rand < local_search_prob
                    % استراتيجية البحث المحلي: استكشاف محيط الحل الأفضل
                    local_range = (varmax - varmin) * 0.01; % نطاق البحث المحلي
                    for d = 1:nx
                        % توليد حل محلي جديد
                        local_sol = gbest;
                        local_sol(d) = gbest(d) + local_range(d) * (2*rand - 1);
                        local_sol = max(min(local_sol, varmax), varmin); % تقييد ضمن الحدود
                        
                        % تقييم الحل المحلي
                        local_z = fobj(local_sol);
                        
                        % تحديث أفضل حل إذا كان الحل المحلي أفضل
                        if local_z < z_gbest
                            z_gbest = local_z;
                            gbest = local_sol;
                            % تحديث أفضل موضع شخصي أيضاً
                            z_pbest(j) = local_z;
                            pp_pbest(j,:) = local_sol;
                        end
                    end
                end
            end
        end
    end

    %% تكثيف البحث حول الحلول الواعدة (استراتيجية جديدة)
    if mod(it, 5) == 0 % تطبيق كل 5 تكرارات
        % اختيار أفضل حل للتكثيف حوله
        best_sol_idx = sorted_indices(1);
        best_sol = pp_pbest(best_sol_idx,:);
        
        % توليد حلول جديدة حول أفضل حل
        for i = 1:3 % توليد 3 حلول جديدة
            intensified_sol = best_sol + (2*rand(1,nx)-1) .* (varmax - varmin) * 0.05 * intensification_factor;
            intensified_sol = max(min(intensified_sol, varmax), varmin); % تقييد ضمن الحدود
            
            % تقييم الحل الجديد
            intensified_z = fobj(intensified_sol);
            
            % استبدال أسوأ حل في المجتمع إذا كان الحل الجديد أفضل
            worst_idx = sorted_indices(end-i+1);
            if intensified_z < z_pbest(worst_idx)
                pp(worst_idx,:) = intensified_sol;
                pp_pbest(worst_idx,:) = intensified_sol;
                z_pbest(worst_idx) = intensified_z;
                
                % تحديث أفضل موضع عام إذا كان الحل الجديد أفضل
                if intensified_z < z_gbest
                    z_gbest = intensified_z;
                    gbest = intensified_sol;
                end
            end
        end
    end

    %% تحديث الإحصائيات والمعاملات التكيفية
    % إعادة حساب الانحراف المعياري للمواضع المثلى الشخصية
    stdev2 = std(pp_pbest);
    max_stdev2 = max(stdev2);
    
    % إعادة حساب المتوسط والانحراف المعياري لقيم دالة الهدف
    ave = mean(z_pbest);
    stdev = std(z_pbest);

    % تحديث عدد أفضل الحلول k بشكل تكيفي محسن
    kbest = round(kbest_max - (kbest_max - kbest_min)*(it/maxit)^1.5);
    n_best = max(kbest, kbest_min); % ضمان عدم النزول عن الحد الأدنى

    %% إعادة حساب مؤشرات اللياقة بطريقة محسنة
    for j = 1:np
        prod = 1;
        for jj = 1:np
            if jj ~= j
                % دالة سيجمويد معدلة ومحسنة
                sigmoid_val = 1/(1+exp((-4*alpha)/(stdev*sqrt(exp(1)))*(z_pbest(jj)-ave)));
                prod = prod * sigmoid_val;
            end
        end
        fit(it,j) = prod;
    end

    % إعادة ترتيب الأفراد تنازلياً حسب قيم اللياقة
    [~, sorted_indices] = sort(fit(it,:), 'descend');

    % إعادة حساب مجموع قيم اللياقة لأفضل k أفراد
    sum1 = sum(fit(it,sorted_indices(1:n_best)));

    % تحديث قيم دالة الهدف ومواضع أفضل k أفراد
    for j = 1:n_best
        idx = sorted_indices(j);
        z_kbest(j) = z_pbest(idx);
        pp_kbest(j,:) = pp_pbest(idx,:);
    end

    %% إعادة حساب موجهات الحركة لكل فرد بطريقة محسنة
    for j = 1:np
        pp_guide(j,:) = zeros(1,nx);
        
        % تعديل نسبة تأثير أفضل موضع عام (gbest) مع تقدم التكرارات
        gbest_influence = 0.2 + 0.3 * (it/maxit); % يزداد مع تقدم التكرارات
        
        for jj = 1:n_best
            idx = sorted_indices(jj);
            if idx ~= j
                % حساب عامل التأثير الديناميكي (DFI) المحسن
                rank_factor = 1 + (n_best - jj) / n_best;
                DFI(jj) = rank_factor * fit(it,idx)/(sum1 + epsilon);
                
                % تحديث موجه الحركة للفرد j
                pp_guide(j,:) = pp_guide(j,:) + DFI(jj).*pp_kbest(jj,:);
            end
        end
        
        % دمج تأثير أفضل موضع عام (gbest)
        pp_guide(j,:) = (1 - gbest_influence) * pp_guide(j,:) + gbest_influence * gbest;
    end

    % تحديث أفضل قيمة هدف وموضع في التكرار الحالي
    [z_optimal(it), best_idx] = min(z_pbest);
    optimal_pos(it,:) = pp_pbest(best_idx,:);
    z_iter(it) = z_optimal(it);

    % طباعة معلومات التكرار الحالي كل 5 تكرارات لتقليل المخرجات
    if mod(it, 5) == 0 || it == maxit
        fprintf('التكرار %3d: أفضل IAE = %.6f, معاملات PID = [%.4f, %.4f, %.4f]\n', ...
            it, z_optimal(it), optimal_pos(it,1), optimal_pos(it,2), optimal_pos(it,3));
    end
end

%% النتيجة النهائية
z_final = z_optimal(maxit);
pos_final = optimal_pos(maxit,:);

% طباعة النتيجة النهائية بتنسيق واضح
fprintf('\n=== النتيجة النهائية ===\n');
fprintf('أفضل IAE = %.6f\n', z_final);
fprintf('معاملات PID المثلى: Kp = %.4f, Ki = %.4f, Kd = %.4f\n', ...
    pos_final(1), pos_final(2), pos_final(3));

end
