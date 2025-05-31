%___________________________________________________________________%
%               GMO: محسّن المتوسط الهندسي                           %
%                                                                   %
% تم تطويره في MATLAB R2016b                                        %% البريد الإلكتروني: elimam_mohamed@univ-biskra.dz                   %
%                medlimame9@gmail.com                               %
%                                                                   %
% الصفحة الشخصية: https://www.linkedin.com/in/mohamed-el-imam-434321335/ %
%                                                                   %
%                                                                   %

% الورقة البحثية الرئيسية:                                          %
% Rezaei, F., Safavi, H.R., Abd Elaziz, M. et al.                   %
% GMO: محسّن المتوسط الهندسي لحل المشكلات الهندسية.                   %
% Soft Comput (2023). https://doi.org/10.1007/s00500-023-08202-z    %
%___________________________________________________________________%

% GMO: محسّن المتوسط الهندسي                                                                 
% الكود الكامل والنهائي لدالة GMO (محسّن المتوسط الهندسي)
function [z_iter, z_final, pos_final] = GMO(np, nx, maxit, varmax, varmin, velmax, velmin, epsilon, fobj)
%% GMO - خوارزمية محسّن المتوسط الهندسي لتحسين معاملات متحكم PID
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

% معاملات استراتيجية k-best التكيفية
kbest_max = np;              % الحد الأقصى لعدد أفضل الحلول (يبدأ بكل المجتمع)
kbest_min = 2;               % الحد الأدنى لعدد أفضل الحلول (ينتهي بأفضل حلين)

%% عملية التهيئة الأولية
it = 1;  % عداد التكرارات
% استدعاء دالة التهيئة لإنشاء المواضع والسرعات الأولية
[pp, pv] = Initialization(np, nx, varmax, varmin, velmax, velmin);

%% تقييم دالة الهدف للمجتمع الأولي
for j = 1:np
    % حساب قيمة دالة الهدف (IAE) للفرد j
    z = fobj(pp(j,:));
    % تخزين قيمة دالة الهدف وموضع الفرد كأفضل موضع شخصي أولي
    z_pbest(j) = z;
    pp_pbest(j,:) = pp(j,:);
end

% حساب الانحراف المعياري للمواضع المثلى الشخصية (يستخدم في آلية الطفرة)
stdev2 = std(pp_pbest);
max_stdev2 = max(stdev2);
% حساب المتوسط والانحراف المعياري لقيم دالة الهدف (تستخدم في حساب اللياقة)
ave = mean(z_pbest);
stdev = std(z_pbest);

% تحديد عدد أفضل الحلول k بشكل تكيفي (يتناقص خطياً مع التكرارات)
kbest = round(kbest_max - (kbest_max - kbest_min)*(it/maxit));
n_best = kbest;

%% حساب مؤشرات اللياقة باستخدام دالة سيجمويد معدلة
% المعادلة (7) في الورقة البحثية الأصلية
for j = 1:np
    index(j) = j;
    prod = 1;
    for jj = 1:np
        if jj ~= j
            % دالة سيجمويد معدلة لحساب احتمالية كون الفرد jj أفضل من المتوسط
            prod = prod * (1/(1+exp((-4)/(stdev*sqrt(exp(1)))*(z_pbest(jj)-ave)))); 
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

%% حساب موجهات الحركة لكل فرد باستخدام عوامل التأثير الديناميكية
% المعادلة (8) في الورقة البحثية الأصلية
for j = 1:np
    pp_guide(j,:) = zeros(1,nx);
    for jj = 1:n_best
        idx = sorted_indices(jj);
        if idx ~= j
            % حساب عامل التأثير الديناميكي (DFI) للفرد idx
            DFI(jj) = fit(it,idx)/(sum1 + epsilon);
            % تحديث موجه الحركة للفرد j
            pp_guide(j,:) = pp_guide(j,:) + DFI(jj).*pp_kbest(jj,:);
        end
    end
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
    % معامل الوزن التكيفي (يتناقص خطياً من 1 إلى 0)
    w = 1 - (it/maxit);

    %% تحديث المواضع والسرعات لكل فرد
    for j = 1:np
        % توليد متجه طفرة باستخدام موجه الحركة والانحراف المعياري
        % المعادلة (9) في الورقة البحثية الأصلية
        mutant = pp_guide(j,:) + w*randn(1,nx).*(max_stdev2 - stdev2);
        
        % تحديث السرعة باستخدام معادلة السرعة المعدلة
        % المعادلة (10) في الورقة البحثية الأصلية
        pv(j,:) = w*pv(j,:) + (1 + (2*rand(1,nx)-1)*w).*(mutant - pp(j,:));

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
        end
    end

    %% تحديث الإحصائيات والمعاملات التكيفية
    % إعادة حساب الانحراف المعياري للمواضع المثلى الشخصية
    stdev2 = std(pp_pbest);
    max_stdev2 = max(stdev2);
    
    % إعادة حساب المتوسط والانحراف المعياري لقيم دالة الهدف
    ave = mean(z_pbest);
    stdev = std(z_pbest);

    % تحديث عدد أفضل الحلول k بشكل تكيفي
    kbest = round(kbest_max - (kbest_max - kbest_min)*(it/maxit));
    n_best = kbest;

    %% إعادة حساب مؤشرات اللياقة
    for j = 1:np
        prod = 1;
        for jj = 1:np
            if jj ~= j
                prod = prod * (1/(1+exp((-4)/(stdev*sqrt(exp(1)))*(z_pbest(jj)-ave))));
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

    %% إعادة حساب موجهات الحركة لكل فرد
    for j = 1:np
        pp_guide(j,:) = zeros(1,nx);
        for jj = 1:n_best
            idx = sorted_indices(jj);
            if idx ~= j
                DFI(jj) = fit(it,idx)/(sum1 + epsilon);
                pp_guide(j,:) = pp_guide(j,:) + DFI(jj).*pp_kbest(jj,:);
            end
        end
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
