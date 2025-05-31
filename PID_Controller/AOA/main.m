%_______________________________________________________________________________________
%  The Arithmetic Optimization Algorithm (AOA) source codes demo version 1.0                  
%                                                                                       
%  Developed in MATLAB R2015a (7.13)                                                    
%                                                                                       
%  Authors: Laith Abualigah, Ali Diabat, Seyedali Mirjalili, Mohamed Abd Elaziz, & Amir H. Gandomi                       
%                                                                                       
%  E-Mail: Aligah.2020@gmail.com  (Laith Abualigah)                                               
%  Homepage:                                                                       
%  1- https://scholar.google.com/citations?user=39g8fyoAAAAJ&hl=en               
%  2- https://www.researchgate.net/profile/Laith_Abualigah                       
%                                                                                       
%  Main paper:   The Arithmetic Optimization Algorithm
%  Reference: Abualigah, L., Diabat, A., Mirjalili, S., Abd Elaziz, M., and Gandomi, A. H. (2021). 
%  The Arithmetic Optimization Algorithm. Computer Methods in Applied Mechanics and Engineering.
%
%_______________________________________________________________________________________
clear;
clc;
close all;

Solution_no = 50;   % Number of search solutions
F_list = {'F8', 'F9', 'F10', 'F11', 'F12', 'F14', 'F22', 'F23'};    % List of test functions
M_Iter = 2000;      % Maximum number of iterations
run = 30;           % ⁄œœ „—«   ‘€Ì· «·ŒÊ«—“„Ì… («· ﬂ—«—«  «·Œ«—ÃÌ…)

for f = 1:length(F_list)
    F_name = F_list{f};

    %  Õ„Ì· «·œ«·… «·Âœ› ÊÕœÊœÂ«
    [LB, UB, Dim, F_obj] = Get_Fu(F_name);

    % „’›Ê›«  · Œ“Ì‰ ‰ «∆Ã ﬂ·  ‘€Ì·
    best_values = NaN(run, 1);   % √›÷· ﬁÌ„… ‰Â«∆Ì…
    all_curves  = NaN(M_Iter, run); % „‰Õ‰Ì«  «· ﬁ«—» ·ﬂ·  ‘€Ì·: [M_Iter ◊ run]

    for nrun = 1:run
        %  ‘€Ì· AOA „—… Ê«Õœ…
        [Best_FF, Best_P, Conv_curve] = AOA(Solution_no, M_Iter, LB, UB, Dim, F_obj);

        % ⁄—÷ «·‰ «∆Ã ﬂ„« ›Ì «·√’·
        disp(['===  ‘€Ì· —ﬁ„ (', num2str(nrun), ') ··œ«·… ', F_name, ' ===']);
        disp(['The best-obtained solution by AOA is: ', num2str(Best_P)]);
        disp(['The best optimal value of the objective function found by AOA is: ', num2str(Best_FF)]);

        % Õ›Ÿ √›÷· ﬁÌ„… ‰Â«∆Ì… ›Ì Â–« «· ‘€Ì·
        best_values(nrun) = Best_FF;

        % ›Ì Õ«· ﬂ«‰  √»⁄«œ Conv_curve „Œ ·›… ⁄‰ [1 ◊ M_Iter], ‰’ÕÕÂ«
        if size(Conv_curve, 1) > 1 && size(Conv_curve, 2) == 1
            Conv_curve = Conv_curve'; 
        end

        %  Œ“Ì‰ „‰Õ‰Ï «· ﬁ«—»
        all_curves(:, nrun) = Conv_curve(:);
    end

    % Õ”«» „ Ê”ÿ Ê«‰Õ—«› „⁄Ì«—Ì ··ﬁÌ„ «·‰Â«∆Ì… ⁄»— ﬂ· «· ‘€Ì·« 
    avg_val = mean(best_values);
    std_val = std(best_values);

    % —”„ „‰Õ‰Ï «· ﬁ«—» („ Ê”ÿ) ·ﬂ· «· ﬂ—«—« 
    mean_curve = mean(all_curves, 2, 'omitnan');  % ≈–« ŸÂ—  ﬁÌ„ NaN
    figure('Position', [454, 445, 694, 297]);

    % subplot(1,2,1): —”„ ›÷«¡ «·œ«·… (Parameter space)
    subplot(1,2,1);
    func_plot(F_name);
    title('Parameter space');
    xlabel('x_1');
    ylabel('x_2');
    zlabel([F_name, '(x_1, x_2)']);

    % subplot(1,2,2): —”„ „ Ê”ÿ „‰Õ‰Ï «· ﬁ«—» ⁄»— run
    subplot(1,2,2);
    semilogy(mean_curve, 'Color','r', 'LineWidth',2);
    title('Convergence curve (Avg)');
    xlabel('Iteration #');
    ylabel('Best fitness function');
    legend('AOA');
    axis tight;

    % Õ›Ÿ «·‰ «∆Ã ›Ì „·› mat (·ﬂ· œ«·…)
    % - best_values: √›÷· 10 ﬁÌ„ (ﬂ·  ‘€Ì·)
    % - all_curves : „‰Õ‰Ì«  «· ﬁ«—» ·ﬂ«›… «· ‘€Ì·« 
    % - avg_val, std_val : ≈Õ’«∆Ì« 
    save(['Results_', F_name, '_AOA.mat'], ...
         'best_values','all_curves','avg_val','std_val');

    % ⁄—÷ «·‰ «∆Ã «·‰Â«∆Ì… »⁄œ ﬂ· «·œ«·…
    disp(['===== ‰ «∆Ã «·œ«·… ', F_name, ' »⁄œ ', num2str(run), '  ‘€Ì·«  =====']);
    disp(['„ Ê”ÿ (avg_val) = ', num2str(avg_val), ...
          ', «‰Õ—«› „⁄Ì«—Ì (std_val) = ', num2str(std_val)]);
    disp(['√›÷· ﬁÌ„… „‰ »Ì‰ Ã„Ì⁄ «· ‘€Ì·«  = ', num2str(min(best_values))]);
end

disp('All test functions have been processed successfully using AOA!');

