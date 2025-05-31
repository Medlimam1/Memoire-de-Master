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
run = 30;           % ��� ���� ����� ���������� (��������� ��������)

for f = 1:length(F_list)
    F_name = F_list{f};

    % ����� ������ ����� �������
    [LB, UB, Dim, F_obj] = Get_Fu(F_name);

    % ������� ������ ����� �� �����
    best_values = NaN(run, 1);   % ���� ���� ������
    all_curves  = NaN(M_Iter, run); % ������� ������� ��� �����: [M_Iter � run]

    for nrun = 1:run
        % ����� AOA ��� �����
        [Best_FF, Best_P, Conv_curve] = AOA(Solution_no, M_Iter, LB, UB, Dim, F_obj);

        % ��� ������� ��� �� �����
        disp(['=== ����� ��� (', num2str(nrun), ') ������ ', F_name, ' ===']);
        disp(['The best-obtained solution by AOA is: ', num2str(Best_P)]);
        disp(['The best optimal value of the objective function found by AOA is: ', num2str(Best_FF)]);

        % ��� ���� ���� ������ �� ��� �������
        best_values(nrun) = Best_FF;

        % �� ��� ���� ����� Conv_curve ������ �� [1 � M_Iter], ������
        if size(Conv_curve, 1) > 1 && size(Conv_curve, 2) == 1
            Conv_curve = Conv_curve'; 
        end

        % ����� ����� �������
        all_curves(:, nrun) = Conv_curve(:);
    end

    % ���� ����� ������� ������ ����� �������� ��� �� ���������
    avg_val = mean(best_values);
    std_val = std(best_values);

    % ��� ����� ������� (�����) ��� ���������
    mean_curve = mean(all_curves, 2, 'omitnan');  % ��� ���� ��� NaN
    figure('Position', [454, 445, 694, 297]);

    % subplot(1,2,1): ��� ���� ������ (Parameter space)
    subplot(1,2,1);
    func_plot(F_name);
    title('Parameter space');
    xlabel('x_1');
    ylabel('x_2');
    zlabel([F_name, '(x_1, x_2)']);

    % subplot(1,2,2): ��� ����� ����� ������� ��� run
    subplot(1,2,2);
    semilogy(mean_curve, 'Color','r', 'LineWidth',2);
    title('Convergence curve (Avg)');
    xlabel('Iteration #');
    ylabel('Best fitness function');
    legend('AOA');
    axis tight;

    % ��� ������� �� ��� mat (��� ����)
    % - best_values: ���� 10 ��� (�� �����)
    % - all_curves : ������� ������� ����� ���������
    % - avg_val, std_val : ��������
    save(['Results_', F_name, '_AOA.mat'], ...
         'best_values','all_curves','avg_val','std_val');

    % ��� ������� �������� ��� �� ������
    disp(['===== ����� ������ ', F_name, ' ��� ', num2str(run), ' ������� =====']);
    disp(['����� (avg_val) = ', num2str(avg_val), ...
          ', ������ ������ (std_val) = ', num2str(std_val)]);
    disp(['���� ���� �� ��� ���� ��������� = ', num2str(min(best_values))]);
end

disp('All test functions have been processed successfully using AOA!');

