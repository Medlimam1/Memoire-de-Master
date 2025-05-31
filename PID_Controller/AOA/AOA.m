function [Best_FF,Best_P,Conv_curve]=AOA(N,M_Iter,LB,UB,Dim,F_obj)
display('AOA Working');
%Two variables to keep the positions and the fitness value of the best-obtained solution

Best_P=zeros(1,Dim);
Best_FF=inf;
Conv_curve=zeros(1,M_Iter);

%Initialize the positions of solution
X=initialization(N,Dim,UB,LB);
Xnew=X;
Ffun=zeros(1,size(X,1));% (fitness values)
Ffun_new=zeros(1,size(Xnew,1));% (fitness values)

MOP_Max=1;
MOP_Min=0.2;
C_Iter=1;
Alpha=5;
Mu=0.499;

% Evaluate initial solutions
for i=1:size(X,1)
    Ffun(1,i)=F_obj(X(i,:));  %Calculate the fitness values of solutions
    if Ffun(1,i)<Best_FF
        Best_FF=Ffun(1,i);
        Best_P=X(i,:);
    end
end

%Main loop
while C_Iter < M_Iter+1
    MOP = 1 - ((C_Iter)^(1/Alpha) / (M_Iter)^(1/Alpha)); % Probability Ratio 
    MOA = MOP_Min + C_Iter*((MOP_Max-MOP_Min)/M_Iter);    % Accelerated function
   
    % Update the Position of solutions
    for i=1:size(X,1)   
        for j=1:size(X,2)
           r1=rand(); % <-- «” Œœ«„ rand »œ·« „‰ randn · ﬁ·Ì· «·ﬁ›“« 
            if (size(LB,2)==1)
                if r1<MOA
                    r2=rand();
                    if r2>0.5
                        Xnew(i,j)=Best_P(1,j)/(MOP+eps)*((UB-LB)*Mu+LB);
                    else
                        Xnew(i,j)=Best_P(1,j)*MOP*((UB-LB)*Mu+LB);
                    end
                else
                    r3=rand();
                    if r3>0.5
                        Xnew(i,j)=Best_P(1,j)-MOP*((UB-LB)*Mu+LB);
                    else
                        Xnew(i,j)=Best_P(1,j)+MOP*((UB-LB)*Mu+LB);
                    end
                end
            else
                % ≈–« ﬂ«‰  «·ÕœÊœ „Œ ·›… ·ﬂ· „ €Ì—
                r1=rand(); % √Ì÷« ‰” Œœ„ rand Â‰«
                if r1<MOA
                    r2=rand();
                    if r2>0.5
                        Xnew(i,j)=Best_P(1,j)/(MOP+eps)*((UB(j)-LB(j))*Mu+LB(j));
                    else
                        Xnew(i,j)=Best_P(1,j)*MOP*((UB(j)-LB(j))*Mu+LB(j));
                    end
                else
                    r3=rand();
                    if r3>0.5
                        Xnew(i,j)=Best_P(1,j)-MOP*((UB(j)-LB(j))*Mu+LB(j));
                    else
                        Xnew(i,j)=Best_P(1,j)+MOP*((UB(j)-LB(j))*Mu+LB(j));
                    end
                end
            end
        end
        
        % ÷»ÿ «·ÕœÊœ
        Flag_UB = Xnew(i,:)>UB;
        Flag_LB = Xnew(i,:)<LB;
        Xnew(i,:) = (Xnew(i,:).*(~(Flag_UB+Flag_LB)))+UB.*Flag_UB+LB.*Flag_LB;

        % Õ”«» œ«·… «·Âœ›
        Ffun_new(1,i) = F_obj(Xnew(i,:));
        % ﬁ»Ê· «· Õ”Ì‰
        if Ffun_new(1,i) < Ffun(1,i)
            X(i,:)=Xnew(i,:);
            Ffun(1,i)=Ffun_new(1,i);
        end
        %  ÕœÌÀ Best_FF Ê Best_P
        if Ffun(1,i) < Best_FF
            Best_FF=Ffun(1,i);
            Best_P=X(i,:);
        end
    end
    
    % ﬁ»·  Œ“Ì‰ Best_FF ›Ì «·„‰Õ‰Ï,  √ﬂœ „‰ ⁄œ„ Ê’Ê·Â« ·‹0  Õ 
    if Best_FF<1e-308
        Best_FF = 1e-308;  % „‰⁄ «·«‰Õœ«— ·’›—
    end

    % Update the convergence curve
    Conv_curve(C_Iter)=Best_FF;
    
    % Print every 50 iterations
    if mod(C_Iter,50)==0
        disp(['At iteration ', num2str(C_Iter), ...
              ' the best solution fitness is ', num2str(Best_FF)]);
    end

    % ⁄—÷ «·„‰Õ‰Ï √À‰«¡ «· ‘€Ì· («Œ Ì«—Ì)
    plot(1:C_Iter, Conv_curve(1:C_Iter), 'b-');
    drawnow;
    
    C_Iter = C_Iter + 1;
end




