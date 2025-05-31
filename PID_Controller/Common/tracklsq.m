function J = tracklsq(K)
% TRACKLSQ: Objective function for PID tuning (IAE minimization)

    K = reshape(K, 1, []);
    if numel(K) ~= 3 || any(K < 0) || any(isnan(K)) || any(isinf(K))
        J = 1e6;
        return;
    end

    % تعريف نموذج المحرك
    Ra = 0.4; La = 2.7; Jm = 0.0004; D = 0.0022; Kt = 0.015; Kb = 0.05;
    den = [Jm*La, Jm*Ra + D*La, D*Ra + Kt*Kb];
    G = tf(Kt, den);

    % توليد المتحكم
    C = pid(K(1), K(2), K(3));
    sys_cl = feedback(C * G, 1);

    % إعداد المحاكاة
    Tfinal = 5; dt = 0.001;
    t = 0:dt:Tfinal;
    r = ones(size(t));

    % المحاكاة وضمان البنية
    y = lsim(sys_cl, r(:), t(:));
    e = r(:) - y(:);
    J = trapz(t(:), abs(e));
end
