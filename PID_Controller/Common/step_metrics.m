function metrics = step_metrics(sys, t)
% STEP_METRICS Computes step response performance metrics
%   Input:
%       sys : closed-loop transfer function
%       t   : time vector
%   Output:
%       metrics : struct with rise time, settling time, overshoot, steady-state error

    % Desired reference (unit step)
    r = ones(size(t));
    y = lsim(sys, r, t);

    % Steady-state value (assume last value is steady-state)
    y_final = y(end);
    e_ss = abs(1 - y_final);

    % Rise time: time to go from 10% to 90%
    y_10 = 0.1 * y_final;
    y_90 = 0.9 * y_final;
    idx_10 = find(y >= y_10, 1, 'first');
    idx_90 = find(y >= y_90, 1, 'first');
    rise_time = t(idx_90) - t(idx_10);

    % Overshoot
    peak = max(y);
    overshoot = max(0, (peak - y_final) / y_final) * 100;

    % Settling time (2% criterion)
    tol = 0.02 * y_final;
    idx_settle = find(abs(y - y_final) > tol, 1, 'last');
    if isempty(idx_settle)
        settling_time = 0;
    else
        settling_time = t(idx_settle);
    end

    % Output struct
    metrics.rise_time = rise_time;
    metrics.settling_time = settling_time;
    metrics.overshoot = overshoot;
    metrics.steady_state_error = e_ss;
end