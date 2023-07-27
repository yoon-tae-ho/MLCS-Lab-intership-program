clear; close all; clc
%%
% constants
V_x = 0.15; % m/s
R_wheel = 0.03; % m
R_robot = 0.075; % m
K_d = 2;
K_p = 1;
y_init = [0.1, 0, 0]'; % [y, dy, e_psi]
tspan = [0, 50]; % for 50 seconds

% 
[t, Y] = ode45(@(t,X) system_fun(t, X, V_x, K_d, K_p), tspan, y_init);

y_r = 0.05 * cos(t / pi);
dy_r = -0.05 / pi * sin(t / pi);

W_z = (1/V_x)*(-K_d*(Y(:,2) - dy_r) - K_p*(Y(:,1) - y_r));
W_R = V_x/R_wheel + (R_robot/(2*R_wheel))*W_z;
W_L = V_x/R_wheel - (R_robot/(2*R_wheel))*W_z;

% plot
figure(1)
plot(t,Y(:,1)); hold on; plot(t,y_r);
legend('y','y_r');
xlabel('time (s)');
ylabel('y (m)');
title('y and y_r')

figure(2)
plot(t, Y(:,3))
legend('e_{psi}');
xlabel('time (s)');
ylabel('e_{psi} (rad)');
title('e_{psi}');

figure(3)
plot(t,W_R); hold on; plot(t,W_L);
legend('W_R','W_L');
xlabel('time (s)');
ylabel('W_{wheel} (rad/s)');
title('W_{wheel}');