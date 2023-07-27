function dY = system_fun(t,Y, V_x, K_d, K_p)
    dY = zeros(3,1);
    y = Y(1);
    dy = Y(2);
    % e_psi = Y(3);
    
    y_r = 0.05 * cos(t / pi);
    dy_r = -0.05 / pi * sin(t / pi);
    ddy_r = -0.05 / pi / pi * cos(t / pi);
    
    temp = -K_d * (dy - dy_r) - K_p * (y - y_r);
    dY(1) = dy;
    dY(2) = temp + ddy_r;
    dY(3) = (1 / V_x) * temp;
end