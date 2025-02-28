%% Perform prox operator:   min ||x||_1 + (1/2t)||x-y||^2

function x = proxL1(y,t)
x = sign(y) .* max(abs(y)-t,0);
end
