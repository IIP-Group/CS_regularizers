%% Perform prox operator:   min ||x||_inf + (1/2t)||x-w||^2
function [ xk ] = proxInf( w,t )
    N = length(w);
    wabs = abs(w);
    ws = (cumsum(sort(wabs,'descend'))- t)./(1:N)';
    alphaopt = max(ws);
    if alphaopt>0 
      xk = min(wabs,alphaopt).*sign(w); % truncation step
    else
      xk = zeros(size(w)); % if t is big, then solution is zero
    end       
end