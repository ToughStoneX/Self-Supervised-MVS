function [] = parsave(name, x, BaseEval)
%save x,y in dir
% so I can save in parfor loop
% save(name, x, t);
save(name, x);
end