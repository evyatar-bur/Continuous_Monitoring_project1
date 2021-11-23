function [zero_cross] = Our_zero_crossing(Window)
% Count zero crossing in window

pos = Window>0;

changes = xor(pos(1:end-1),pos(2:end));

zero_cross = sum(changes) + 1;

end