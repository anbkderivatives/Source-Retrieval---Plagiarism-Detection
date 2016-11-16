function [AvgVector]=AvgVector(Vector)

AvgVector = Vector(1);

for i = 2:length(Vector)
    AvgVector(end+1)=(AvgVector(end)+Vector(i))/2;
    %disp(AvgVector(end));
end

