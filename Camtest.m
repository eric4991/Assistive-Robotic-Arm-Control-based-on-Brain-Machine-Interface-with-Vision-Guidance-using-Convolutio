clear all; close all; clc;

py.SharedDemo_6_VisionTrial_1.getPerspectiveMine();

intentionList=[3];
for i=1:length(intentionList)
    py.SharedDemo_6_VisionTrial_1.mainRunning(intentionList(i))
    %pause(0)
end