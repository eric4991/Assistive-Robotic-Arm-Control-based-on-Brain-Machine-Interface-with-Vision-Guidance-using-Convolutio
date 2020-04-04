%% @ written by Kyung-Hwan Shim
% Construction Date: 20190620
clear all; close all; clc;

modIdx=1;
hzIdx=2;

hz=[100,250,1000];
nameList={'GIGA_20190708_eslee','GIGA_20190708_hjwang','GIGA_20190708_jmlee','GIGA_20190710_dslim','GIGA_20190712_wjyun'};
SessionList={'session1','session2','session3'};
task={'reaching','multigrasp','twist'};
mode={'MI','realMove'};

ref='FCz';
channelMatrix={'F3','F1','Fz','F2','F4';
    'FC3','FC1','FCz','FC2','FC4';
    'C3','C1', 'Cz', 'C2', 'C4';
    'CP3','CP1','CPz','CP2','CP4';
    'P3','P1','Pz','P2','P4'};

for sub=1:length(nameList)
    ival=[0 3000];
    [cntGrasp,mrkGrasp,mntGrasp]=eegfile_loadMatlab(['./1_ConvertedData/' mode{modIdx} '/' num2str(hz(hzIdx)) '/' nameList{sub} '_multigrasp_' mode{modIdx}]);
    [cntReach,mrkReach,mntReach]=eegfile_loadMatlab(['./1_ConvertedData/' mode{modIdx} '/' num2str(hz(hzIdx)) '/' nameList{sub} '_reaching_' mode{modIdx}]);
    [cntTwist,mrkTwist,mntTwist]=eegfile_loadMatlab(['./1_ConvertedData/' mode{modIdx} '/' num2str(hz(hzIdx)) '/' nameList{sub} '_twist_' mode{modIdx}]);
    cntGrasp=proc_filtButter(cntGrasp,5,[4 40]);
    cntTwist=proc_filtButter(cntTwist,5,[4 40]);
    cntReach=proc_filtButter(cntReach,5,[4 40]);
    epoGrasp=cntToEpo(cntGrasp,mrkGrasp,ival);
    epoTwist=cntToEpo(cntTwist,mrkTwist,ival);
    epoReach=cntToEpo(cntReach,mrkReach,ival);
    
    epoGrasp=proc_selectChannels(epoGrasp,{'F3','F1','Fz','F2','F4',...
        'FC3','FC1','FCz','FC2','FC4',...
        'C3','C1', 'Cz', 'C2', 'C4', ...
        'CP3','CP1','CPz','CP2','CP4',...
        'P3','P1','Pz','P2','P4'});
    epoTwist=proc_selectChannels(epoTwist,{'F3','F1','Fz','F2','F4',...
        'FC3','FC1','FCz','FC2','FC4',...
        'C3','C1', 'Cz', 'C2', 'C4', ...
        'CP3','CP1','CPz','CP2','CP4',...
        'P3','P1','Pz','P2','P4'});
    epoReach=proc_selectChannels(epoReach,{'F3','F1','Fz','F2','F4',...
        'FC3','FC1','FCz','FC2','FC4',...
        'C3','C1', 'Cz', 'C2', 'C4', ...
        'CP3','CP1','CPz','CP2','CP4',...
        'P3','P1','Pz','P2','P4'});
    %% Retrieve Trial amounts
    epoGrasp=proc_selectClasses(epoGrasp,{'Grasp'});
    epoTwist=proc_selectClasses(epoTwist,{'LeftTwist'});
    epoLeft=proc_selectClasses(epoReach,{'Left'});
    epoRight=proc_selectClasses(epoReach,{'Right'});
    epoForward=proc_selectClasses(epoReach,{'Forward'});
    epoBackward=proc_selectClasses(epoReach,{'Backward'});
    epoUp=proc_selectClasses(epoReach,{'Up'});
    epoDown=proc_selectClasses(epoReach,{'Down'});
    epoRest=proc_selectClasses(epoReach,{'Rest'});
    
    minTrial=min([size(epoGrasp.x,3),size(epoTwist.x,3),size(epoLeft.x,3),size(epoRight.x,3),...
        size(epoForward.x,3),size(epoBackward.x,3),size(epoUp.x,3),size(epoDown.x,3),size(epoRest.x,3)]);
    
    %% Data shuffling
    epoGrasp.x=datasample(epoGrasp.x,minTrial,3,'Replace',false);
    epoTwist.x=datasample(epoTwist.x,minTrial,3,'Replace',false);
    epoLeft.x=datasample(epoLeft.x,minTrial,3,'Replace',false);
    epoRight.x=datasample(epoRight.x,minTrial,3,'Replace',false);
    epoForward.x=datasample(epoForward.x,minTrial,3,'Replace',false);
    epoBackward.x=datasample(epoBackward.x,minTrial,3,'Replace',false);
    epoUp.x=datasample(epoUp.x,minTrial,3,'Replace',false);
    epoDown.x=datasample(epoDown.x,minTrial,3,'Replace',false);
    epoRest.x=datasample(epoRest.x,minTrial,3,'Replace',false);
    
    LeftData=epoLeft.x;LeftLabel=epoLeft.y;
    RightData=epoRight.x;RightLabel=epoRight.y;
    ForwardData=epoForward.x;ForwardLabel=epoForward.y;
    BackwardData=epoBackward.x;BackwardLabel=epoBackward.y;
    UpData=epoUp.x;UpLabel=epoUp.y;
    DownData=epoDown.x;DownLabel=epoDown.y;
    GraspData=epoGrasp.x;GraspLabel=epoGrasp.y;
    TwistData=epoTwist.x;TwistLabel=epoTwist.y;
    RestData=epoRest.x;RestLabel=epoRest.y;
    
    save(["./2_2DData/"+mode{modIdx}+'/'+num2str(hz(hzIdx))+"/"+"Left"+"/sub"+string(sub)],'LeftData');
    save(["./2_2DData/"+mode{modIdx}+'/'+num2str(hz(hzIdx))+"/"+"Right"+"/sub"+string(sub)],'RightData');
    save(["./2_2DData/"+mode{modIdx}+'/'+num2str(hz(hzIdx))+"/"+"Forward"+"/sub"+string(sub)],'ForwardData');
    save(["./2_2DData/"+mode{modIdx}+'/'+num2str(hz(hzIdx))+"/"+"Backward"+"/sub"+string(sub)],'BackwardData');
    save(["./2_2DData/"+mode{modIdx}+'/'+num2str(hz(hzIdx))+"/"+"Up"+"/sub"+string(sub)],'UpData');
    save(["./2_2DData/"+mode{modIdx}+'/'+num2str(hz(hzIdx))+"/"+"Down"+"/sub"+string(sub)],'DownData');
    save(["./2_2DData/"+mode{modIdx}+'/'+num2str(hz(hzIdx))+"/"+"Grasp"+"/sub"+string(sub)],'GraspData');
    save(["./2_2DData/"+mode{modIdx}+'/'+num2str(hz(hzIdx))+"/"+"Twist"+"/sub"+string(sub)],'TwistData');
    save(["./2_2DData/"+mode{modIdx}+'/'+num2str(hz(hzIdx))+"/"+"Rest"+"/sub"+string(sub)],'RestData');
    
    %% 2D Spatial conversion
    
    for xMat=1:size(channelMatrix,1)
        tmpLeft{xMat}=proc_selectChannels(epoLeft,channelMatrix{xMat,:});
        tmpRight{xMat}=proc_selectChannels(epoRight,channelMatrix{xMat,:});
        tmpForward{xMat}=proc_selectChannels(epoForward,channelMatrix{xMat,:});
        tmpBackward{xMat}=proc_selectChannels(epoBackward,channelMatrix{xMat,:});
        tmpUp{xMat}=proc_selectChannels(epoUp,channelMatrix{xMat,:});
        tmpDown{xMat}=proc_selectChannels(epoDown,channelMatrix{xMat,:});
        tmpGrasp{xMat}=proc_selectChannels(epoGrasp,channelMatrix{xMat,:});
        tmpTwist{xMat}=proc_selectChannels(epoTwist,channelMatrix{xMat,:});
        tmpRest{xMat}=proc_selectChannels(epoRest,channelMatrix{xMat,:});
    end
    MatFinLeft.x=zeros(size(tmpLeft{1,1}.x,1),size(channelMatrix,1),size(channelMatrix,2), size(tmpLeft{1,1}.x,3));
    MatFinRight.x=zeros(size(tmpRight{1,1}.x,1),size(channelMatrix,1),size(channelMatrix,2), size(tmpRight{1,1}.x,3));
    MatFinForward.x=zeros(size(tmpForward{1,1}.x,1),size(channelMatrix,1),size(channelMatrix,2), size(tmpForward{1,1}.x,3));
    MatFinBackward.x=zeros(size(tmpBackward{1,1}.x,1),size(channelMatrix,1),size(channelMatrix,2), size(tmpBackward{1,1}.x,3));
    MatFinUp.x=zeros(size(tmpUp{1,1}.x,1),size(channelMatrix,1),size(channelMatrix,2), size(tmpUp{1,1}.x,3));
    MatFinDown.x=zeros(size(tmpDown{1,1}.x,1),size(channelMatrix,1),size(channelMatrix,2), size(tmpDown{1,1}.x,3));
    MatFinGrasp.x=zeros(size(tmpGrasp{1,1}.x,1),size(channelMatrix,1),size(channelMatrix,2), size(tmpGrasp{1,1}.x,3));
    MatFinTwist.x=zeros(size(tmpTwist{1,1}.x,1),size(channelMatrix,1),size(channelMatrix,2), size(tmpTwist{1,1}.x,3));
    MatFinRest.x=zeros(size(tmpRest{1,1}.x,1),size(channelMatrix,1),size(channelMatrix,2), size(tmpRest{1,1}.x,3));
    
    try find(strcmp(channelMatrix,ref));
        [refPosX,refPosY]=find(strcmp(channelMatrix,ref));
        for concatMat=1:length(tmpLeft)
            if concatMat==refPosX
                idxMat=1;
                for concatMatY=1:size(tmpLeft{1,concatMat}.x,2)
                    if concatMatY==refPosY
                        MatFinLeft.x(:,concatMat,idxMat,:)=zeros(size(tmpLeft{1,concatMat}.x,1),size(tmpLeft{1,concatMat}.x,3));
                        MatFinRight.x(:,concatMat,idxMat,:)=zeros(size(tmpRight{1,concatMat}.x,1),size(tmpRight{1,concatMat}.x,3));
                        MatFinForward.x(:,concatMat,idxMat,:)=zeros(size(tmpForward{1,concatMat}.x,1),size(tmpForward{1,concatMat}.x,3));
                        MatFinBackward.x(:,concatMat,idxMat,:)=zeros(size(tmpBackward{1,concatMat}.x,1),size(tmpBackward{1,concatMat}.x,3));
                        MatFinUp.x(:,concatMat,idxMat,:)=zeros(size(tmpUp{1,concatMat}.x,1),size(tmpUp{1,concatMat}.x,3));
                        MatFinDown.x(:,concatMat,idxMat,:)=zeros(size(tmpDown{1,concatMat}.x,1),size(tmpDown{1,concatMat}.x,3));
                        MatFinGrasp.x(:,concatMat,idxMat,:)=zeros(size(tmpGrasp{1,concatMat}.x,1),size(tmpGrasp{1,concatMat}.x,3));
                        MatFinTwist.x(:,concatMat,idxMat,:)=zeros(size(tmpTwist{1,concatMat}.x,1),size(tmpTwist{1,concatMat}.x,3));
                        MatFinRest.x(:,concatMat,idxMat,:)=zeros(size(tmpRest{1,concatMat}.x,1),size(tmpRest{1,concatMat}.x,3));
                        idxMat=idxMat+1;
                    end
                    MatFinLeft.x(:,concatMat,idxMat,:)=tmpLeft{1,concatMat}.x(:,concatMatY,:);
                    MatFinRight.x(:,concatMat,idxMat,:)=tmpRight{1,concatMat}.x(:,concatMatY,:);
                    MatFinForward.x(:,concatMat,idxMat,:)=tmpForward{1,concatMat}.x(:,concatMatY,:);
                    MatFinBackward.x(:,concatMat,idxMat,:)=tmpBackward{1,concatMat}.x(:,concatMatY,:);
                    MatFinUp.x(:,concatMat,idxMat,:)=tmpUp{1,concatMat}.x(:,concatMatY,:);
                    MatFinDown.x(:,concatMat,idxMat,:)=tmpDown{1,concatMat}.x(:,concatMatY,:);
                    MatFinGrasp.x(:,concatMat,idxMat,:)=tmpGrasp{1,concatMat}.x(:,concatMatY,:);
                    MatFinTwist.x(:,concatMat,idxMat,:)=tmpTwist{1,concatMat}.x(:,concatMatY,:);
                    MatFinRest.x(:,concatMat,idxMat,:)=tmpRest{1,concatMat}.x(:,concatMatY,:);
                    idxMat=idxMat+1;
                end
            else
                MatFinLeft.x(:,concatMat,:,:)=tmpLeft{1,concatMat}.x;
                MatFinRight.x(:,concatMat,:,:)=tmpRight{1,concatMat}.x;
                MatFinForward.x(:,concatMat,:,:)=tmpForward{1,concatMat}.x;
                MatFinBackward.x(:,concatMat,:,:)=tmpBackward{1,concatMat}.x;
                MatFinUp.x(:,concatMat,:,:)=tmpUp{1,concatMat}.x;
                MatFinDown.x(:,concatMat,:,:)=tmpDown{1,concatMat}.x;
                MatFinGrasp.x(:,concatMat,:,:)=tmpGrasp{1,concatMat}.x;
                MatFinTwist.x(:,concatMat,:,:)=tmpTwist{1,concatMat}.x;
                MatFinRest.x(:,concatMat,:,:)=tmpRest{1,concatMat}.x;
            end
        end
    catch
        for concatMat=1:length(tmpLeft)
            MatFinLeft.x(:,concatMat,:,:)=tmpLeft{1,concatMat}.x;
            MatFinRight.x(:,concatMat,:,:)=tmpRight{1,concatMat}.x;
            MatFinForward.x(:,concatMat,:,:)=tmpForward{1,concatMat}.x;
            MatFinBackward.x(:,concatMat,:,:)=tmpBackward{1,concatMat}.x;
            MatFinUp.x(:,concatMat,:,:)=tmpUp{1,concatMat}.x;
            MatFinDown.x(:,concatMat,:,:)=tmpDown{1,concatMat}.x;
            MatFinGrasp.x(:,concatMat,:,:)=tmpGrasp{1,concatMat}.x;
            MatFinTwist.x(:,concatMat,:,:)=tmpTwist{1,concatMat}.x;
            MatFinRest.x(:,concatMat,:,:)=tmpRest{1,concatMat}.x;
        end
    end
    LeftData=MatFinLeft.x;
    RightData=MatFinRight.x;
    ForwardData=MatFinForward.x;
    BackwardData=MatFinBackward.x;
    UpData=MatFinUp.x;
    DownData=MatFinDown.x;
    GraspData=MatFinGrasp.x;
    TwistData=MatFinTwist.x;
    RestData=MatFinRest.x;;
    save(["./3_3DSpatial/"+mode{modIdx}+'/'+num2str(hz(hzIdx))+"/"+"Left"+"/sub"+string(sub)],'LeftData');
    save(["./3_3DSpatial/"+mode{modIdx}+'/'+num2str(hz(hzIdx))+"/"+"Right"+"/sub"+string(sub)],'RightData');
    save(["./3_3DSpatial/"+mode{modIdx}+'/'+num2str(hz(hzIdx))+"/"+"Forward"+"/sub"+string(sub)],'ForwardData');
    save(["./3_3DSpatial/"+mode{modIdx}+'/'+num2str(hz(hzIdx))+"/"+"Backward"+"/sub"+string(sub)],'BackwardData');
    save(["./3_3DSpatial/"+mode{modIdx}+'/'+num2str(hz(hzIdx))+"/"+"Up"+"/sub"+string(sub)],'UpData');
    save(["./3_3DSpatial/"+mode{modIdx}+'/'+num2str(hz(hzIdx))+"/"+"Down"+"/sub"+string(sub)],'DownData');
    save(["./3_3DSpatial/"+mode{modIdx}+'/'+num2str(hz(hzIdx))+"/"+"Grasp"+"/sub"+string(sub)],'GraspData');
    save(["./3_3DSpatial/"+mode{modIdx}+'/'+num2str(hz(hzIdx))+"/"+"Twist"+"/sub"+string(sub)],'TwistData');
    save(["./3_3DSpatial/"+mode{modIdx}+'/'+num2str(hz(hzIdx))+"/"+"Rest"+"/sub"+string(sub)],'RestData');
    clear LeftData RightData ForwardData BackwardData UpData DownData GraspData TwistData RestData;
end