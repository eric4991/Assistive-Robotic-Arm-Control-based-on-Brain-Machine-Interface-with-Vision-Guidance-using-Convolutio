%% @written by Kyung-Hwan, Shim

bbci_acquire_bv('close');
clear all; close all;clc; 
%% robot arm connection
disp((sprintf('Data transmission')));
disp((sprintf('Command: Home\n')));
%% onilne experiement
% startup_bbci
% classes=[['Forward','Grasp','Twist'],['Left','Right','Forward','Backward','Grasp','Twist'],['Left','Right','Forward','Backward','Up','Down','Grasp','Twist']]

refIdx=8
global EEG_MAT_DIR
EEG_MAT_DIR = '';
params=struct;
state=bbci_acquire_bv('init',params);
EEGData=[];
mnt=getElectrodePositions(state.clab);
cnt.clab=state.clab; 
cnt.fs=state.fs;
cnt.title='./tmp';
fs=cnt.fs;
toggle=0;
Trial_Results = [];
if count(py.sys.path,'')==0
    insert(py.sys.path,int32(0),'');
end

while(1)
    pause('on');
    pause(7);
    infor = '\nTrial : %d\n';
    fprintf(infor,total_loop);
    
    data=bbci_acquire_bv(state);
    EEGData=[EEGData;data];
    if size(EEGData,1)>=6001
        timeIdx=1;
        EEGData=downsample_Mine(EEGData,4)
        for timeWinStart=1:50:251
            cnt.x=EEGData(timeWinStart:timeWinStart+300,:);
            Wps=[42 49]/cnt.fs*2;
            [n, Ws]=cheb2ord(Wps(1),Wps(2),3,40);
            [filt.b,filt.a]=cheby2(n,50,Ws);
            cnt=proc_filt(cnt,filt.b,filt.a);
            cnt=proc_filtButter(cnt,5,[4 40]);
            cnt=proc_selectChannels(cnt,{'F3','F1','Fz','F2','F4',...
                'FC3','FC1','FCz','FC2','FC4',...
                'C3','C1', 'Cz', 'C2', 'C4', ...
                'CP3','CP1','CPz','CP2','CP4',...
                'P3','P1','Pz','P2','P4'});
            b4DL=cnt.x;
            refData=zeros(1,751);
            tmpEEGdata=cat(1,b4DL(1:refIdx-1,:),refData)
            b4DL=cat(1,tmpEEGdata,b4DL(refIdx:size(b4DL,1),:))
            b4DL=reshape(b4DL,[1,18775]);
            outputs=py.Online_5_Online.onlineMain(b4DL);
            if timeWinStart==1
                data=double(py.array.array('d',py.numpy.nditer(outputs)));
            else
                data=data+double(py.array.array('d',py.numpy.nditer(outputs)));
            end
        end
        finDecision=find(data==max(data))
        print(FinDecision)
        py.SharedDemo_6_VisionTrial_1.mainRunning(FinDecision)
    end
end