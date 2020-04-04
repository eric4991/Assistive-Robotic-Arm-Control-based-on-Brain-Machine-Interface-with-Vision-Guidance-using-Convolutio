clear all; close all; clc;

dd='./0_RawData';

modIdx=1;

hz=[100,250,1000];
nameList={'GIGA_20190708_eslee','GIGA_20190708_hjwang','GIGA_20190708_jmlee','GIGA_20190710_dslim','GIGA_20190712_wjyun'};
SessionList={'session1','session2','session3'};
task={'reaching','multigrasp','twist'};
mode={'MI','realMove'};
for hzIdx=1:length(hz)
    for nameIdx=1:length(nameList)
        for sessionIdx=1:length(SessionList)
            for taskIdx=1:length(task)
                dir=[dd '\' nameList{nameIdx} '\' SessionList{sessionIdx} '\' task{taskIdx} '_' mode{modIdx}];
                opt=[];
                fprintf('** Processing of %s **\n', dir);
                try
                    hdr=eegfile_readBVheader(dir);
                catch
                    fprintf('file could not found');
                    continue;
                end
                Wps= [42 49]/hdr.fs*2;
                [n, Ws]= cheb2ord(Wps(1), Wps(2),3, 40);
                [filt.b, filt.a]= cheby2(n, 50, Ws);

                [cnt,mrk_orig]=eegfile_loadBV(dir,...
                    'filt',filt,'clab',{'not','EMG*'},'fs',hz(hzIdx));    
                cnt.title=['./1_ConvertedData/' mode{modIdx} '/' num2str(hz(hzIdx)) '/' nameList{nameIdx} '_' task{taskIdx} '_' mode{modIdx}];
                mrk=SharedDemo_1_ImageArrow(mrk_orig,taskIdx);
                mnt=getElectrodePositions(cnt.clab);
                fs_orig=mrk_orig.fs;
                var_list={'fs_orig',fs_orig,'mrk_orig',mrk_orig,'hdr',hdr};
                eegfile_saveMatlab(cnt.title,cnt,mrk,mnt,...
                    'channelwise',1,...
                    'format','int16',...
                    'resolution',NaN);
            end
        end
    end
end
