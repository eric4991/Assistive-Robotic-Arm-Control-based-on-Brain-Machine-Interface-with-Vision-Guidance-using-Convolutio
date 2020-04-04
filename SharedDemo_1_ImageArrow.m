function mrk = Caps_1_ImageArrow( mrko,task, varargin )
%% Reaching 23~
switch task
    case 1
        stimDef= {'S 11',    'S 21',      'S 31',  'S 41',   'S 51',  'S 61' 'S  8';
                 'Forward', 'Backward', 'Left',  'Right',  'Up',    'Down' 'Rest'};
    case 2
%         stimDef= {'S 71',  'S 81', 'S  8';
%                   'Grasp', 'Open', 'Rest'};
        stimDef= {'S 11','S  8';
                  'Grasp','Rest'};
    case 3
        stimDef ={'S 91';
            'LeftTwist'};
end

% Default
miscDef= {'S 13',    'S 14';
          'Start',   'End'};

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, 'stimDef', stimDef, ...
                       'miscDef', miscDef);

mrk= mrk_defineClasses(mrko, opt.stimDef);
mrk.misc= mrk_defineClasses(mrko, opt.miscDef);