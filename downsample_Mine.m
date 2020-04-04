function varargout = downsample_Mine(dat, varargin)
% proc_downsample - Downsample by subsampling
%
% Synopsis:
% cnt_d          = proc_downsample(dat, n)
% [cnt_d, mrk_d] = proc_downsample(dat, mrk, n)
%
% Ryota Tomioka

if isstruct(varargin{1})
  mrk = varargin{1};
  n = varargin{2};
  
else
  n = varargin{1}
end


varargout{1} = dat;
dat = dat(1:n:end,:);

  