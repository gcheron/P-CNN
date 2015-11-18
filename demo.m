%% Demo to compute P-CNN
% Report bugs to guilhem.cheron@inria.fr
%
% 
% ENABLE GPU support (in my_build.m) and MATLAB Parallel Pool to speed up computation (parpool) 

if ~isdeployed
    addpath('brox_OF'); % Brox 2004 optical flow
end
matconvpath = 'matconvnet-1.0-beta11'; % MatConvNet
run([matconvpath '/my_build.m']); % compile: modify this file to enable GPU support (much faster)
run([matconvpath '/matlab/vl_setupnn.m']) ; % setup  

%% reproduce paper (ICCV 15) results (-0.9% acc, see README.md)
reproduce_ICCV15_results 

%% P-CNN computation
% ----- PARAMETERS --------
param=[];
param.lhandposition=13; % pose joints positions in the structure (JHMDB pose format)
param.rhandposition=12;
param.upbodypositions=[1 2 3 4 5 6 7 8 9 12 13];
param.lside = 40 ; % length of part box side (also depends on the human scale)
param.savedir = 'p-cnn_features_split1'; % P-CNN results directory
param.impath = 'JHMDB/images' ; % input images path (one folder per video)
param.imext = '.png' ; % input image extension type
param.jointpath = 'JHMDB/joint_positions' ; % human pose (one folder per video in which there is a file called 'joint_positions.mat')
param.trainsplitpath = 'JHMDB/splits/JHMDB_train1.txt'; % split paths
param.testsplitpath = 'JHMDB/splits/JHMDB_test1.txt';
param.cachepath = 'cache'; % cache folder path
param.net_app  = load('models/imagenet-vgg-f.mat') ; % appearance net path
param.net_flow = load('models/flow_net.mat') ; % flow net path
param.batchsize = 128 ; % size of CNN batches
param.use_gpu = false ; % use GPU or CPUs to run CNN?
param.nbthreads_netinput_loading = 20 ; % nb of threads used to load input images
param.compute_kernel = true ; % compute linear kernel and save it. If false, save raw features instead.


% get video names
video_names = dir(param.impath);
video_names={video_names.name};
video_names=video_names(~ismember(video_names,{'.','..'}));

if ~exist(param.cachepath,'dir'); mkdir(param.cachepath) ; end % create cache folder

% 1 - pre-compute OF images for all videos
compute_OF(video_names,param); % compute optical flow between adjacent frames

% 2 - extract part patches
extract_cnn_patches(video_names,param)

% 3 - extract CNN features for each patch and group per video
extract_cnn_features(video_names,param)

% 4 - compute final P-CNN features + kernels
compute_pcnn_features(param); % compute P-CNN for split 1


% compute for another split
param.savedir = 'p-cnn_features_split2';
param.trainsplitpath = 'JHMDB/splits/JHMDB_train2.txt';
param.testsplitpath = 'JHMDB/splits/JHMDB_test2.txt';
compute_pcnn_features(param); % compute P-CNN for split 2

