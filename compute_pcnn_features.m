function compute_pcnn_features(param)

% ----- PARAMETERS --------
param.do_dyn = 1 ; % use dynamic features (differences)
param.do_acc = 0 ; % use differences of dynamic features
param.do_max = 1 ; % use max aggregation
param.do_min = 1 ; % use min aggregation
param.do_std = 0 ; % use std aggregation
param.do_mean = 0 ; % use mean aggregation
param.perpartL2 = 1 ; % normalize according to each part norm (from the training set)
param.partids = [1 2 3 4 5] ; % use "left_hand" "right_hand" "upper_body" "full_body" "full_image" parts respectively


if ~exist(param.savedir,'dir'); mkdir(param.savedir) ; end % create res folder if necessary

fprintf('\n------ Compute P-CNN features ------\n')

featdir_app=sprintf('%s/cnn_features_app/video_features',param.cachepath);
featdir_flow=sprintf('%s/cnn_features_flow/video_features',param.cachepath);

disp('In appearance')
if isfield(param,'perpartL2') && param.perpartL2
    fprintf('Compute per part norms --->  '); tic;
    norms=get_partnorms(param.trainsplitpath,featdir_app,param);
    fprintf('%d sec\n',round(toc));
else
    norms=[];
end

[Xn_train,Xn_test] = get_Xn_train_test(featdir_app,param,norms);


disp('In flow')
if isfield(param,'perpartL2') && param.perpartL2
    fprintf('Compute per part norms --->  '); tic;
    norms=get_partnorms(param.trainsplitpath,featdir_flow,param);
    fprintf('%d sec\n',round(toc));
else
    norms=[];
end
[Xn_trainOF,Xn_testOF] = get_Xn_train_test(featdir_flow,param,norms);

Xn_train = cat(1,Xn_train,Xn_trainOF); clear Xn_trainOF ;
Xn_test = cat(1,Xn_test,Xn_testOF); clear Xn_testOF ;

if param.compute_kernel
    disp('Compute Kernel Test')
    Ktest   = Xn_test'*Xn_train;
    savename=sprintf('%s/Ktest.mat',param.savedir);
    disp(['Save test kernel in: ',savename])
    assert(sum(isinf(Ktest(:)))==0 && sum(isnan(Ktest(:)))==0)
    save(savename,'Ktest','-v7.3')
    clear Ktest ;
    clear Xn_test ;
    
    disp('Compute Kernel Train')
    Ktrain  = Xn_train'*Xn_train;
    assert(sum(isinf(Ktrain(:)))==0 && sum(isnan(Ktrain(:)))==0)
    savename=sprintf('%s/Ktrain.mat',param.savedir);
    disp(['Save train kernel in: ',savename])
    save(savename,'Ktrain','-v7.3')
    clear Ktrain ;
    clear Xn_train ;
else
    savename=sprintf('%s/Xn_test.mat',param.savedir);
    disp(['Save test features in: ',savename])
    save(savename,'Xn_test','-v7.3')
    savename=sprintf('%s/Xn_train.mat',param.savedir);
    disp(['Save train features in: ',savename])
    save(savename,'Xn_train','-v7.3')
end


function [Xn_train,Xn_test] = get_Xn_train_test(featdirraw,param,norms)
fprintf('Collect train samples --->  '); tic;
Xn_train=collect_samples(param.trainsplitpath,featdirraw,param,norms) ;
fprintf('%d sec\n',round(toc));

fprintf('Collect test samples --->  '); tic;
Xn_test=collect_samples(param.testsplitpath,featdirraw,param,norms);
fprintf('%d sec\n',round(toc));


function Xn=collect_samples(splitpath,featdirraw,param,norms)
do_dyn = param.do_dyn ;
do_acc = param.do_acc ;
do_min = param.do_min ;
do_std = param.do_std ;
do_mean =param.do_mean;
partids =param.partids;
partL2=isfield(param,'perpartL2') && param.perpartL2 ;

if isfield(param,'do_max')
    do_max = param.do_max ;
else
    do_max = 1 ; % do maximum aggregation by default
end

assert(~(~do_min && ~do_max && ~do_mean && ~do_std));

%% Load features

% get sample list
[samplelist,numfil]=get_sample_list(splitpath,featdirraw);

% pre-allocate memory
tmp=load(samplelist{1}) ;
Xn=zeros((do_dyn+do_acc+1)*(do_min+do_std+do_mean+do_max)*length(partids)*length(tmp.features(1).x(1,:)),numfil) ;
if partL2
    invrepnorms=1./repmat(norms',length(tmp.features(1).x(1,:)),1);invrepnorms=invrepnorms(:)';
else
    invrepnorms=[];
end

parfor ii=1:numfil
    pathname=samplelist{ii};
    tmp=load(pathname) ;
    cnnf=[ tmp.features(partids).x ];
    
    if partL2
        cnnf=bsxfun(@times,cnnf,invrepnorms);
    end
    
    cnnf_diff=[];cnnf_acc=[];maxV = [];minV = [];stdV = [];meanV=[];
    if do_dyn
        if size(cnnf,1)>3 ;
            cnnf_diff = cnnf(4:end,:) - cnnf(1:end-3,:) ;
        elseif size(cnnf,1)>1;
            cnnf_diff = cnnf(2:end,:) - cnnf(1:end-1,:) ;
        else
            cnnf_diff=zeros(size(cnnf));
        end
        
        if do_acc
            if size(cnnf_diff,1)>1
                cnnf_acc = cnnf_diff(2:end,:) - cnnf_diff(1:end-1,:);
            else
                cnnf_acc=zeros(size(cnnf));
            end
        end
    end
    
    if do_max
        maxV = [max(cnnf,[],1)' ; max(cnnf_diff,[],1)' ; max(cnnf_acc,[],1)'];
    end
    if do_min
        minV = [min(cnnf,[],1)' ; min(cnnf_diff,[],1)' ; min(cnnf_acc,[],1)'];
    end
    if do_std
        stdV = [std(cnnf,0,1)' ; std(cnnf_diff,0,1)'; std(cnnf_acc,0,1)'];
    end
    if do_mean
        meanV = [mean(cnnf,1)' ; mean(cnnf_diff,1)' ; mean(cnnf_acc,1)'];
    end
    
    Xn(:,ii)=[maxV ; minV ; stdV ; meanV];
    
    %fprintf('%d out of %d\n',ii,numfil)
end

function norms=get_partnorms(splitpath,featdirraw,param)
partids = param.partids;

%% Compute norms
[samplelist,numfil]=get_sample_list(splitpath,featdirraw);


norms = zeros(length(partids),numfil);
nframes=zeros(length(partids),numfil);
parfor ii=1:numfil
    pathname=samplelist{ii};
    tmp=load(pathname) ;
    norms_ii=norms(:,ii);
    nframes_ii=nframes(:,ii);
    for nd=1:length(partids)
        cnnf=tmp.features(partids(nd)).x;
        norms_ii(nd)=norms_ii(nd)+sum(sqrt(sum(cnnf.^2,2)));
        nframes_ii(nd)=nframes_ii(nd)+size(cnnf,1);
    end
    norms(:,ii)=norms_ii;
    nframes(:,ii)=nframes_ii;
    
    %fprintf('NORM: %d out of %d\n',ii,numfil)
end
norms=sum(norms,2);
nframes=sum(nframes,2);
norms=norms./nframes;


function [samplelist,numfil]=get_sample_list(splitpath,featdirraw)
% open image list
split = fopen(splitpath) ;

% pre-allocate memory
%setenv('filepath',splitpath);
%[~,numfil]=system('cat $filepath | wc -l');numfil=str2num(numfil);
%samplelist=cell(numfil,1) ;
samplelist=cell(1000,1) ;

[sample,~] = strtok(fgetl(split));
ii=0; % number of loaded samples
while ischar(sample)
    ii=ii+1;
    
    if ii > length(samplelist) % allocate more
        samplelist=cat(1,samplelist,cell(1000,1));
    end
    
    samplelist{ii}=[featdirraw '/' sample '.mat'];
    %fprintf('Collect Sample: %d out of %d : %s\n',ii,numfil,sample)
    [sample,~] = strtok(fgetl(split));
end
fclose(split);

samplelist=samplelist(1:ii);
numfil=ii;
%assert(numfil == ii);
