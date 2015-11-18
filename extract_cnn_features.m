function extract_cnn_features(video_names,param)

% create cache folders
cdirs={'cnn_features_app','cnn_features_app/frame_features','cnn_features_app/video_features' ...
    'cnn_features_flow','cnn_features_flow/frame_features','cnn_features_flow/video_features'};
for d=1:length(cdirs)
    dname=sprintf('%s/%s',param.cachepath,cdirs{d});
    if ~exist(dname,'dir'); mkdir(dname) ; end
end

fprintf('\n------ Extract CNN features ------\n')

suf={'app','flow'} ;

for i=1:2 % appearance and flow
    
    % get folders
    patchesdir = sprintf('%s/patches_%s',param.cachepath,suf{i}) ;
    framefeaturesdir = sprintf('%s/cnn_features_%s/frame_features',param.cachepath,suf{i});
    videofeaturesdir = sprintf('%s/cnn_features_%s/video_features',param.cachepath,suf{i});
    
    % get list of part patches
    [filelist,outlist]=get_patches_list(patchesdir,framefeaturesdir);
    
    % get net
    net=param.(sprintf('net_%s',suf{i}));
    if param.use_gpu ; net = vl_simplenn_move(net, 'gpu') ; end % move net on GPU if needed
    bsize=param.batchsize;
    nim=length(filelist);
    
    % extract CNN features per frame
    for b=1:bsize:nim
        
        fprintf('%s -- feature extraction: %d\tover %d:\t',suf{i},b,nim);tic;
        im = vl_imreadjpeg(filelist(b:min(b+bsize-1,nim)),'numThreads', param.nbthreads_netinput_loading) ;
        im = cat(4,im{:}) ;
        im = bsxfun(@minus, im, net.normalization.averageImage) ;
        if param.use_gpu ; im = gpuArray(im) ; end
        res=vl_simplenn(net,im);
        fprintf('extract %.2f s\t',toc);tic;
        save_feats(squeeze(res(end-2).x),outlist(b:min(b+bsize-1,nim)),param); % take features after last ReLU
        fprintf('save %.2f s\n',toc)
    end
    
    % group frame features in their corresponding video
    % features(1).x <--- left hand
    % features(2).x <--- rigth hand
    % features(3).x <--- upper body
    % features(4).x <--- full body
    % features(5).x <--- full image
    
    parfor vi=1:length(video_names)
        group_cnn_features(framefeaturesdir,videofeaturesdir,video_names{vi});
    end
end



function [filelist,outlist]=get_patches_list(indirname,outdirname)

bodyparts={'left_hand'  'right_hand' 'upper_body' 'full_body' 'full_image'};

images=dir(sprintf('%s/%s/*jpg',indirname,bodyparts{1}));
images = {images.name};
[~,resnames,~]=cellfun(@(x) fileparts(x),images,'UniformOutput',false);
resnames=strcat(resnames,repmat({'.mat'},1,length(images)));

filelist=cell(1,length(bodyparts)*length(images));
outlist=cell(1,length(bodyparts)*length(images));
for i=1:length(bodyparts)
    indirpath=sprintf('%s/%s/',indirname,bodyparts{i});
    destdirpath=sprintf('%s/%s/',outdirname,bodyparts{i});
    if ~exist(destdirpath,'dir'); mkdir(destdirpath) ; end
    
    pimages=repmat({indirpath},1,length(images));
    qimages=repmat({destdirpath},1,length(images));
    
    filelist(1,1+(i-1)*length(images):i*length(images)) = strcat(pimages,images);
    outlist(1,1+(i-1)*length(images):i*length(images)) = strcat(qimages,resnames);
end

function save_feats(feats,outlist,param)
assert(length(outlist)==size(feats,2));
if param.use_gpu; feats=gather(feats); end

parfor i=1:length(outlist)
    out=outlist{i};
    features=feats(:,i)';
    parsave(out,features);
end

function parsave(out,features)
save(out,'features');

function group_cnn_features(framefeaturesdir,videofeaturesdir,video_name)
% features(1).x <--- left hand
% features(2).x <--- rigth hand
% features(3).x <--- upper body
% features(4).x <--- full body
% features(5).x <--- full image

% part sub-directories to check
subD={'left_hand'  'right_hand' 'upper_body' 'full_body' 'full_image'};

features = [] ;
for i=1:length(subD)
    dirpath=sprintf('%s/%s',framefeaturesdir,subD{i});
    pathname=sprintf('%s/%s_im*',dirpath,video_name);
    td=dir(pathname) ;
    assert(~isempty(td));
    x=zeros(length(td),4096) ;
    features(i).name=sprintf('CNNf_%s',subD{i}) ;
    for j=1:length(td)
        samplepath=sprintf('%s/%s',dirpath,td(j).name);
        tmp=load(samplepath) ;
        x(j,:)=tmp.features ;
    end
    features(i).x=x;
end
save(sprintf('%s/%s.mat',videofeaturesdir,video_name),'features');
