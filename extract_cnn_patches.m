function extract_cnn_patches(video_names,param)

% create cache folders
cdirs={'patches_app','patches_flow','patches_app/left_hand','patches_flow/left_hand', ...
    'patches_app/right_hand','patches_flow/right_hand','patches_app/upper_body','patches_flow/upper_body', ...
    'patches_app/full_body','patches_flow/full_body','patches_app/full_image','patches_flow/full_image'};
for d=1:length(cdirs)
    dname=sprintf('%s/%s',param.cachepath,cdirs{d});
    if ~exist(dname,'dir'); mkdir(dname) ; end
end

fprintf('\n------ Compute CNN patches ------\n')

parfor vi = 1:length(video_names)
    fprintf('extract patches .. : %d out of %d videos\n',vi,length(video_names))
    
    % get image list in the current video
    vidname=video_names{vi} ;
    images=dir(sprintf('%s/%s/*%s',param.impath,vidname,param.imext));
    images = {images.name};
    
    % get video joint positions and human scales
    positions=load(sprintf('%s/%s/joint_positions',param.jointpath,vidname)) ;
    scale=positions.scale ;
    positions=positions.pos_img ;
    
    suf={'app','flow'} ;
    imdirs = {param.impath,sprintf('%s/OF',param.cachepath)};
    
    for i=1:2 % appearance and flow
        imdirpath = imdirs{i};
        
        net=param.(sprintf('net_%s',suf{i}));
        
        for idim=1:min(length(images),length(positions))
            if exist(sprintf('%s/full_image/%s_im%05d.jpg',param.cachepath,vidname,idim),'file')
                continue;
            end
            % get image
            if i==1 % appearance
                impath = sprintf('%s/%s/%s',imdirpath,vidname,images{idim}) ;
            else % flow
                [~,iname,~]=fileparts(images{idim});
                impath = sprintf('%s/%s/%s.jpg',imdirpath,vidname,iname) ; % flow has been previously saved in JPG
                if ~exist(impath,'file'); continue ; end ; % flow was not computed (see compute_OF.m for info)
            end
            im = imread(impath);
            
            % get part boxes
            
            % part CNN (fill missing part before resizing)
            sc=scale(idim); lside=param.lside*sc ;
            % left hand
            lhand = get_box_and_fill(positions(:,param.lhandposition,idim)-lside,positions(:,param.lhandposition,idim)+lside,im);
            lhand = imresize(lhand, net.normalization.imageSize(1:2)) ;
            
            % right right
            rhand = get_box_and_fill(positions(:,param.rhandposition,idim)-lside,positions(:,param.rhandposition,idim)+lside,im);
            rhand = imresize(rhand, net.normalization.imageSize(1:2)) ;
            
            % upper body
            sc=scale(idim); lside=3/4*param.lside*sc ;
            upbody = get_box_and_fill(min(positions(:,param.upbodypositions,idim),[],2)-lside,max(positions(:,param.upbodypositions,idim),[],2)+lside,im);
            upbody = imresize(upbody, net.normalization.imageSize(1:2)) ;
            
            % full body
            fullbody = get_box_and_fill(min(positions(:,:,idim),[],2)-lside,max(positions(:,:,idim),[],2)+lside,im);
            fullbody = imresize(fullbody, net.normalization.imageSize(1:2)) ;
            
            % full image CNNf (just resize frame)
            fullim = imresize(im, net.normalization.imageSize(1:2)) ;
            
            imwrite(lhand,sprintf('%s/patches_%s/left_hand/%s_im%05d.jpg',param.cachepath,suf{i},vidname,idim));
            imwrite(rhand,sprintf('%s/patches_%s/right_hand/%s_im%05d.jpg',param.cachepath,suf{i},vidname,idim));
            imwrite(upbody,sprintf('%s/patches_%s/upper_body/%s_im%05d.jpg',param.cachepath,suf{i},vidname,idim));
            imwrite(fullbody,sprintf('%s/patches_%s/full_body/%s_im%05d.jpg',param.cachepath,suf{i},vidname,idim));
            imwrite(fullim,sprintf('%s/patches_%s/full_image/%s_im%05d.jpg',param.cachepath,suf{i},vidname,idim));
        end
    end
end
