function box=get_box_and_fill(topleft,botright,im)
% given boxe positions and image, return the corresponding box with pixels
% out of the image filled with gray

[h,w,~]=size(im);
topleft=round(topleft);
botright=round(botright);

box=uint8(128*ones(botright(2)-topleft(2)+1,botright(1)-topleft(1)+1,3));

% check if a part of the box is in the image
if topleft(1) > w || topleft(2) > h || botright(1) < 1 || botright(2) < 1
    return % return a gray box
end

left_min = max(topleft(1),1) ;
top_min = max(topleft(2),1) ;

right_max = min(botright(1),w) ;
bot_max = min(botright(2),h) ;


im_w=left_min:right_max ;
im_h=top_min:bot_max ;

box(top_min-topleft(2)+1:top_min-topleft(2)+length(im_h),left_min-topleft(1)+1:left_min-topleft(1)+length(im_w),:)=im(im_h,im_w,:);
