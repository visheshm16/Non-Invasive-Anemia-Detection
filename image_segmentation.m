clc
clear all
close all


% reading images
x=imread('eye images/50.jpg');
% resizing and cropping
x=imresize(x,[250 NaN]);
[rx,rect]=imcrop(x);


% image segmentation operations
% creating mask image
mask_x=bwareaopen(imfill(imclose(imclose(bwareaopen(edge(rgb2gray(rx),'canny'),70),strel('line',100,0)),strel('line',50,90)),'holes'),500);
% red component
final_red=rx(:,:,1).*uint8(mask_x);
% green component
final_green=rx(:,:,2).*uint8(mask_x);
% blue component
final_blue=rx(:,:,3).*uint8(mask_x);
% combining all 3 components
final_x=cat(3,final_red,final_green,final_blue);


% displaying the images
subplot(1,3,1);
imshow(rx);
title('Original image');
subplot(1,3,2);
imshow(mask_x);
title('Mask image');
subplot(1,3,3);
imshow(final_x);
title('Final image');


% saving the image
% imwrite(final_x,'cropped/50.png')