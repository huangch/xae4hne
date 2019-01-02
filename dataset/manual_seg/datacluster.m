%function [F]=datacluster()
tic
clear all, close all
tic
% tgtFnStr1 = '*.tif';
% tgtFnStr2='*.mat';
% srcDir1 = 'K:\Image analysis\Images\finalresults\';
% fullPathSrcI = srcDir;     
% imgFileList1 = dir(strcat(fullPathSrcI,tgtFnStr1));
% imgFileList2 = dir(strcat(fullPathSrcI,tgtFnStr2));
% for j = 1:length(imgFileList1)
%     fnHisto1 = imgFileList1(j).name;
%     imgstring1=strcat(fullPathSrcI,imgFileList1(j).name);
%     imgstring2=strcat(fullPathSrcI,imgFileList2(j).name);
%     fnamemac = [fnHisto1(1:end-4),'_mac.fig'];
%     fnameEM1 = [fnHisto1(1:end-4),'_EM.mat'];
%     fnameEM2 = [fnHisto1(1:end-4),'_EM.fig'];

k=1; C1=[];C2=[];C3=[];S1=[];S2=[];S3=[];image = imread('image1.tif');
[m,n]=size(image(:,:,1));Riy=[];R=[];C=[];
for dd=1:m
    Riy=cat(2,Riy,repmat(dd,1,n));
end
Rix=repmat(1:n,1,m);
load image1.mat
c=ICf;
while k<length(c)
    numpts=c(2,k);
    px = c(1,k+1:numpts+k);
    py = c(2,k+1:numpts+k);
    k=k+numpts+1;
    [IN]=inpolygon(Rix,Riy,px,py);
    W=find(IN==1);L1=[];L2=[];L3=[];
    BW=poly2mask(px,py,m,n);
    LL=bwlabel(BW);
    stat=regionprops(LL,'centroid');
    stats=[stat.Centroid];
    cc=stats(1:2:end-1);cc=mean(cc);
    rr=stats(2:2:end);rr=mean(rr);
    R=[R;rr];C=[C;cc];
    for kk=1:length(W)
        L1=[L1 image(Riy(W(kk)),Rix(W(kk)),1)];
        L2=[L2 image(Riy(W(kk)),Rix(W(kk)),2)];
        L3=[L3 image(Riy(W(kk)),Rix(W(kk)),3)];
    end
    C1=[C1 (sum(L1)./length(L1))];S1=[S1 std(double(L1))];
    C2=[C2 (sum(L2)./length(L2))];S2=[S2 std(double(L2))];
    C3=[C3 (sum(L3)./length(L3))];S3=[S3 std(double(L3))];
end
F=[C1;C2;C3;S1;S2;S3];
save('feature.mat','F')
D=F';i=kmeans(D,2);
o1=find(i==1);
o2=find(i==2);
imagesc(image(:,:,1:3));hold on
if o1>o2
    ra=R(o1,1);ca=C(o1,1);rn=R(o2,1);cn=C(o2,1);
else
    ra=R(o2,1);ca=C(o2,1);rn=R(o1,1);cn=C(o1,1);
end
plot(ca,ra,'.g','MarkerSize',15),plot(cn,rn,'.r','MarkerSize',15),hold off
saveas(gcf,'cluster_EMimage35.fig')
save('im1_auto_seg.mat','ra','ca','rn','cn')
toc
%end
    
    