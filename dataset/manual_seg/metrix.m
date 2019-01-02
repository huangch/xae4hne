function [P,TP,FP,FN]= metrix('im1_auto_seg.mat','1m.tif')
load(ima),A=imread(imm);
[rm,cm]=find(A(:,:,2)~=0);
rm=rm';cm=cm';
ra=ra';ca=ca';P=size(rm,1);
TP=0;FP=0;
for j=1:length(ra)
    Ar=repmat(ra(j),1,length(rm));Ac=repmat(ca(j),1,length(rm));
    d=sqrt(((Ar-rm).^2)+((Ac-cm).^2));
    k=find(d==min(d));D=d(k);
        if D<=3
            TP=TP+1;
            rm=[rm(1:k-1),rm(k+1:end)];
            cm=[cm(1:k-1),cm(k+1:end)];
        else
            FP=FP+1;
        end
end
FN=size(rm,1);
save('metric1.mat','P','TP','FP','FN')
