%xavier

load 'mircs_for_collaborations.mat'
for mind = 1:10
	fprintf('printing %d\n', mind)
	
	x = round(0.6*mean(mircs(mind).mirc_bbox(:,1:2)')'); 
	y = round(0.6*mean(mircs(mind).mirc_bbox(:,3:4)')');

	% filter:	
	sz = mircs(mind).mirc_bbox(:,2)- mircs(mind).mirc_bbox(:,1);
	inds = find(and(sz'>0.3*50,sz'<0.5*50));
	% inds = 1:length(y) 

	% plot dots:
	A = zeros(30);	
	for k = inds
		A(x(k),y(k))=255; 
	end

	imwrite(mircs(mind).org_image,sprintf('img%d.png',mind)) 	
	%imwrite(A,sprintf('map%d.png',mind))

	FA = imfuse(A,imresize(mircs(mind).org_image,[30,30]));

	% plot rects:
	FAR = FA;
%	for k = inds
%		bb = round(0.6*mircs(mind).mirc_bbox(k,:));
%		bb = [bb(3),bb(1),bb(4)-bb(3),bb(2)-bb(1)];  
%		FAR = insertShape(FAR,'Rectangle',bb,'LineWidth',1);
%	end	
	imwrite(FAR,sprintf('map%d.png',mind))

	mircs_humans(mind).img_org = imresize(mircs(mind).org_image,[30,30]);
	mircs_humans(mind).mircs_loc = FA;
end 
save('mircs_humans','mircs_humans');