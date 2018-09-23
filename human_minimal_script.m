% ======================================
% Getting maps for human minimal images
% ======================================

% Orig file with the PNAS paper data
load 'mircs_for_collaborations.mat'

% runnning over all 10 image objects from PNAS
for mirc_ind = 1:10
	fprintf('printing %d\n', mirc_ind)

	% print original image:
	imwrite(mircs(mirc_ind).org_image,sprintf('img%d.png',mirc_ind)) 	
	
	% [x,y] center location for minimal images:
	x = round(0.6*mean(mircs(mirc_ind).mirc_bbox(:,1:2)')'); 
	y = round(0.6*mean(mircs(mirc_ind).mirc_bbox(:,3:4)')');

	% filter minimal image size:	
	sz = mircs(mirc_ind).mirc_bbox(:,2)- mircs(mirc_ind).mirc_bbox(:,1);
	inds = find(and(sz'>0.3*50,sz'<0.5*50));
	% or, taking all minimal images:
	% inds = 1:length(y) 

	% plot dots at minimal image center:
	minimal_map = zeros(30);	
	for k = inds
		minimal_map(x(k),y(k))=255; 
	end

	% fused map:
	fused_minimal_map = imfuse(minimal_map,imresize(mircs(mirc_ind).org_image,[30,30]));

	% plot rects (frames) for minimal iamges: (optional)
	fused_minimal_map_and_rects = fused_minimal_map;

	rects_flag = false;
	if(rects_flag)	
		for k = inds
			bb = round(0.6*mircs(mirc_ind).mirc_bbox(k,:));
			bb = [bb(3),bb(1),bb(4)-bb(3),bb(2)-bb(1)];  
			fused_minimal_map_and_rects = insertShape(fused_minimal_map_and_rects,'Rectangle',bb,'LineWidth',1);
		end	
	end

	% printing fused map:	
	imwrite(fused_minimal_map_and_rects,sprintf('map%d.png',mirc_ind))

	% raw data:
	mircs_humans(mirc_ind).img_org = imresize(mircs(mirc_ind).org_image,[30,30]);
	mircs_humans(mirc_ind).mircs_loc = minimal_map;
end 

% Saving raw data:
save('mircs_humans','mircs_humans');

% Usage:
% >~/MIRCInterpretation$ matlab -nodesktop -nojvm -nosplash
% 