%% Written by Mira Liu for Imaging Practicum 1 2021, CAD Lab

function [Areas_all, Circularities_all,Areas_tlesions,Circularities_tlesions,LDA_features_f,LDA_features_t] = Exp1(varargin)
%varargin{1} = threshold, varargin{2} = gaussian sigma

%% set up to run over all 6 images and get appropriate information from each
ImageFiles = fullfile('../Imaging 1 Practicum/CTcase','*.pgm');
ImageNames = dir(ImageFiles);
    Areas_all = []; %all object areas
    Circularities_all = []; %all object circularities
    Areas_tlesions = []; %all true lesion areas
    Circularities_tlesions = []; %all true lesion circularities
    PotentialLesions_all = []; %number of potential lesions per case.
    LDA_features_f = []; %all potential lesion features
    LDA_features_t = [];% true lesion features
    for i=1:6%6 images
        fprintf(strcat('processing number...  ', string(i), '\n'))
        fullfilename = fullfile('../Imaging 1 Practicum/CTcase', ImageNames(i).name);
        Image = imread(fullfilename);
        
%% Experiment pt 1: 
        % Thresholding function: ~180 seems to be an appropriate threshold.
        threshold = 180;
        BinaryThresholdImage = Image<threshold; %this will set all values < Threshold = varargin{1} to 1
        OtsuThreshold = ~imbinarize(Image); %this uses Otsu's method which "chooses the threshold value to minimize the intraclass variance". https://www.mathworks.com/help/images/ref/imbinarize.html
        %Otsu threshold seems fairly similar to the Binary Threshold.
        
        %Segmenting lungs (using binary threshold image) 
        Masked = bsxfun(@times, Image, cast(BinaryThresholdImage,class(Image))); %apply binary mask to segment lung
        
        %showing the original image, thresholding, and segmentation
        %{
        figs1 = figure;
        subplot(1,3,1),imshow(Image,[])
        subplot(1,3,2), imshow(BinaryThresholdImage)
        subplot(1,3,3),imshow(Masked)
        sgtitle('Original Image, Threshold, Segmentation')
        %saveas(figs1,fullfile('../Imaging 1 Practicum/CTcase',strcat(ImageNames(i).name(1:end-4),'_Masks.tif')));
        %}
        
%% Experiment pt 2: 

        %First just gray-level threshold
        simplethreshold = Masked>60; % ~ 60 seems fairly good visually.

        %LoGFilter: gaussian smoothing followed by a Laplacian operation!

        %Create gaussian laplacian of specific kernel size
        sigma = 3.5; %set sigma
        kernel = 19; %set kernel dimensions
        fG = zeros(kernel, kernel); %make empty matrix to fill
        for xx = 1:kernel %to make an x by y kernel fG following the equation given
            for yy = 1:kernel
                a = -kernel/2 + xx; %to shift to be centered at (0,0)
                b = -kernel/2 + yy; %to shift to be centered at (0,0)
                fG(xx,yy)=(1/(3.14*sigma^4))*((a^2 + b^2 - 2*(sigma^2))/(2*(sigma^2)))*exp(-(a^2 + b^2)/(2*(sigma^2))); %equation given
            end
        end
        %now convolve the LoG with the segmented lung
        LesionEnhanced = conv2(double(Masked), fG,'same'); 
        
        
        %gray-level thresholding of LoG filter
        graylevelimagemask = LesionEnhanced< -1; % binary (either is or is not in the stated range, tested by eye? not sure why it's negative values, but it seems to work)
        graylevelimage=bsxfun(@times, LesionEnhanced, cast(graylevelimagemask,class(LesionEnhanced))); %thresholded image (values shown between the given range)

        %Showing the original image, single value thresholding, LoG filtered image, LoG Filtered image then thresholded
        %{
        figs2 = figure;
        subplot(1,4,1), imshow(Image)
        subplot(1,4,2),imshow(simplethreshold,[])
        subplot(1,4,3),imshow(LesionEnhanced,[])
        subplot(1,4,4), imshow(graylevelimagemask,[])
        %saveas(figs2, fullfile('../Imaging 1 Practicum/CTcase',strcat(ImageNames(i).name(1:end-4),'_LoG.tif')))
        %}
        
%% Experiment pt 3
        %labeling the binary image
        [x,y]=size(graylevelimagemask); %number of pixels along x and y dimensions
        ObjectNumber = 1; %how many objects have been found so far (starting with 1)
        SavePixels = [];
        LabelArray_attempt = zeros(x,y); %set up as binary image to now label as connected components
        Perimeter_image = zeros(x,y); %make image of the object perimeters...
        
        %first pass of labeling the objects
        for m = 1:x %x position (# of rows down) (so travelling in horizontal raster order left to right, top to bottom)
            for n = 1:y %y position (# of columns to the right)
                if graylevelimagemask(m,n) == 1 %if the pixel is an object
                    [CheckAll, CheckPrior, CheckPost] = checkneighbors(graylevelimagemask,m,n,x,y); %check how many neighbors it has using nested function "checkneighbors"
                    if CheckAll == 0 %so no neighbors 
                        ObjectNumber = ObjectNumber + 1; % increase the number of objects
                        LabelArray_attempt(m,n) = ObjectNumber; %label that pixel the new numbered object
                    else %so if there are neighbors, i.e. it is in an object of more than one pixel
                        if CheckAll == 8 %if it is in the middle of the object
                            [minlabel,maxlabel] = checklabels(LabelArray_attempt,m,n,x,y); %get label of object surrounding it using nested function "checklabels"
                            LabelArray_attempt(m,n)=minlabel; %number it the smallest surrounding label
                        else %if it is a perimeter
                            if CheckAll < 8
                                Perimeter_image(m,n)=1; %start making an image of all perimeters
                            end
                            if CheckPrior == 0 % so no priors, i.e. it's the first pixel of the object to be encountered
                                ObjectNumber = ObjectNumber + 1; % increase the number of objects
                                LabelArray_attempt(m,n)=ObjectNumber; %label that pixel the new numbered object
                            else %if there ARE priors
                                % then name it the surrounding (smallest?) labeled object
                                [minlabel,maxlabel] = checklabels(LabelArray_attempt,m,n,x,y); %get label of object surrounding it using nested function "checklabels"
                                if isempty(minlabel) %if none of the priors have been labeled yet
                                    ObjectNumber = ObjectNumber + 1; %add 1
                                    LabelArray_attempt(m,n)=ObjectNumber;
                                else
                                    LabelArray_attempt(m,n)=minlabel; %give it that minimum label.
                                end
                            end
                        end
                    end
                end
            end
        end
        
        %second pass of labeling the objects to clean labels up
        CheckPasses = zeros(x,y); %set up a binary image to label the number of pixels that have neighbors that are different numbers.
        [LabelArray_attempt,CheckPasses] = checkingsecondpass(graylevelimagemask,CheckPasses,LabelArray_attempt,x,y); %going through labled objects and making sure all numbers are same within one object using nested function "checkingsecondpass"
        
        %now go through a 3rd time to renumber them correctly. 
        CheckPass3 = unique(LabelArray_attempt); %get the number of objects in the image
        for idxs = 1: length(CheckPass3)
            LabelArray_attempt(LabelArray_attempt==CheckPass3(idxs))=idxs-1; %so starting with the 2nd value, label according to index (because zero is background and should remain zero) to get N objects numbered from 1 to N 
        end

        
        % Showing the original image, the LoG Filtered and then thresholded image, and then the labeled image
        %{
        figs3 = figure;
        subplot(1,3,1), imshow(Image)
        subplot(1,3,2), imshow(graylevelimagemask,[])
        subplot(1,3,3),imshow(LabelArray_attempt,[])
        saveas(figs3,fullfile('../Imaging 1 Practicum/CTcase',strcat(ImageNames(i).name(1:end-4),'_Labeled.tif')));
        %}
        
        
        %Now analyzing pattern features
        
        %Area and Perimeter calculation
        NumberOfObjects = max(max(LabelArray_attempt, [],'all')); %number of objects in the image
        Areas = zeros(1,NumberOfObjects); %number of areas to calculate
        Perimeters = zeros(1,NumberOfObjects); %number of perimeters to calculate
        Circularities = zeros(1,NumberOfObjects); %number of circularities to calculate
        LabelArray_rbs = zeros(x,y); %potential lesion candidates
        for Object_number = 1:NumberOfObjects %for each of the objects
            [row1, column1, dataval]=find(LabelArray_attempt == Object_number); %find all pixels in array with certain number
            %Area calculation
            Area = length(row1);
            Areas(Object_number)=Area;  %the number of pixels with the value of a specific object number
            %Perimeter calculation (using a modified version of the equation given to us)
            Perimeter_count = 0; %starting with zero
            for idx1 = 1:length(row1) %now go through pixels of the object of interest
                [CheckAll, CheckPrior, CheckPost] = checkneighbors(graylevelimagemask,row1(idx1),column1(idx1),x,y); %check how many neighbors it has using nested function "checkneighbors"
                [Perim_neighbors, h, d]=checkneighbors_perim(Perimeter_image,row1(idx1),column1(idx1),x,y); %so check the indices returned and if they are one of the perimeters, keep same max image size of x,y, using the nested function "checkneighbors_perim"
                if CheckAll <8 %if there is one background pixel (to remove double counting)
                    if Perim_neighbors > 0 %if there is a neighbor that is also a perimeter, we have to check if it's diagonal or horizontal/vertical
                        Perimeter_count = Perimeter_count + (1/Perim_neighbors)*(h + sqrt(2)*d); %1/perim_neighbors to make up for double counting (i.e. number of paths per pixel because I check every pixel in background rather than going in order) and using h and d to determine vertical/horizontal vs diagonal path
                    else
                        Perimeter_count = Perimeter_count + 1;
                    end
                end
            end
            Perimeters(Object_number)=Perimeter_count; %update the perimeter for this specific object!
            Circularity = 4*3.14*(Area/(Perimeter_count^2)); %calculate circularity given area and perimeter
            Circularities(Object_number)=Circularity; %update the circularities
            
            %now rule based scheme
            %so rules: 110 =< Area < 580, circularity above .21? (determined by scatter plot of area v. circularity)
            if Area >= 110 && Area < 580 && Circularity > .21 %
                for idx2 = 1:length(row1)
                    LabelArray_rbs(row1(idx2),column1(idx2))=Object_number; %put that object in the rule based scheme output.
                end
            end
        end
        

        PotentialLesions_all = [PotentialLesions_all length(unique(LabelArray_rbs))]; %add the number of objects remaining as potential lesions
        %so Case 1 has no false positives, Case 2 has 2 false positives, Case 3 has 1, Case 4 has 2, Case 5 has 3, case 5 has 3.
        %also there are no false negatives, so sensitivity is 100%
        
        % now applying the Rule based scheme
        %{
        fig5 = figure;
        subplot(1,3,1),imshow(Image,[])
        subplot(1,3,2),imshow(LabelArray_attempt,[])
        subplot(1,3,3), imshow(LabelArray_rbs,[])
        saveas(fig5,fullfile('../Imaging 1 Practicum/CTcase',strcat(ImageNames(i).name(1:end-4),'_RBS.tif')));

        %}
        
        %To show the development of the CAD code...
        %{
        fig5 = figure;
        sgtitle('Image, LoG Filtered, LoG Threshold, Labeled Objects')
        n = 5;
        subplot(1,n,1), imshow(Image,[]) %show original image
        subplot(1,n,2), imshow(simplethreshold,[]) %just threshold on segmented image
        subplot(1,n,3), imshow(LesionEnhanced,[]) %LoG filtered
        %subplot(1,n,4), imshow(graylevelimage,[]) %LoG with thresholded 
        subplot(1,n,5), imshow(LabelArray_attempt,[]) %show labeled objects
        subplot(1,n,6,), imshow(LabelArray_rbs,[]) %show RBS classification
        %}
        
        %save all areas and circularities
        Areas_all = [Areas_all, Areas];
        Circularities_all = [Circularities_all, Circularities];
        %by eye, for case #6, lesion = 22, #5 = 33, #4 = 10, #3 = 14, #2 = 20, #1 = 24. clunky code but gets the job done.
        if i == 1
            tLesion_number = 24;
            Centerx = 80; %center 
            Centery = 166;
        end
        if i == 2
            tLesion_number = 20;
            Centerx = 110;
            Centery = 233;
        end
        if i == 3
            tLesion_number = 14;
            Centerx = 57;
            Centery = 120;
        end
        if i == 4
            tLesion_number = 10;
            Centerx = 138;
            Centery = 87;
        end
        if i == 5
            tLesion_number = 33;
            Centerx = 73;
            Centery = 182;
        end
        if i == 6
            tLesion_number = 22;
            Centerx = 78;
            Centery = 226;
        end
        
        %save areas and circularities of true lesions only
        Areas_tlesions = [Areas_tlesions Areas(tLesion_number)];
        Circularities_tlesions = [Circularities_tlesions Circularities(tLesion_number)];
        
    
       
%% Experiment pt 4
    
        %Determining features for the 6 CT images (average, stdev, and contrast)
        Potential_lesions = nonzeros(unique(LabelArray_rbs));% nonzero labels remaining as potential_lesion numbers
        for Lesion_count = 1:length(Potential_lesions)% for every single potential lesion
            Lesion_number = Potential_lesions(Lesion_count); %for this specific lesion number
            [Lesion_average, Lesion_stdev, Lesion_contrast]= Feature_calcs(Image,LabelArray_attempt,Lesion_number);
            if Lesion_number == tLesion_number %if it's the true lesion
                LDA_features_t = [LDA_features_t Lesion_average Lesion_stdev Lesion_contrast];% true lesion features
            else
                LDA_features_f = [LDA_features_f Lesion_average Lesion_stdev Lesion_contrast]; % false lesion features
            end
        end
        
        %Classification based on distance criteria for the 6 CT images, used to calculate true and false positives based on distance
        TP = 0;
        FP = 0;
        for Lesion_count = 1:length(Potential_lesions) %for every single potential lesion
            Lesion_number = Potential_lesions(Lesion_count);%for this specific lesion number
            [row1, column1, dataval]=find(LabelArray_rbs == Lesion_number); %get pixels of that lesion
            Distance_x = abs(column1-Centerx); %get distances of pixels to true center
            Distance_y = abs(row1-Centery); %get distances of pixels to true center
            Distance_total = sqrt(min(Distance_x)^2 + min(Distance_y)^2); %get total distance of the closest pixel to the true center of the lesion
            if Distance_total <=10
                TP = TP + 1; %if it's close, it's a true positive
            else
                FP = FP + 1; %It's a false positive
            end
        end
        % the resulting values of TP and FP will be the number of true positives and false positives for each case.
        
        % LDA will be run on the other data in "Exp4.m", as according to Giger's Thursday lecture
            
    
        
    end %all cases have been run through
   
    % some plotting of all cases 
    
    %Showing Area vs circularity of all cases and true cases
    %{
    fig4 = figure;
    scatter(Areas_all, Circularities_all, 'filled'),
    hold on,
    scatter(Areas_tlesions, Circularities_tlesions,'filled'),
    hold on,
    xline(109.5),hold on, xline(580), hold on,
    yline(.21), hold on, yline(1), hold on
    xlim([0 700])
    ylim([0 1.1])
    xlabel('Area')
    ylabel('Circularity')
    saveas(fig4,fullfile('../Imaging 1 Practicum/','AreaVCirc.tif'));

    %}
    
    %Now plotting features in 2D graphs
    %{
    Avs= [];
    Stdevs= [];
    contrasts= [];
    tAvs= [];
    tStdevs = [];
    tcontrasts = [];
    for Count = 1:3:length(LDA_features_f)-2
        Avs = [Avs LDA_features_f(Count)];
        Stdevs = [Stdevs LDA_features_f(Count+1)];
        contrasts = [contrasts LDA_features_f(Count+2)];
    end
    
    for Count = 1:3:length(LDA_features_t)-2
        tAvs = [tAvs LDA_features_f(Count)];
        tStdevs = [tStdevs LDA_features_f(Count+1)];
        tcontrasts = [tcontrasts LDA_features_f(Count+2)];
    end
    
    fig1 = figure;
    scatter(Avs,Stdevs), hold on, scatter(tAvs,tStdevs)
    ylabel('stdev')
    xlabel('av')
    fig2 = figure;
    scatter(contrasts,Stdevs), hold on, scatter(tcontrasts, tStdevs)
    xlabel('contrast')
    ylabel('stdev')
    fig3 = figure;
    scatter(Avs,contrasts), hold on, scatter(tAvs,tcontrasts)
    xlabel('av')
    ylabel('contrasts')
    %}
    
    
    %now performing LDA on nodules and non-nodules... in the other code provided
    
%% Nested functions written for the script above are shown below.  
%input and output are described, and the code is annotated to describe purpose.
    
        %% nested function to check neighbors
        % input the image to be checked, the pixel of interest (m,n), and the size of the image (x,y) to set edges
        % output the total number of surrounding pixels (8 max, 0 min), the total number of neighboring pixels that have been counted before the current one (max 3, min 0), and the total number of neighboring pixels that are after (max 5, min 0)
            function [totalsum, priorsum,postsum] = checkneighbors(ImageChecked,m,n,x,y) %check surrounding 8 pixels to see if any are non-zero
                totalsum = -1; % number of ALL surrounding pixels occupied (-1 to cancel the center pixel already being 1)
                priorsum = 0; %number of previous pixels (so the row above and the one directly before the current pixel 
                for j = m-1:m+1 %x position (# of rows down)
                    for k = n-1:n+1 %y position i.e. column
                        if k>=1 && j >= 1 && k <=y && j <=x  %to avoid passing the perimeter of the image
                            totalsum = totalsum + ImageChecked(j,k); %add all the pixels.
                            if j == m-1 && k <=n %if the row of pixels being checked is above the current row
                                priorsum = priorsum + ImageChecked(j,k); %add any non-zero pixels
                            end
                            if k == n-1 && j <=m %for column directly before the current one 
                                priorsum = priorsum + ImageChecked(j,k); % add any non-zero pixel
                            end
                        end
                    end
                end
                postsum = totalsum - priorsum; %the number of pixels after the current pixel
            end
        
        %% nested function to check neighbors and get perimeters
        % input image of perimeter-only pixels, pixel of interes (m,n), and size of image (x,y) to set edges
        % output number of surrounding pixels occupied, number of pixels horizontal to pixel of interest (max 4, min 0), number of pixels diagonal to pixel of interest (max 4, min 0)
            function [totalsum, horvert,diag] = checkneighbors_perim(Perimeter_image,m,n,x,y) %check surrounding 8 pixels to see if any are non-zero
                totalsum = -1; % number of ALL surrounding pixels occupied (-1 to cancel the center pixel already being 1)
                horvert = -1; %number of previous pixels (so the row above and the one directly before the current pixel (-1 to cancel the center pixel) 
                diag = 0;
                for j = m-1:m+1 %x position (# of rows down)
                    for k = n-1:n+1 %y position i.e. column
                        if k>=1 && j >= 1 && k <=y && j <=x  %to avoid passing the perimeter of the image
                            %[CheckAll_perim, CheckPrior_perim, CheckPost_perim] = checkneighbors(Perimeter_image,j,k,x,y); %check if the neighboring perimeter of interest in als on on the perimeter
                            if j == m || k ==n %if it's the same row or column
                                horvert = horvert + Perimeter_image(j,k); %add any non-zero pixels, divided by 2 asssuming pixel is unit of 1.
                                totalsum = totalsum + Perimeter_image(j,k);
                            else
                                if j ~= m && k ~= n %if it is in a different row AND a different column (diagonal)
                                    diag = diag + Perimeter_image(j,k); % add any non-zero pixel
                                    totalsum = totalsum + Perimeter_image(j,k);
                                end
                            end
                        end
                    end
                end
            end
        
            %% nested function to check label array neighbors
            % input the firstpass labeled image, pixel of interest (m,n), and the size of the image (x,y) to set edges
            % output the minimum numbered label among neighbors (including self), and the maximum numbered label among neighbors (including self)
            function [minlabel,maxlabel] = checklabels(LabelArrayChecked,m,n,x,y)
                labels = []; %the 3 by 3 kernel
                for j = m-1:m+1 %x position (# of rows down)
                    for k = n-1:n+1 %y position i.e. column
                        if k>=1 && j >= 1 && k <=y && j <=x  %to avoid passing the perimeter of the image
                            if LabelArrayChecked(j,k) >1 %if it's greater than 1, so it is an actual label
                                labels =[labels, LabelArrayChecked(j,k)]; %add that label to the array of labeled neighbors
                            end
                        end
                    end
                end
                minlabel =min(labels); %get minimum label of surrounding neighbors. (is empty if no surrounding labels)
                maxlabel = max(labels);  %get maximum label of surrounding neighbors (is empty if no surrounding labels)
            end
    
            %% nested recursive function to go through image to even out the number of objects
            % input the LoG filtered and threshold image, empty checking-array to start with, labeled array after the second pass, and the size of the object (x,y)
            % output the new labeled image and the updated checking-array for recursion
            function  [LabelArray2nd,CheckPass2nd] = checkingsecondpass(graylevelimage2nd,CheckPass2nd,LabelArray2nd,x,y)
                %first go through and check all pixels that have a difference in labels between neighbors.
                for mm = 1:x %x position (# of rows down) (so travelling in horizontal raster order left to right, top to bottom)
                    for nn = 1:y %y position (# of columns to the right)
                        if LabelArray2nd(mm,nn) > 0 %if the pixel is an object
                            [CheckAll2nd, CheckPrior2nd, CheckPost2nd] = checkneighbors(graylevelimage2nd,mm,nn,x,y);
                            if CheckAll2nd > 0 %if there are neighbors
                                [minlabel2nd,maxlabel2nd] = checklabels(LabelArray2nd,mm,nn,x,y); %should never be empty, gets the range of pixels within neighbors
                                if maxlabel2nd-minlabel2nd >0 %if there is a difference between neighboring points... 
                                    CheckPass2nd(mm,nn)=1; %record that there is a difference.
                                else
                                    if CheckPass2nd(mm,nn)>0 %if there was previously a difference, and now there isn't
                                        CheckPass2nd(mm,nn)=0; %make it zero now, to update how many pixels we have to check now.
                                    end
                                end
                            end
                        end
                    end
                end
                
                %now with an updated display of values that are different
                if max(CheckPass2nd,[],'all') > 0 %if there are still pixels that are different...
                    [row, column, dataval]=find(CheckPass2nd == 1);%now find the pixels that do show a difference between different points.
                    for idx = 1:length(row) %now for all of these pixels that have different valued neighbors...
                        currentlabel = LabelArray2nd(row(idx),column(idx));% get the value of the pixel that has different neighbors
                        [minlabel2nd,maxlabel2nd] = checklabels(LabelArray2nd,row(idx),column(idx),x,y); %get the values and their difference
                        %now set all neighboring pixels to the 'min' label
                        for j = row(idx)-1:row(idx)+1 %x position (# of rows down)
                            for k = column(idx)-1:column(idx)+1 %y position i.e. column
                                if k>=1 && j >= 1 && k <=y && j <=x  %to avoid passing the perimeter of the image
                                    if LabelArray2nd(j,k)== maxlabel2nd %if the value of the pixel or any of its neighbors is the max
                                        LabelArray2nd(j,k)= minlabel2nd; %relabel it the min
                                    end
                                end
                            end
                        end
                    end
                    [LabelArray2nd,CheckPass2nd] = checkingsecondpass(graylevelimage2nd,CheckPass2nd,LabelArray2nd,x,y); %recursion, so if there are still values that are different... run through image again
                end
            end
       
        %% nested function to determine features
        %Average pixel value of lesion, stdev of lesion pixels, and contrast
        %input image, labeled array, and number of lesion of interest
        % output the average, stdev, and contrast of that numbered lesion.
    function [Lesion_average, Lesion_stdev, Lesion_contrast]= Feature_calcs(OriginalImage,Labelled_array,Lesion_number)
            Lesion_mask = Labelled_array== Lesion_number; %mask of just the lesion
            Lesion_segmented = bsxfun(@times, OriginalImage, cast(Lesion_mask,class(OriginalImage))); %get value from original image in lesion
            %Average of pixel values
            Lesion_average = (1/sum(Lesion_mask,'all'))*sum(Lesion_segmented,'all'); %mean written out in case we're not supposed to use it

            %STDev of pixel values
            Mean_subtract = Lesion_segmented - Lesion_average; %subtract average
            Mean_subtract2 = bsxfun(@times, Mean_subtract, cast(Lesion_mask,class(Mean_subtract))); %eliminate zeros
            Lesion_stdev = sqrt((1/sum(Lesion_mask,'all'))*sum(Mean_subtract2.^2,'all')); %stdev eq 

            %Contrast of pixel values
            Lesion_contrast = double(max(nonzeros(Lesion_segmented))-min(nonzeros(Lesion_segmented)));
    end
                
end
      
    

