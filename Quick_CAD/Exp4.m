%% Written by Mira Liu for Imaging Practicum 1 2021, CAD Lab
% loads nodule and non-nodule data and uses LDA to create FROC curve and demonstrate feature difference with 2D plots 
function [discInfo] = Exp4
    tnodules = load('nodule.dat','ascii'); %true nodules
    fnodules=load('non-nodule.dat','ascii'); %false nodules
    
    %extracting features from columns for plotting
    tfeat1 = tnodules(:,1);
    tfeat2 = tnodules(:,2);
    tfeat3 = tnodules(:,3);
    tfeat4 = tnodules(:,4);
    
    ffeat1 = fnodules(:,1);
    ffeat2 = fnodules(:,2);
    ffeat3 = fnodules(:,3);
    ffeat4 = fnodules(:,4);
    
    %this maximises the distance between the means (Fisher's linear discriminant)
    %the outputs are the distances from the determined separation plane?? 
    
%% 2D scatter plots of features, their bisectors, and the corresponding discriminant
    % now show the separation planes in 2D plots (feature A vs feature B)
    
    p1 = [mean(tfeat4), mean(tfeat3)]; %the means of the trues of feature A and feature B
    p2 = [mean(ffeat4), mean(ffeat3)]; %the means of the falses of feature A and feature B
    
    range = 10;
    [xout,yout,xout_perp,yout_perp] = line_eq(p1,p2,range); %draw bisector and line of discrimination using nested function "line_eq"
    
    %plot figure
    fig1 = figure;
    scatter(ffeat4,ffeat3,'magenta'), hold on, %plot the false (PINK)
    scatter(tfeat4,tfeat3,'blue'), hold on, %plot the true (BLUE)
    scatter(p1(1),p1(2),'green'), hold on, %plot mean of the true (GREEN)
    scatter(p2(1),p2(2),'red'), hold on, %plot mean of the false (RED)
    plot(xout,yout,'black') %this is the bisector between the mean of true and false
    plot(xout_perp,yout_perp,'black') %this is the division line! 
    xlabel('feature 4')
    ylabel('feature 3')
    
    xlim([min(min(ffeat4),min(tfeat4)) max(max(ffeat4),max(tfeat4))]) %just to determine range.
    ylim([min(min(ffeat3),min(tfeat3)) max(max(ffeat3),max(tfeat3))]) %just to determine range.
    
    %below was to check that I was understanding the interpretation of fout and tout, ignore it.
    %{
    FPs = 0;
    check = 0;
    for j=1:length(ffeat1)
        [d1,d2]=distance_eq(p1,p2,[ffeat1(j),ffeat4(j)]); %this should match fout, as it's the distance between all the false and the discrimant
        if d1<d2 %it is 'positive'
            FPs = FPs+1; %so if it's closer to mean 1 (which is the true nodule means), it will be labled as true, even though it's a non-nodule
        end
        if fout(j) < 0 %if it's negative it's deemed positive.
            check = check+1;
        end
    end
    FPs
    check
    
    TPs = 0;
    check2 = 0;
    for j=1:length(tfeat1)
        [d1,d2]=distance_eq(p1,p2,[tfeat1(j),tfeat4(j)]); %this should match fout, as it's the distance between all the false and the discrimant
        if d1<d2 %it is 'positive'
            TPs = TPs+1; %so if it's closer to mean 1 (which is the true nodule means), it will be labled as true, and it is a nodule
        end
        if tout(j) < 0 %if it's larger, it's deemed positive?
            check2 = check2+1;
        end
    end
    TPs
    check2
    %}
    
%% FROC with LDA
    %ok so can use the above as an example in 2d space.... now do it with all features
    
    [tout fout, discInfo] = LinDisc(tnodules, fnodules, [2 3])
    
    %first get range of threshold values (so min of tout and fout to max of tout and fout)
    min_k = min(min(tout),min(fout));
    max_k = max(max(tout),max(fout));
    %first get the sensitivities
    Sensitivities = [];
    for k = linspace(min_k-.01,max_k,25) %25 points on roc curve from min to max
        [row column data] = find(tout<=k); %find number of values below k (which would then be classified as positive)
        Sensitivity = length(row)/length(tout);
        Sensitivities = [Sensitivities Sensitivity]; %add it to list
    end
    %now get the corresponding false positives using the same thresholds... 
    FPs = [];
    for k = linspace(min_k-.01,max_k,25) %25 points on roc curve from min to max
        [row column data] = find(fout<=k); %find number of values below k
        FPs = [FPs length(row)];
    end
    
    %also get what threshold 0 is (should be the 'best' in some ways)
    [row column data] = find(tout<=0);
    sensitivity_0 = length(row)/length(tout);
    [row column data] = find(fout<=0);
    FP_0 = length(row);
    
    %FROC curve!
    figure,
    scatter(FPs,Sensitivities), %this is an fROC!
    hold on,
    scatter(FP_0, sensitivity_0),
    hold on,
    guess_x = linspace(0, 33,20);
    guess_y = linspace(0,1,20);
    plot(guess_x,guess_y,'black')
    xlim([0 length(fout)])
    ylim([0 1])
    xlabel('FP')
    ylabel('Sensitivity')
    
    
%% including LinDisc as a nested function and annotating 
    function [tout,fout,discInfo]=LinDisc(trueData,falseData,feaList)
        tx=trueData(:,feaList); %extract features from the columns, (so feature 1,2,3 &4)
        fx=falseData(:,feaList);
        [n1,p]=size(tx); %n1 representing the number of cases, p representing the number of features
        [n2,p]=size(fx);
        discInfo.meanT = mean(tx); %mean of the four true features 
        discInfo.meanF = mean(fx); %mean of the four false features
        covt=cov(tx); 
        covf=cov(fx);
        discInfo.cv=((n1-1)*covt+(n2-1)*covf)/(n1+n2-2);
        discInfo.cv=inv(discInfo.cv);
        tx=tx';
        fx=fx';
        discInfo.meanT=discInfo.meanT';
        discInfo.meanF=discInfo.meanF';


        tout = zeros(n1,1); %
        for i=1:n1, %for each of the cases
         tout(i,1)=(discInfo.meanF-discInfo.meanT)'*discInfo.cv*tx(:,i)-...
             .5*(discInfo.meanF-discInfo.meanT)'*discInfo.cv*...
             (discInfo.meanF+discInfo.meanT);
        end

        fout = zeros(n2,1);
        for i=1:n2,
         fout(i,1)=(discInfo.meanF-discInfo.meanT)'*discInfo.cv*fx(:,i)-...
             .5*(discInfo.meanF-discInfo.meanT)'*discInfo.cv*...
             (discInfo.meanF+discInfo.meanT);
        end
        %larger values for negative, and smaller values for positive??
    end

%% including LinDisc as a nested function and annotating 
%input true data, false date, feature list
%outputs distances from linear discriminant
    function [tout,fout,discInfo]=LinDisc_threshold(trueData,falseData,feaList)
        tx=trueData(:,feaList); %extract features from the columns, (so feature 1,2,3 &4)
        fx=falseData(:,feaList);
        [n1,p]=size(tx); %n1 representing the number of cases, p representing the number of features
        [n2,p]=size(fx);
        
        %discInfo.meanT = mean(tx); %mean of the four true features 
        %discInfo.meanF = mean(fx); %mean of the four false features
        covt=cov(tx); 
        covf=cov(fx);
        discInfo.cv=((n1-1)*covt+(n2-1)*covf)/(n1+n2-2);
        discInfo.cv=inv(discInfo.cv);
        tx=tx';
        fx=fx';
        discInfo.meanT=discInfo.meanT';
        discInfo.meanF=discInfo.meanF';
        
        

        tout = zeros(n1,1); %
        for i=1:n1, %for each of the cases
         tout(i,1)=(discInfo.meanF-discInfo.meanT)'*discInfo.cv*tx(:,i)-...
             .5*(discInfo.meanF-discInfo.meanT)'*discInfo.cv*...
             (discInfo.meanF+discInfo.meanT);
        end

        fout = zeros(n2,1);
        for i=1:n2,
         fout(i,1)=(discInfo.meanF-discInfo.meanT)'*discInfo.cv*fx(:,i)-...
             .5*(discInfo.meanF-discInfo.meanT)'*discInfo.cv*...
             (discInfo.meanF+discInfo.meanT);
        end
        %larger values for negative, and smaller values for positive??
    end
    
%% equation of getting perpendicular bisector (so drawing the separation plane)
%input two points and the range of x values wanted
%output the line (x and y values) conencting the means and the perpendicular bisector.
    function [xout,yout,xout_perp,yout_perp] = line_eq(p1,p2,range)
        slope = (p2(2)-p1(2))/(p2(1)-p1(1)); %y2-y1/x2-x1
        perp_slope = -1/slope; %opposite reciprocal
        perp_point = [(p1(1)+ p2(1))/2, (p1(2)+p2(2))/2]; %average x, average y, to get the bisector
        yout = [];
        xout = [];
        yout_perp = [];
        xout_perp = [];
        for x=perp_point(1)-range:range/10:perp_point(1)+range
            y = slope*(x-perp_point(1)) + perp_point(2); %y -y1 = m(x-x1)
            xout = [xout x];
            yout = [yout y];
            y = perp_slope*(x-perp_point(1)) + perp_point(2); %y -y1 = m(x-x1)
            xout_perp = [xout_perp x];
            yout_perp = [yout_perp y];
        end
    end

%% now checking what it has been classified as (mean1 and mean 2 being means of the 2 features for true and false, p being a point of interest...
%calculate distance between mean of true and false values for classification
    function [dist1,dist2] = distance_eq(meantrue,meanfalse,p)
        dist1 = sqrt((meantrue(1)-p(1))^2+(meantrue(2)-p(2))^2); %d = sqrt[(y1-y2)^2 + (x1-x2)^2]
        dist2 = sqrt((meanfalse(1)-p(1))^2+(meanfalse(2)-p(2))^2);
        %so whichever distance is smaller, that point will be classified as
    end
end