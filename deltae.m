ieInit

%% Overview of the function call

files = dir('~/Stanford/f19/psych221/postprocess');

ip = ipCreate;

filenames = {};

for num = 3:numel(files)
    nameparts = strsplit(files(num).name,'_');
    filenames = [filenames, nameparts{1}];
end

filenames = unique(filenames);
nearest_acc = 0;
bilinear_acc = 0; 
cnn_acc = 0;
    
for num = 1:numel(filenames)  
    % Set up to read an image and a JPEG compressed version of it
    filehigh = fullfile('/Users/edng/Stanford/f19/psych221/images', strcat(filenames(num), '_high_ip.png'));
    filenearest = fullfile('/Users/edng/Stanford/f19/psych221/images', strcat(filenames(num), '_nearest_ip.png'));
    filebilinear = fullfile('/Users/edng/Stanford/f19/psych221/images', strcat(filenames(num), '_bilinear_ip.png'));
    filecnn = fullfile('/Users/edng/Stanford/f19/psych221/images', strcat(filenames(num), '_cnn_ip.png'));

    % We will treat the two images as if they are on a CRT display seen from 12
    % inches.
    vDist = 0.3;          % 12 inches
    dispCal = 'crt.mat';   % Calibrated display

    %% Spatial scielab reads the files and display

    % Convert the RGB files to a scene and then call scielabRGB
    % The returns are an error image and two scenes containing the two images
    % The display variable is the implicit display we used to transform the RGB
    % images into the spectral image.  It does nt play a further role.
    [eImage,scene1,scene2,disp] = scielabRGB(filehigh{1}, filenearest{1}, dispCal, vDist);
    nearest_acc = nearest_acc + mean(eImage(:));
    
    [eImage,scene1,scene2,disp] = scielabRGB(filehigh{1}, filebilinear{1}, dispCal, vDist);
    bilinear_acc = bilinear_acc + mean(eImage(:)); 
    
    [eImage,scene1,scene2,disp] = scielabRGB(filehigh{1}, filecnn{1}, dispCal, vDist);
    cnn_acc = cnn_acc + mean(eImage(:));
end