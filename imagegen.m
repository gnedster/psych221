files = dir('~/Stanford/f19/psych221/postprocess');

ip = ipCreate;

for num = 3:numel(files)
    filename = files(num).name;
    fileparts = strsplit(files(num).name,'.');
    nameparts = strsplit(fileparts{1},'_');
    
    load(strcat('~/Stanford/f19/psych221/postprocess/', files(num).name));
    fullName = strcat('~/Stanford/f19/psych221/images/', fileparts{1}, '.png');
    img = imageShowImage(ipS, 1, true,0);
    imwrite(img,fullName);
    fprintf('Saved image file %s\n',fullName);
end


