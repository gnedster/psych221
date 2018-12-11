files = dir('~/Stanford/f19/psych221/output');

ip = ipCreate;

for num = 3:numel(files)
    filename = files(num).name;
    fileparts = strsplit(files(num).name,'.');
    nameparts = strsplit(fileparts{1},'_');
    
    load(strcat('~/Stanford/f19/psych221/trainingdata/', nameparts{1}, '_high.mat'));
    
%     load(strcat('~/Stanford/f19/psych221/output/', filename));
%     sensorH.data.volts = volts;
    ipS = ipCompute(ip,sensorH);

    save(strcat(nameparts{1}, '_high_ip'), 'ipS')
end
