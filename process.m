files = dir('~/Stanford/f19/psych221/rawdata/train2017');

for num = 3:numel(files)
    filename = files(num).name;
    fileparts = strsplit(files(num).name,'.');
    scene = sceneFromFile(strcat('~/Stanford/f19/psych221/rawdata/train2017/', filename), 'RGB');

    % Treat the scene as small
    scene = sceneSet(scene,'fov',fov);
    % ieAddObject(scene); 
    % sceneWindow;

    oi = oiCreate;
    oi = oiCompute(oi,scene);
    % ieAddObject(oi); oiWindow;

    %% Create a low (sensorL) and high (sensorH) resolution sensor

    sensor = sensorCreate;
    sz = sensorGet(sensor,'pixel size');  % Row Col
    sensorL = sensorSetSizeToFOV(sensor,fov);

    % Make the pixel half the size
    sensorH = sensorSet(sensorL,'pixel size same fill factor',sz(1)/2);
    sensorH = sensorSetSizeToFOV(sensorH,fov);

    %%  Compute the low and high resolution sensor images

    sensorH = sensorCompute(sensorH,oi);
    sensorL = sensorCompute(sensorL,oi);

    save(strcat(fileparts{1}, '_low'), 'sensorL')
    save(strcat(fileparts{1}, '_high'), 'sensorH')
end
