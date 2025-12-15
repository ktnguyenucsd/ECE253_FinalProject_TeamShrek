net = denoisingNetwork("DnCNN");

inputFolder  = "C:/Users/khoa/person_seg_project/best_images/images_with_person_saltpepper";
outputFolder = "images_with_person_saltpepper_dncnn_med";

if ~exist(outputFolder, "dir")
    mkdir(outputFolder);
end

imgFiles = dir(fullfile(inputFolder, "*.jpg"));
imgFiles = [imgFiles; dir(fullfile(inputFolder, "*.png"))];
imgFiles = [imgFiles; dir(fullfile(inputFolder, "*.jpeg"))];


for k = 1:numel(imgFiles)
    filename  = imgFiles(k).name;
    inPath    = fullfile(inputFolder, filename);
    outPath   = fullfile(outputFolder, filename);

    I = imread(inPath);

    
    if ndims(I) == 3
        I_med = zeros(size(I), 'like', I);
        for c = 1:3
            I_med(:,:,c) = medfilt2(I(:,:,c), [3 3]);   
        end
    else
        I_med = medfilt2(I, [3 3]);
    end

    
    if ndims(I_med) == 3
        I_denoised = zeros(size(I_med), 'like', I_med);
        for c = 1:3
            I_denoised(:,:,c) = denoiseImage(I_med(:,:,c), net);
        end
    else
        I_denoised = denoiseImage(I_med, net);
    end

    imwrite(I_denoised, outPath);

end


