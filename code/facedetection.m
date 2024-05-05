clc;
clear all;
tic;
whos;



while (1==1) 
    
    choice=menu('Face Detection',...
    'Face Database',...
    'Non Face Database',...
    'Test Database',...
    'Train Test',...
    'Exit');

    if (choice == 1)
        clc;
        clear all;
        choice = 1;
        
        pause(0.01);
        folder_name = uigetdir;
        filenames       =   dir(fullfile(folder_name, '*.jpg'));           %   Read all images with specified extention, its jpg in our case
        total_images    =   numel(filenames);                              %   Count total number of photos present in that folder

        D = zeros(total_images,24);

        for i = 1:total_images

            full_name= fullfile(folder_name, filenames(i).name);           %   It will specify images names with full path and extension
            I = imread(full_name);                                         %   Read images
            fprintf('...\n..\n.  Image %d\n ',i);

            % Extract feature        
            features = sfta(I, 4);                                         %    sfta features
            D(i,1:24) = features(1:24);    
        end    
        AllData = table(D);
        writetable(AllData,['face.xlsx'],'Sheet',1); 
    end

    if (choice == 2)
        clc;
        clear all;
        choice = 2;
        
        pause(0.01);
        folder_name = uigetdir;
        filenames = dir(fullfile(folder_name, '*.jpg'));                   %   Read all images with specified extention, its jpg in our case
        total_images = numel(filenames);                                   %   Count total number of photos present in that folder

        D = zeros(total_images,24);

        for i = 1:total_images

            full_name= fullfile(folder_name, filenames(i).name);           %   It will specify images names with full path and extension
            I = imread(full_name);                                         %   Read images
            fprintf('...\n..\n.  Image %d\n ',i);

            % Extract feature
            features = sfta(I, 4);
            D(i,1:24)=features(1:24);
        end
        AllData = table(D);
        writetable(AllData,['non-face.xlsx'],'Sheet',1);  
    end

    if (choice == 3)
        clc;
        clear all;
        choice = 3;
        
        pause(0.01);              
        folder_name = uigetdir;
        filenames       =   dir(fullfile(folder_name, '*.jpg'));           %   Read all images with specified extention, its jpg in our case
        total_images    =   numel(filenames);                              %   Count total number of photos present in that folder

        D = zeros(total_images,24);
        
        for i = 1:total_images

            full_name= fullfile(folder_name, filenames(i).name);           %   It will specify images names with full path and extension
            I = imread(full_name);                                         %   Read images
            fprintf('...\n..\n.  Image %d\n ',i);
            testfile_names{i} = full_name;
            
            % Extract feature
            features = sfta(I, 4);                                         % SFTA feature -- features = extractHOGFeatures(I); %HOG features
            D(i,1:24)=features(1:24);
        end
        AllData = table(D);
        writetable(AllData,['testset.xlsx'],'Sheet',1);
        save('testFileNames.mat','testfile_names');
    end

    if(choice == 4)
    
        clc;
        clear all;
        choice = 4;
        
        % load data
        Face = xlsread('face.xlsx');
        Non_face = xlsread('non-face.xlsx');
        [row, col] = size(Face);
        [row2, col2] = size(Non_face);

        % creating training group
        TT1=Face(1:row,1:col);
        TT2=Non_face(1:row2,1:col2);
        TT_Set = [TT1;TT2];
        
        TX1 = ones(row,1);
        TX2 = ones(row2,1)*2;
        TTgroup = [TX1;TX2];

        %creating testing group
        load('testFileNames.mat');
        result = testfile_names';
        testset=xlsread('testset.xlsx');
        [m,n]=size(testset);        
        facetest = testset(1:m,1:n);

        
        % training using SVM
        SVMStruct = svmtrain(TT_Set,TTgroup, 'kernel_function', 'rbf');
        
        %testing using SVM model
        Group = svmclassify(SVMStruct,facetest);
        for i=1:1:size(Group,1)
            result{i,2} = Group(i);
            
            if(Group(i)==1)
                result{i,3} = 'Face';
            else
                result{i,3} = 'Non-Face';
            end
            
        end
        
        % saving results
        save('SVMStruct.mat','SVMStruct');
        save('Result.mat','result');
        
        AllData = table(result);
        writetable(AllData,['Result.xlsx'],'Sheet',1);
        
        fprintf('Finished! please check Result.mat or Result.xlsx file');
    end

    if (choice == 5)
        clc;
        clear all;
        choice = 5;
        
        clear all;
        clc;
        close all;
        return;
    end

end






