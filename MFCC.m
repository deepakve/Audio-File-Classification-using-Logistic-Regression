folder ={'C:\Users\Deepak\Downloads\MLHW3\data\classical','C:\Users\Deepak\Downloads\MLHW3\data\country', 'C:\Users\Deepak\Downloads\MLHW3\data\jazz', 'C:\Users\Deepak\Downloads\MLHW3\data\metal', 'C:\Users\Deepak\Downloads\MLHW3\data\pop', 'C:\Users\Deepak\Downloads\MLHW3\data\rock'}; 
CC_Val = ones(14,600);
row=1;
ccc = zeros(13,1);
Tw = 25;           % analysis frame duration (ms)
Ts = 10;           % analysis frame shift (ms)
alpha = 0.97;      % preemphasis coefficient
R = [ 300 3700 ];  % frequency range to consider
M = 20;            % number of filterbank channels 
C = 13;            % number of cepstral coefficients
L = 22;    


for g = 1 : max(length(folder)) %Reading for each genre
    List = dir(folder{g});  %stores all the files present in that particular folder
    %as 1 stores current path directory and 2 stores parent path directory
    %we start looping names for 3
    for k = 3 : 102
        name = strcat(folder{g},'\',List(k).name);
        [y,fs]=audioread(name); %reading the audio file
        [cc, fbe, frames] = mfcc(y,fs,Tw,Ts,alpha,@hamming,R,M,C,L);    %mfcc values for each audio file
        cc = cc(:,301:2693);
        for c = 1 : 13
            ccc(c,1) = mean(cc(c,:));
        end
        row = k+((g-1)*100)-2;
        CC_Val(2:14,row) = ccc; %storing mfcc values into a matrix
    end
end






