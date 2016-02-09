folder ='data'; 
genre = {'classical','country','jazz','metal','pop','rock'};
fft_Val= ones(600,1001);
for g = 1 : max(length(genre))      %Reading for each genre
    fold = strcat(folder,'\',genre{g});
    List = dir(fold);   %stores all the files present in that particular folder into list
    %as 1 stores current path directory and 2 stores parent path directory
    %we start looping names for 3
    for names = 3 : max(length(List))
        name = strcat(fold,'\',List(names).name);
         [data,rate]=audioread(name);   %reading the audio file
         row=names+((g-1)*100)-2;       %as loop starts and #of songs=600, i updated it for every 100
         %multiplied by 100 to increment to 101.... and -2 as names start
         %from 3
         fft_Val(row,2:1001)=fft(data(1:1000)); %storing only first 1000 values into a matrix
    end
end