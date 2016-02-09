%calculating standard deviation for each and every song with respect to genre
classical = std(transpose(fft_Val(1:100,:)));
country = std(transpose(fft_Val(101:200,:)));
jazz = std(transpose(fft_Val(201:300,:)));
metal = std(transpose(fft_Val(301:400,:)));
pop = std(transpose(fft_Val(401:500,:)));
rock = std(transpose(fft_Val(501:600,:)));

%selecting the best features which are in middle when sorted
Best=zeros(6,20);
[classical,Index] = sort(classical);Best(1,1:20)=Index(1,41:60);
[country,Index] = sort(country);Best(2,1:20)=Index(1,41:60);
[jazz,Index] = sort(jazz);Best(3,1:20)=Index(1,41:60);
[metal,Index] = sort(metal);Best(4,1:20)=Index(1,41:60);
[pop,Index] = sort(pop);Best(5,1:20)=Index(1,41:60);
[rock,Index] = sort(rock);Best(6,1:20)=Index(1,41:60);
for i = 1:6
    Best(i,:) = Best(i,:) + (i-1)*100;
end
Bests=zeros(1,120);
Bests(1,1:20) = Best(1,1:20);Bests(1,21:40) = Best(2,1:20);Bests(1,41:60) = Best(3,1:20);
Bests(1,61:80) = Best(4,1:20);Bests(1,81:100) = Best(5,1:20);Bests(1,101:120) = Best(6,1:20);

delta = zeros(6,120);   %delta marix which contains original genres for each song
delta(1,1:20)=1;delta(2,21:40)=1;delta(3,41:60)=1;
delta(4,61:80)=1;delta(5,81:100)=1;delta(6,101:120)=1;
labels = zeros(1,120);  %used to input for 10 fold cross validation to input for 10 sets
labels(1,1:20)=1;labels(1,21:40)=2;labels(1,41:60)=3;
labels(1,61:80)=4;labels(1,81:100)=5;labels(1,101:120)=6;

eta=0.01; %learning parameter
weight_best = zeros(6,1001);    %weights
accuracy_test_best=zeros(10,1); %Accuracy values for each fold
increase=1;

%10 FOLD CROSS VALIDATION
indices = crossvalind('kfold',labels,10);
training = zeros(1,108);
testing = zeros(1,12);

lambda = 0.001; %penalty term
confusion_best = zeros(6,6);    %confusion matrix
Probability = zeros(6,108);

for folds = 1 : 10
    col=1;
    cols=1;
    for i = 1 : 120
        if indices(i)~=folds    %seperating training data for each fold
            training(col)=i;
            col=col+1;
        else
            testing(cols)=i;    %seperating testing data for each fold
            cols=cols+1;
        end
    end

    %Training
    for epoch = 1 : 200     
        eta = eta/(1+(epoch/100));  %updating eta value for each epoch
        train_b=ones(108,1001);
        train_label=zeros(1,108);
        for i = 1 : 108
            train_b(i,:) = fft_Val(training(i),:);  %storing fft values for training data
            train_label(1,i) = labels(1,training(i));   %creating label to calculate accuracy
        end
        %NORMALIZATION for fft values
        for i = 1 : 1001
            maximum = max(train_b(:,i));
            train_b(:,i) = train_b(:,i)./maximum;
        end

        train_b= transpose(train_b);
        Probability = exp(weight_best * train_b);   %Logistic Regression
        %NORMALIZATION for Probability
        for i = 1 : 6
            total = sum(Probability(i,:));
            Probability(i,:) = Probability(i,:)/total;
        end
        
        %WEIGHT UPDATION USING GRADIENT DESCENT
        del = (delta(:,training(:))-Probability)*transpose(train_b);
        temp1 = eta .* (del - (lambda.*weight_best));
        weight_best = weight_best+temp1;
    end

    %calculating accuracy
    [maximum_train,index_train] = max(Probability);
    acc=0;
    for i = 1 : 108
        if(train_label(i)==index_train(i))
            acc=acc+1;
        end
        accuracy_train_b = acc/108;
    end
    
    %TESTING
    test_label = zeros(1,12);
    test_b = ones(12,1001);
    for i = 1 : 12
        test_b(i,:) = fft_Val(testing(i),:);    %storing fft values for testing data
        test_label(1,i) = labels(1,testing(i)); %creating label for calculating accuracy
    end
    %NORMALIZATION for fft values
    for i = 1 : 1001
        maximum = max(test_b(:,i));
        test_b(:,i) = test_b(:,i)./maximum;
    end

    test_b = transpose(test_b);
    temp_test = exp(weight_best * test_b);  %Logistic Regression
    %NORMALIZATION for probability
    for song = 1 : 6
        total = sum(temp_test(song,:));
        temp_test(song,:) = temp_test(song,:)/total;
    end
    %calculating accuracy
    [maximum_test, index_test] = max(temp_test);
    acc=0;
    for i = 1 : 12
        if(test_label(i)==index_test(i))
            acc=acc+1;
        end
    end
    accuracy_test_best(increase) = acc/12;
    increase=increase+1;
    
    %CONFUSION MATRIX
    confusion_best(:,:) = confusion_best(:,:)+confusionmat(test_label,index_test);
end