delta = zeros(6,600);   %delta matrix which contains the original genres for the songs
delta(1,1:100)=1;delta(2,101:200)=1;delta(3,201:300)=1;
delta(4,301:400)=1;delta(5,401:500)=1;delta(6,501:600)=1;
labels = zeros(1,600);  %used to input for 10 fold cross validation to seperate into 10 sets
labels(1,1:100)=1;labels(1,101:200)=2;labels(1,201:300)=3;
labels(1,301:400)=4;labels(1,401:500)=5;labels(1,501:600)=6;


lambda = 0.001; %penality term
accuracy_test_fft=zeros(10,1);  %Accuracy values for each fold  
increase=1;
confusion_fft = zeros(6,6); %confusion matrix

probability = zeros(6,540);

%10 FOLD CROSS VALIDATION
indices = crossvalind('kfold',labels,10);
training = zeros(1,540);
testing = zeros(1,60);

for folds = 1 : 10
    col=1;
    cols=1;
    for i = 1 : 600
        if indices(i)~=folds    %seperating training data for each fold
            training(col)=i;
            col=col+1;
        else
            testing(cols)=i;    %seperating testing data for each fold
            cols=cols+1;
        end
    end

    %TRAINING
    %zeroing weights for every fold
    eta=0.01;   %learning rate
    weight_fft = zeros(6,1001); %weights
    
    for epoch = 1 : 100     %#iterations for each fold
        eta = eta/(1+(epoch/100));      %updating eta value for each fold
        train_fft=ones(540,1001);
        train_label=zeros(1,540);
        for i = 1 : 540
            train_fft(i,:) = fft_Val(training(i),:);    %storing fft values for training data
            train_label(1,i) = labels(1,training(i));   %creating label to calculate accuracy
        end
        %NORMALIZATION for fft values
        for i = 1 : 1001
            maximum = max(train_fft(:,i));
            train_fft(:,i) = train_fft(:,i)./maximum;
        end
        train_fft= transpose(train_fft);
        probability = exp(weight_fft * train_fft);  %Logistic Regression
        %NORMALIZATION for probability
        for i = 1 : 6
            total = sum(probability(i,:));
            probability(i,:) = probability(i,:)/total;
        end
        
        %WEIGHT UPDATION USING GRADIENT DESCENT
        del = (delta(:,training(:))-probability)*transpose(train_fft);
        temp1 = eta .* (del - (lambda.*weight_fft));
        weight_fft = weight_fft+temp1;
    end

    %calculating Accuracy
    [maximum_train,index_train] = max(probability);
    acc=0;
    for i = 1 : 540
        if(train_label(i)==index_train(i))
            acc=acc+1;
        end
        accuracy_train_fft = acc/540;
    end
    
    %TESTING 
    test_label = zeros(1,60);
    test_fft = ones(60,1001);
    for i = 1 : 60
        test_fft(i,:) = fft_Val(testing(i),:);  %storing fft values of testing data for each fold
        test_label(1,i) = labels(1,testing(i)); %creating label to calculate accuracy
    end
    %NORMALIZATION for fft values
    for i = 1 : 1001
        maximum = max(test_fft(:,i));
        test_fft(:,i) = test_fft(:,i)./maximum;
    end

    test_fft = transpose(test_fft);
    temp_test = exp(weight_fft * test_fft); %classifying 
    %NORMALIZATION for classification
    for song = 1 : 6
        total = sum(temp_test(song,:));
        temp_test(song,:) = temp_test(song,:)/total;
    end
    [maximum_test, index_test] = max(temp_test);
    acc=0;
    %calculating accuracy
    for i = 1 : 60
        if(test_label(i)==index_test(i))
            acc=acc+1;
        end
    end
    accuracy_test_fft(increase) = acc/60;
    increase=increase+1;
    
    %CONFUSION MATRIX
    confusion_fft(:,:) = confusion_fft(:,:)+confusionmat(test_label,index_test);
end