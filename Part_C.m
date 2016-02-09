delta = zeros(6,600);       %delta matrix which contains original genres for the song
delta(1,1:100)=1;delta(2,101:200)=1;delta(3,201:300)=1;
delta(4,301:400)=1;delta(5,401:500)=1;delta(6,501:600)=1;
labels = zeros(1,600);      %used to input for 10 fold cross validation to input for 10 sets
labels(1,1:100)=1;labels(1,101:200)=2;labels(1,201:300)=3;
labels(1,301:400)=4;labels(1,401:500)=5;labels(1,501:600)=6;


accuracy_test_mfcc=zeros(10,1); %Accuracy values for each fold
increase=1;
lambda = 0.001; %penalty term
confusion_mfcc = zeros(6,6);    %confusion matrix
Probability = zeros(6,540);

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
            testing(cols)=i;    %seperating testing data for each 
            cols=cols+1;
        end
    end

    %TRAINING
    %zeroing weights for every fold
    eta=0.01;   %learning rate
    weight_mfcc = zeros(6,14);  %weights
    for epoch = 1 : 100
        eta = eta/(1+(epoch/100));  %updating eta value with every epoch
        train_mfcc=ones(14,540);
        train_label=zeros(1,540);
        for i = 1 : 540
            train_mfcc(:,i) = CC_Val(:,training(i));    %storing mfcc values for training data
            train_label(1,i) = labels(1,training(i));   %creating label to calculate accuracy
        end
        %NORMALIZATION for mfcc values
        for i = 1 : 540
            maximum = max(train_mfcc(:,i));
            train_mfcc(:,i) = train_mfcc(:,i)./maximum;
        end

        Probability = exp(weight_mfcc * train_mfcc);    %Logistic Regression
        %Normalization for probability
        for i = 1 : 6
            total = sum(Probability(i,:));
            Probability(i,:) = Probability(i,:)/total;
        end
        
        %WEIGHT UPDATION USING GRADIENT DESCENT
        del = (delta(:,training(:))-Probability)*transpose(train_mfcc);
        temp1 = eta .* (del - (lambda.*weight_mfcc));
        weight_mfcc = weight_mfcc+temp1;
    end
    
    %calculating accuracy
    [maximum_train,index_train] = max(Probability);
    acc=0;
    for i = 1 : 540
        if(train_label(i)==index_train(i))
            acc=acc+1;
        end
        accuracy_train = acc/540;
    end
    
    %TESTING
    test_label = zeros(1,60);
    test_mfcc = ones(60,14);
    for i = 1 : 60
        test_mfcc(i,:) = CC_Val(:,testing(i));  %storing mfcc values for testing data
        test_label(1,i) = labels(1,testing(i)); %creating label to calculate accuracy
    end
    %NORMALIZATION for mfcc values
    for i = 1 : 14
        maximum = max(test_mfcc(:,i));
        test_mfcc(:,i) = test_mfcc(:,i)./maximum;
    end
    
    test_mfcc = transpose(test_mfcc);
    temp_test = exp(weight_mfcc * test_mfcc);   %Logistic Regression
    %NORMALIZATION for probability
    for song = 1 : 6
        total = sum(temp_test(song,:));
        temp_test(song,:) = temp_test(song,:)/total;
    end
    %calculating accuracy
    [maximum_test, index_test] = max(temp_test);
    acc=0;
    for i = 1 : 60
        if(test_label(i)==index_test(i))
            acc=acc+1;
        end
    end
    accuracy_test_mfcc(increase) = acc/60;
    increase=increase+1;
    
    %CONFUSION MATRIX
    confusion_mfcc(:,:) = confusion_mfcc(:,:)+confusionmat(test_label,index_test);
end