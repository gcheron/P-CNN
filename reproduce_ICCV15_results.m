nb_classes=21;

acc = zeros(1,3) ;
for s = 1:3 % for each split
    % load kernels and labels
    labels = load(sprintf('ICCV15_results/labels_split%d.mat',s)) ; labels = labels.labels ;
    kernel_test = load(sprintf('ICCV15_results/ktest_split%d.mat',s)) ; kernel_test=kernel_test.Ktest ;
    
    svms=dir(sprintf('ICCV15_results/SVMs/JHMDB_*_test%d.mat',s));
    scores=[];
    for c=1:nb_classes
        % load SVMs
        svm = load(sprintf('ICCV15_results/SVMs/%s',svms(c).name)) ; svm=svm.svm ;
        % test
        conf=kernel_test(:,svm.sv_indices) * svm.sv_coef - svm.rho ;
        scores=[scores conf] ;
    end
    assert(sum(sum(labels==1,2)==1)==size(labels,1)) ; % only one label per example
    assert(size(labels,1)==size(scores,1) && size(labels,2)==size(scores,2) && nb_classes==size(scores,2)) ;
    
    [~,GT_classes] = max(labels,[],2);
    [~,test_classes] = max(scores,[],2);
    
    TP=sum(GT_classes==test_classes) ;
    accuracy = TP/length(GT_classes) ;
    acc(s)=accuracy ;
end
fprintf('\nP-CNN accuracy on the 3 splits of JHMDB is %.1f\n',100*mean(acc));

