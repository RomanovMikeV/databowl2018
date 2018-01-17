cd ..
kg config -c data-science-bowl-2018
cd data
kg download
unzip stage1_sample_submission.csv.zip
rm stage1_sample_submission.csv.zip
unzip stage1_test.zip -d test
rm stage1_test.zip
unzip stage1_train.zip -d train
rm stage1_train.zip
unzip stage1_train_labels.csv.zip 
rm stage1_train_labels.csv.zip 
echo "Completed"