for DATASET in "CIFAR-10" "CIFAR-100" "MNIST" "SIGN";
do	

# generate the dataset distribution
python -m src.data_division.split_data 40 $DATASET 5.0

# run the evalution with a single dataset
python -m src.centralized.torch.main -ds=$DATASET

# remove old file to avoid full disk
rm -rf datasets/$DATASET/distributions

done

