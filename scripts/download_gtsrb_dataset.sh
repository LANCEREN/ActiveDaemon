data=/mnt/data03/renge/public_dataset/pytorch
mkdir $data/gtsrb-data;
mkdir $data/gtsrb-data/Train;
mkdir $data/gtsrb-data/Test;
mkdir $data/temps;
mkdir $data/temps/Train;
mkdir $data/temps/Test;

wget -P $data https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip
wget -P $data https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip
wget -P $data https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip

unzip $data/GTSRB_Final_Training_Images.zip -d $data/temps/Train;
unzip $data/GTSRB_Final_Test_Images.zip -d $data/temps/Test;
mv $data/temps/Train/GTSRB/Final_Training/Images/* $data/gtsrb-data/Train;
mv $data/temps/Test/GTSRB/Final_Test/Images/* $data/gtsrb-data/Test;
unzip $data/GTSRB_Final_Test_GT.zip -d $data/gtsrb-data/Test/;
rm -r $data/temps;
rm $data/*.zip;
echo "Download Completed";