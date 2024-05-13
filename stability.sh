# 

cd /scratch
mkdir medleydb
cd medleydb
aws s3 sync s3://stability-aws/MedleyDB ./
tar -xvf MedleyDB_v1.tar
tar -xvf MedleyDB_v2.tar