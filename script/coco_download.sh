mkdir -p ../dataset/coco

cd ../dataset/coco
wget http://images.cocodataset.org/zips/val2014.zip
wget http://images.cocodataset.org/zips/train2014.zip
wget https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip

unzip *.zip

mkdir -p karpathy
mv dataset_coco.json karpathy/dataset_coco.json