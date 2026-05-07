#!/bin/bash

set -e  # stop if anything fails

DATA_DIR="project/data/raw"

mkdir -p "$DATA_DIR"

echo "🚀 Downloading Yelp dataset..."

# -----------------------
# Download
# -----------------------
wget -O "$DATA_DIR/business.json.zip" "https://storage.googleapis.com/kaggle-data-sets/4865078/8209747/compressed/yelp_dataset/yelp_academic_dataset_business.json.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20260507%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260507T151329Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=73ca70d0db8402a51b3acb8d412e0c9b465e0da764c4644a2e16db75527bbd86d88045b6a32fb6a218d31ff31993f7ee039d54033680f51517ab545d1a39d22c13ba56be48a4a08dab5819164264b8bcd19c36fc4c12a34f10bd69d2a3583e6240b170202b8e9b3259bf51a6d37e106aefdb1a2d700c61c947840b2e1f229955236f2387f72d2bc80ee6a3195e3e0b0eed71dc070f53e791a94ee072544ab2e61054a2f9f7182d181c19aa39ce4ee6601ae40c13989963a12445f3f2f0fbbaa39204e1818a7664d9723c66034654b8684e8885f7142744cbb1fa0df94dc1dc50beaca34adf4f08ba2895133c8b409af0bf1f252800aafaf54d55ef716a9dd097"
wget -O "$DATA_DIR/review.json.zip" "https://storage.googleapis.com/kaggle-data-sets/4865078/8209747/compressed/yelp_dataset/yelp_academic_dataset_review.json.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20260506%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260506T175313Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=6793ee425dcce80a5ae660cebb71d668e28ccc43cdaec906c4b033565db4188ca8107988c87ae5ebf637866c20284e93f7f144842d29d727bbe8b561169caf16830f02f4e14129fd3c5944aff210ecc503bd140b9a7f5cb84bbb86aa67b2183e051433609f4694b819744965e3261d6a1badc83ad8bcf10e43418090dd12f6afe65deb4752363aa17a59d7d361f07fcd94ebcfe0308328827597a5430a800c537fecd86903ede58d221fef5daadc1308784a3fa7270a6b6898b68912c894e393034db866c9dd07537a75473a09fdf0e6403e71bb52a36de9600f8e9ebe438da179f93dea3dc4426e279f11cef59d63928bdc9a6e777ba839e11a2c6dd59bd025"
wget -O "$DATA_DIR/user.json.zip" "https://storage.googleapis.com/kaggle-data-sets/4865078/8209747/compressed/yelp_dataset/yelp_academic_dataset_user.json.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20260506%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260506T175340Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=55d1851492b6cf0e0f011b950a68c1dcfec5b2697c69f07455244959724e1ae64dc425bed841d17f982e08156f6b6a320229f3bb72b9ebce824892778272f44759f79d0f6078e26013d5d3735c4d42bd93e66b68eee616be396cdd2400a3b9369be4df87cbaeb6b3ad3ddcb77ee5a68e45649277de4c3ff11fd628341a747b8e099a80abcad7b6aa0aa3fa5672b3ab7afe975c7f4d0ef48d20f801649bb2bd9e81c6d874491f28b68433c54aceff99b8ad53f87f16fe15b4911e643070ef77e4ffdf5634d92534d4b242e3dc1f54435a929d800b6baaa9465e7e4af9c6ca604e6f752fa68710a47c393811c9662e8275080bd0bf1b281b4fe26778adf92a4e8d"


echo "📦 Unzipping files..."

# -----------------------
# Unzip
# -----------------------
unzip -o "$DATA_DIR/business.json.zip" -d "$DATA_DIR"
unzip -o "$DATA_DIR/review.json.zip" -d "$DATA_DIR"
unzip -o "$DATA_DIR/user.json.zip" -d "$DATA_DIR"


echo "🧹 Removing zip files..."

# -----------------------
# Cleanup zip files
# -----------------------
rm "$DATA_DIR"/*.zip


echo "🏷️ Renaming files..."

# -----------------------
# Rename to clean format
# -----------------------
mv "$DATA_DIR"/yelp_academic_dataset_business.json "$DATA_DIR/business.json"
mv "$DATA_DIR"/yelp_academic_dataset_review.json "$DATA_DIR/review.json"
mv "$DATA_DIR"/yelp_academic_dataset_user.json "$DATA_DIR/user.json"


echo "✅ Done. Dataset is ready!"