# Ray image to use for the server
RAY_IMAGE='rayproject/ray:3d764d-py38'

# Create temp directory to store temporary files
mkdir .temp
cd .temp
git clone https://github.com/ray-project/ray
cd ray/deploy/charts

# Create 
helm -n ray install example-cluster --create-namespace ./ray --set image=$RAY_IMAGE


cd ..
rm -rf .temp