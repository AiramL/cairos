pkill -f "src/federated_learning/client/torch/app.py"
pkill -f "src/federated_learning/server/torch/app.py"
pkill -f "src/federated_learning/client/keras/app.py"
pkill -f "src/federated_learning/server/keras/app.py"

rm -rf results/*
rm -rf logs/*
rm -rf models/*
