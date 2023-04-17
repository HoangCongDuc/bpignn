echo "gpu_used = $1"
echo "exp = $2"
echo "method = $3"
echo "add_noise = $4"

export CUDA_VISIBLE_DEVICES=$1

python run.py --exp=$2 --method=$3 --add_noise=$4

python plot.py --exp=$2 --method=$3
