set -eux
NETWORK_NAME=ecmwf
STACK_NAME=chatecmwf
NGINX_NAME=nginx_proxy
IP_ADDR=$(ifconfig | grep 'bridge100' | grep 'inet6' | cut -d' ' -f 2 | cut -d '%' -f 1)
source .env
echo 'Building the base image'
docker build -t $STACK_NAME:latest .
cd ./nginx
docker build -t $NGINX_NAME:latest .
cd ../
echo 'Checking if the swarm is already up'

docker-compose up
