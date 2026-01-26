kill -9 $(ps aux | grep server | grep torch | awk '{printf $2"\n"}')
kill -9 $(ps aux | grep client | grep torch | awk '{printf $2"\n"}')

rm -rf results/*
rm -rf logs/*
