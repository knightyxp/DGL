ps -ef | grep main.py | grep -v grep | awk '{print $2}' | xargs kill -9
ps -ef | grep msrvtt.sh | grep -v grep | awk '{print $2}' | xargs kill -9