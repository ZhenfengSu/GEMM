#杀死所有python进程
ps -ef | grep python | grep -v grep | awk '{print $2}' | xargs kill -9
