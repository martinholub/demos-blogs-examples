#!/bin/bash

source ~/.bashrc
tar --exclude={atari/saved_models/*,atari/server_results,logs/*,anim/*,__pycache__,summary/*} -zcvf atari_server.tar.gz atari/*
scp atari_server.tar.gz $HOST:~/rl-gym/
ssh $HOST "cd rl-gym && tar -xvzf atari_server.tar.gz && chmod u+x atari/main.py"
# Or
ssh $HOST "cd rl-gym && mkdir atari3 && tar -xvzf atari_server.tar.gz -C atari3 --strip-components 1 && chmod u+x atari3/main.py"
# test
ssh $HOST "cd rl-gym/atari && python main.py --network=DQN --mode=train --save"

# full run
ssh $HOST "cd rl-gym/atari && bsub -J 'atari' -W 7000 -n 8 -R 'rusage['mem=16384']' -oo output.txt 'python main.py -n=DQN -m=train -e=BreakoutDeterministic-v4 --save'" #-l=spaceinvader_ep7532.h5

# getting mean reward
`grep "Mean reward" output.txt  | grep -o -E '[0-9]{1,3}\.[0-9]{1,3}' | awk -F'\n' '{ sum += $1 } END { print sum / NR }'`
`grep "Mean reward" logs/atari_20180609-163841.log  | grep -o -E '[0-9]{1,3}\.[0-9]{1,3}' | tail -n200 | awk -F'\n' '{ sum += $1 } END { print sum / NR }'`
# Getting q values
`grep "Q_val: " logs/atari_20180606-225502.log | sed -E 's/.*, Q_val: (.*)$/\1/g' > q_vals.txt`
# Getting stats on actions:
`grep "Action: " logs/atari_20180609-135902.log | sed -E 's/.*: Action: ([0-6]), .*$/\1/g'| sort | uniq -c | awk '{print $1}' | paste -sd+ - | bc`

# Playing with jobs at euler:
`bqueues`
`bbjobs`
`bmod -W <time> <jobid>` # change allowed max time for job
`bswitch <queue-name> <jobid>` # move job to different queue
