#!/bin/bash

dataset='Taobao' # Taobao / Userbehavior / IJCAI

echo " " >> log.txt
echo "**************************************************" >> log.txt
echo "Dataset: $dataset" >> log.txt
if test $dataset = "Userbehavior";then
  python process_user.py > preprocess.txt 2>&1
  echo 'Data ready!'
  echo 'Data ready!' >> log.txt
  python main_user.py >> log.txt
  python main_user_task.py >> log.txt
elif test $dataset = "Taobao";then
  python old_gen_0.py > preprocess1.txt 2>&1
  python old_gen_1.py > preprocess2.txt 2>&1
  python old_gen_din.py > preprocess3.txt 2>&1
  echo 'Data ready!'
  echo 'Data ready!' >> log.txt
  python main_taobao.py >> log.txt
  python main_taobao_task.py >> log.txt
elif test $dataset = "IJCAI";then
  python process_ijcai.py > preprocess.txt 2>&1
  echo 'Data ready!'
  echo 'Data ready!' >> log.txt
  python main_ijcai.py >> log.txt
  python main_ijcai_task.py >> log.txt
else
  echo 'Wrong Datset!'
fi

