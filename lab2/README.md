<i>函数说明见report</i>

### 实验环境

见env文件夹下的`requirement.txt`

### 运行方式

#### bert

- 从rec网下载文件夹bert-base-uncased和model文件夹放在bert根目录下

  > 链接：https://rec.ustc.edu.cn/share/db5aba00-4e9c-11eb-8d02-fdd0d0399d0c密码：vyf2

- 如果需要训练

  则运行`train.py`

- 如果想直接测试

  运行`test.py`

- 注意可能会出现路径问题(因为我直接从服务器拉的代码)， 如果出现路径问题，只需要更改为相应路径即可(应该只有bert-bas-uncased可能有问题)

#### rnn

- 环境配置好后直接运行`python rnn_att.py`即可得到`predict.txt`
- 运行`python data_process.py` 得到最终符合提交标准的答案

- 由于时间关系，并没有单独写`test`部分，但训练模型我上传到rec的`model`文件夹下，如若读者有兴趣可以自己试一下
- rnn的data路径可能会出问题，同bert，只需要稍作修改即可

### 文件说明

- report.pdf是实验报告
- bert文件夹存放bert模型的代码
  - 其中src为代码
  - data为训练所以需要数据
- rnn文件夹存放rnn模型的代码
  - 其中src为代码
  - data为训练所以需要数据