说明：

运行test.py就可以运行示例。

如果caffe没有加入环境变量，需要修改test.py的搜索路径。

已训练好了的CTPN模型路径http://textdet.com/downloads/ctpn_trained_model.caffemodel将下好的载模型放在./ctpn/models

已训练好了的CTC模型路径http://pan.baidu.com/s/1i5d5zdN

ctc模块采用单例模式，这样caffe模型就不会加载多次。

可修改ctpn下的cfg.py参数来适应实际中的项目。
