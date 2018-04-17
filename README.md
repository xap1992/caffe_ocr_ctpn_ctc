说明：

感谢senlinuc贡献的https://github.com/senlinuc/caffe_ocr

感谢kasyoukin贡献的https://github.com/kasyoukin/caffe_ocr_for_linux

感谢tianzhi0549贡献的https://github.com/tianzhi0549/CTPN

我这里的是linux版本，将上述三个项目与官方项目整合在一起。

测试环境 ubuntu 16.04 、cuda 8.0、cudnn 6.0。按官方信息提示理论上支持cuda9.0，cudnn7.0


没有使用caffe_ocr中的多标签输入源码，采用方法重载方式支持多标签输入。