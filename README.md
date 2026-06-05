```aiignore
requirements.txt
paddlepaddle==2.6.2
paddleocr==2.8.1
opencv-python==4.9.0.80
numpy==1.26.3
```


1. ocr_ffmpeg.py 检测字幕并且剪辑视频
2. paddleocr_word.py 检测字幕
3.CENTER_BIAS_RATIO = 0.25 的问题：

    这个参数假设字幕应该在画面中央区域
    但实际上很多视频的字幕确实在左下角、右下角、或者底部居中
    如果字幕在边缘，这个检测会漏掉
    这是一个设计取舍：
    
    好处：避免把边缘水印、台标识别成字幕
    坏处：可能漏掉确实在边缘的字幕