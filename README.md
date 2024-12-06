# 视频处理工具

这是一个基于 Flask 的网页应用程序，用于处理视频文件。目前支持以下功能：

1. 从视频文件中提取音频
2. 提取视频的场景帧并生成 CSV 报告

## 功能特点

- 支持多种视频格式（MP4, AVI, MOV, MKV）
- 简单直观的用户界面
- 异步处理大文件
- 自动保存处理结果
- 支持结果文件下载

## 安装说明

1. 确保已安装 Python 3.7 或更高版本
2. 安装依赖包：
   ```bash
   pip install -r requirements.txt
   ```

## 使用方法

1. 启动应用：
   ```bash
   python app.py
   ```

2. 在浏览器中访问：
   ```
   http://localhost:5000
   ```

3. 选择要处理的视频文件并上传
4. 等待处理完成后下载结果文件

## 项目结构

```
.
├── app.py              # 主应用程序文件
├── requirements.txt    # 项目依赖
├── templates/         
│   └── index.html     # 前端模板
├── uploads/           # 上传文件临时存储
├── extracted_audio/   # 提取的音频文件
├── extracted_frames/  # 提取的场景帧
└── csv_results/      # CSV 结果文件
```

## 扩展性

该项目设计时考虑了扩展性，可以方便地添加新功能：

1. 在 `app.py` 中添加新的处理函数
2. 在 `templates/index.html` 中添加对应的 UI 组件
3. 根据需要添加新的存储目录

## 注意事项

- 上传文件大小限制为 500MB
- 确保系统有足够的存储空间
- 建议定期清理临时文件夹

## 技术栈

- 后端：Flask
- 视频处理：MoviePy, OpenCV
- 前端：Bootstrap 5
- 数据处理：Pandas, NumPy
