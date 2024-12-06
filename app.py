import os
import uuid
import json
import time
import logging
import threading
import traceback
from datetime import datetime
from flask import Flask, request, render_template, send_file, jsonify
from werkzeug.utils import secure_filename
from moviepy.editor import VideoFileClip
import cv2
import numpy as np
import pandas as pd
import yt_dlp
from PIL import Image
from dotenv import load_dotenv
import base64
import zhipuai
import concurrent.futures
import socket
import urllib3
import ssl
import requests
from aip import AipSpeech
from pydub import AudioSegment
from paddleocr import PaddleOCR
from functools import lru_cache
from difflib import SequenceMatcher
import subprocess
import platform

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB
app.debug = True

# 百度语音识别API配置
BAIDU_APP_ID = "116456613"
BAIDU_API_KEY = "HrJvfV3kgzpNFWFBC5VyYOiX"
BAIDU_SECRET_KEY = "4g5OBobzwyZjRBnm1WKtbmfC91hPb9jb"

# 创建百度语音识别客户端
speech_client = AipSpeech(BAIDU_APP_ID, BAIDU_API_KEY, BAIDU_SECRET_KEY)

# 配置日志
logging.basicConfig(level=logging.INFO,
                   format='[%(asctime)s] [%(levelname)7s] %(filename)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

# 设置全局超时和连接配置
socket.setdefaulttimeout(30)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 创建自定义SSL上下文
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# 清除代理设置
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)

# 加载环境变量
load_dotenv()

# 配置日志记录
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 配置智谱AI
zhipuai.api_key = os.getenv('ZHIPUAI_API_KEY')

# 配置上传文件的存储路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
AUDIO_FOLDER = os.path.join(BASE_DIR, 'extracted_audio')
FRAMES_FOLDER = os.path.join(BASE_DIR, 'extracted_frames')
CSV_FOLDER = os.path.join(BASE_DIR, 'csv_results')
YOUTUBE_FOLDER = os.path.join(BASE_DIR, 'youtube_downloads')
TEXT_FOLDER = os.path.join(BASE_DIR, 'text_results')
COOKIES_FILE = os.path.join(os.path.dirname(__file__), 'cookies', 'youtube.cookies')

# 存储处理进度的字典
processing_status = {}

# 初始化 PaddleOCR
ocr = PaddleOCR(
    use_angle_cls=True,
    lang='ch',
    use_gpu=False,  # AMD GPU 暂时不支持，保持 False
    enable_mkldnn=True,  # 启用 MKL-DNN 加速
    cpu_threads=8,  # 根据你的 CPU 核心数调整
    det_db_thresh=0.3,  # 检测阈值
    det_db_box_thresh=0.5,  # 文本框阈值
    rec_batch_num=6,  # 识别批处理大小
    cls_batch_num=6,  # 分类批处理大小
    max_batch_size=10,  # 最大批处理大小
    det_limit_side_len=960,  # 限制图像尺寸
)
logger.info("PaddleOCR初始化完成")

# 确保所需文件夹存在
for folder in [UPLOAD_FOLDER, AUDIO_FOLDER, FRAMES_FOLDER, CSV_FOLDER, YOUTUBE_FOLDER, TEXT_FOLDER]:
    os.makedirs(folder, exist_ok=True)
    logger.info(f"确保目录存在: {folder}")

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 配置线程池
executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {
        'mp4', 'avi', 'mov', 'mkv',  # 视频格式
        'mp3', 'wav', 'm4a', 'flac'  # 音频格式
    }

def allowed_image_file(filename):
    """检查是否为允许的图片文件类型"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

def analyze_image(image_path):
    """分析图片内容并返回详细描述"""
    try:
        # 确保图片文件存在
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片文件不存在: {image_path}")
        
        # 读取图片并进行基本检查
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        # 检查图片是否全白
        if np.mean(img) > 250 and np.std(img) < 10:
            return "图片似乎是全白的，可能是帧提取过程中出现了问题。请检查视频源和帧提取设置。"
        
        # 将图片转换为base64
        _, buffer = cv2.imencode('.jpg', img)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # 创建API客户端
        client = zhipuai.ZhipuAI(api_key=os.getenv('ZHIPUAI_API_KEY'))
        
        # 构建详细的分析提示词
        prompt = """请仔细分析这张图片，并按以下五个维度提供详细描述：

1. 主要对象识别：
   - 画面中的人物特征（性别、年龄、衣着等）
   - 动物种类和特征
   - 主要物品和物体

2. 场景环境描述：
   - 场景类型（室内/室外、具体位置）
   - 空间布局和环境特征
   - 光线、色彩和氛围

3. 动��状态分析：
   - 人物/动物的具体动作
   - 动作的连贯性和目的性
   - 互动和关系

4. 情感氛围评估：
   - 人物表情和情绪状态
   - 场景营造的情感基调
   - 画面传达的整体感受

5. 特殊细节说明：
   - 独特或显著的视觉元素
   - 重要的背景细节
   - 特殊的拍摄角度或效果

请确保对每个维度都进行详细分析，如果某些维度在图片中未能体现，请明确说明。"""
        
        # 调用智谱AI API
        response = client.chat.completions.create(
            model="glm-4v",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ]
        )
        
        # 提取分析结果
        analysis = response.choices[0].message.content
        logging.info(f"图片分成功: {image_path}")
        return analysis
        
    except Exception as e:
        error_msg = f"图片分析失败: {str(e)}"
        logging.error(error_msg)
        return error_msg

def extract_audio(video_path, output_path):
    """从视频中提取音频"""
    try:
        video = VideoFileClip(video_path)
        audio = video.audio
        if audio is None:
            return False, "视频文件没有音频轨道"
        audio.write_audiofile(output_path)
        video.close()
        return True, output_path
    except Exception as e:
        return False, str(e)

def detect_scene_changes(video_path, output_csv, task_id):
    """检测场景变化并保存帧"""
    try:
        logger.info(f'开始处理视频: {video_path}')
        processing_status[task_id] = {
            'status': 'processing',
            'progress': 0,
            'message': '开始处理视频...'
        }

        # 检查文件是否存在
        if not os.path.exists(video_path):
            error_msg = f'视频文件不存在: {video_path}'
            logger.error(error_msg)
            processing_status[task_id].update({
                'status': 'error',
                'message': error_msg
            })
            return False, error_msg

        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            error_msg = f'无法打开视频文件: {video_path}'
            logger.error(error_msg)
            processing_status[task_id].update({
                'status': 'error',
                'message': error_msg
            })
            return False, error_msg

        # 获取视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps
        logger.info(f'视频信息 - 总帧数: {total_frames}, FPS: {fps}, 时长: {duration:.2f}秒')
        
        processing_status[task_id]['message'] = f'视频信息 - 总帧数: {total_frames}, FPS: {fps}, 时长: {duration:.2f}秒'

        # 初始化变量
        frames_info = []
        prev_frame = None
        frame_count = 0
        extracted_count = 0
        min_scene_duration = 0.5  # 最小场景持续时间（秒）
        last_scene_time = 0
        
        # 差异计算窗口
        diff_scores_window = []
        window_size = 30
        high_diff_threshold = 0.6
        
        # 创建线程池用于并行处理图片分析
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_frame = {}
            
            # 处理每一帧
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = frame_count / fps
                
                # 更新进度（每50帧）
                if frame_count % 50 == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(f'处理进度: {progress:.1f}% ({frame_count}/{total_frames})')
                    processing_status[task_id].update({
                        'progress': progress,
                        'message': f'处理进度: {progress:.1f}% ({frame_count}/{total_frames})'
                    })

                try:
                    # 确保最小场景持续时间
                    if timestamp - last_scene_time < min_scene_duration:
                        frame_count += 1
                        continue

                    # 初始化第一帧
                    if prev_frame is None:
                        if frame is not None:
                            prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            frame_count += 1
                        continue

                    # 处理当前帧
                    if frame is not None:
                        curr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        
                        # 应用高斯模糊
                        if curr_frame is not None and prev_frame is not None:
                            curr_frame = cv2.GaussianBlur(curr_frame, (5, 5), 0)
                            prev_frame = cv2.GaussianBlur(prev_frame, (5, 5), 0)
                            
                            # 计算帧差异
                            frame_diff = cv2.absdiff(curr_frame, prev_frame)
                            diff_score = np.mean(frame_diff)
                            diff_scores_window.append(diff_score)
                            
                            # 维护窗口大小
                            if len(diff_scores_window) > window_size:
                                diff_scores_window.pop(0)
                            
                            # 检测关键帧
                            if len(diff_scores_window) == window_size:
                                mean_diff = np.mean(diff_scores_window)
                                std_diff = np.std(diff_scores_window)
                                threshold = mean_diff + std_diff * 1.5
                                
                                # 判断是否为关键帧
                                if diff_score > threshold and diff_score > (max(diff_scores_window) * high_diff_threshold):
                                    frame_filename = f'frame_{frame_count}.jpg'
                                    frame_path = os.path.join(FRAMES_FOLDER, frame_filename)
                                    
                                    # 保存帧
                                    cv2.imwrite(frame_path, frame)
                                    
                                    # 异步分析图像
                                    future = executor.submit(analyze_image, frame_path)
                                    future_to_frame[future] = (frame_count, timestamp, frame_path)
                                    
                                    extracted_count += 1
                                    last_scene_time = timestamp
                                    
                                    processing_status[task_id]['message'] = f'提取关键帧 {extracted_count}: 帧号 {frame_count}, 时间戳 {timestamp:.2f}秒'
                                    logger.info(f'提取关键帧 {extracted_count}: 帧号 {frame_count}, 时间戳 {timestamp:.2f}秒')

                        prev_frame = curr_frame
                    frame_count += 1

                except Exception as e:
                    logger.error(f'处理帧 {frame_count} 时出错: {str(e)}')
                    continue

            # 等待所有图片分析完成
            for future in concurrent.futures.as_completed(future_to_frame):
                frame_count, timestamp, frame_path = future_to_frame[future]
                try:
                    frame_analysis = future.result()
                    frames_info.append({
                        'frame_number': frame_count,
                        'timestamp': timestamp,
                        'frame_path': os.path.basename(frame_path),
                        'description': frame_analysis
                    })
                except Exception as e:
                    logger.error(f"图片分析时出错: {str(e)}")

        # 保存结果到CSV
        if frames_info:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            csv_filename = f'scene_analysis_{timestamp}.csv'
            csv_path = os.path.join(CSV_FOLDER, csv_filename)
            
            df = pd.DataFrame(frames_info)
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            
            logger.info('视频处理完成:')
            logger.info(f'结果已保存到: {csv_path}')
            
            processing_status[task_id].update({
                'status': 'completed',
                'progress': 100,
                'message': '处理完成',
                'csv_path': csv_filename
            })
            
            return True, csv_filename
        else:
            logger.warning('未检测到关键帧')
            processing_status[task_id].update({
                'status': 'completed',
                'progress': 100,
                'message': '处理完成，但未检测到关键帧'
            })
            return False, "未检测到关键帧"

    except Exception as e:
        error_msg = f'处理视频时发生错误: {str(e)}'
        logger.error(error_msg)
        processing_status[task_id].update({
            'status': 'error',
            'message': error_msg
        })
        return False, error_msg
    finally:
        if 'cap' in locals() and cap is not None:
            cap.release()

def process_frame_ocr(frame):
    """处理单帧图像的OCR识别
    Args:
        frame: OpenCV格式的图像帧
    Returns:
        text: 识别到的文本，如果没有识别到则返回空字符串
    """
    try:
        # OpenCV的BGR格式转为RGB格式
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # OCR识别
        result = ocr.ocr(frame_rgb)
        
        if not result or not result[0]:
            return ""
            
        # 提所有识别到文本
        texts = []
        for line in result[0]:
            if line[1][0]:  # 如果有识别结果
                texts.append(line[1][0])
        
        return " ".join(texts)
    except Exception as e:
        logging.error(f"OCR处理失败: {str(e)}")
        return ""

@lru_cache(maxsize=1000)
def recognize_text(frame_bytes):
    """缓存OCR识别结果"""
    # 将frame转换为bytes用于缓存
    frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
    result = ocr.ocr(frame, cls=True)
    return result

def similar(a, b):
    """计算两个字符串的相似度"""
    return SequenceMatcher(None, a, b).ratio()

def extract_subtitles(video_path, task_id):
    try:
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise Exception("无法打开视频文件")

        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 设置采样间隔 (每秒1帧)
        frame_interval = int(fps)  # 增加采样间隔
        if frame_interval < 1:
            frame_interval = 1

        subtitles = []
        frame_count = 0
        processed_frames = 0
        all_text = []
        
        # 用于存储最近识别的文本，避免重复
        recent_texts = []
        min_time_gap = 2.0  # 相同文本的最小时间间隔(秒)

        while True:
            ret, frame = video.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                processed_frames += 1
                progress = (processed_frames * frame_interval / total_frames) * 100
                update_task_status(task_id, 'processing', progress)

                frame = preprocess_frame(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                result = ocr.ocr(frame, cls=True)
                
                current_time = frame_count / fps
                
                if result and len(result) > 0:
                    for line in result[0]:
                        if line[1][1] > 0.6:  # 提高置信度阈值
                            text = line[1][0].strip()
                            
                            # 忽略太短的文本
                            if len(text) < 2:
                                continue
                                
                            # 检查是否与最近的文本重复或相似
                            is_duplicate = False
                            for recent in recent_texts:
                                if similar(text, recent['text']) > 0.8:  # 相似度阈值
                                    if current_time - recent['time'] < min_time_gap:
                                        is_duplicate = True
                                        break
                            
                            if not is_duplicate:
                                subtitles.append({
                                    'timestamp': current_time,
                                    'text': text
                                })
                                all_text.append(text)
                                
                                # 更新最近文本列表
                                recent_texts.append({
                                    'text': text,
                                    'time': current_time
                                })
                                # 只保留最近的几条记录
                                if len(recent_texts) > 5:
                                    recent_texts.pop(0)

            frame_count += 1

        video.release()

        if subtitles:
            # 合并相邻的相似字幕
            merged_subtitles = []
            i = 0
            while i < len(subtitles):
                current = subtitles[i]
                j = i + 1
                while j < len(subtitles):
                    if similar(current['text'], subtitles[j]['text']) > 0.8:
                        # 如果时间间隔小于阈值，跳过后面的重复文本
                        if subtitles[j]['timestamp'] - current['timestamp'] < min_time_gap:
                            j += 1
                            continue
                    break
                merged_subtitles.append(current)
                i = j

            srt_path = generate_srt(merged_subtitles, video_path)
            # 将所有文本合并为一个字符串，去除重复
            unique_texts = []
            for text in all_text:
                if not any(similar(text, ut) > 0.8 for ut in unique_texts):
                    unique_texts.append(text)
            
            full_text = '\n'.join(unique_texts)
            update_task_status(task_id, 'completed', 
                             output_file=os.path.basename(srt_path),
                             message=full_text)
        else:
            update_task_status(task_id, 'completed', message='未检测到字幕')

    except Exception as e:
        update_task_status(task_id, 'error', message=str(e))
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/extract_audio', methods=['POST'])
def handle_extract_audio():
    try:
        if 'video' not in request.files:
            logger.error('没有上传文件')
            return jsonify({'error': '没有上传文件'}), 400
        
        file = request.files['video']
        if file.filename == '':
            logger.error('没有选择文件')
            return jsonify({'error': '没有选择文件'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}_{filename}")
            
            logger.info(f'保存视频文件到: {video_path}')
            file.save(video_path)
            
            audio_filename = f"{timestamp}_audio_{os.path.splitext(filename)[0]}.mp3"
            audio_path = os.path.join(AUDIO_FOLDER, audio_filename)
            
            logger.info(f'开始提取音频，输出路径: {audio_path}')
            success, result = extract_audio(video_path, audio_path)
            
            if success:
                logger.info('音频提取成功')
                return jsonify({
                    'success': True,
                    'message': '音频提取成功',
                    'audio_path': audio_filename
                })
            else:
                logger.error(f'音频提取失败: {result}')
                return jsonify({'error': f'音频提取失败: {result}'}), 500
        
        logger.error(f'不支持的文件格式: {file.filename}')
        return jsonify({'error': '不支持的文件格式'}), 400
    
    except Exception as e:
        error_msg = f'处理视频时发生错误: {str(e)}\n{traceback.format_exc()}'
        logger.error(error_msg)
        return jsonify({'error': error_msg}), 500

@app.route('/extract_frames', methods=['POST'])
def handle_extract_frames():
    """处理场景帧提取请求"""
    try:
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': '没有上传文件'})
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'success': False, 'error': '没有选择文件'})
            
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': '不支持的文件格式'})
        
        # 生成任务ID
        task_id = f"task_{int(time.time())}"
        
        # 保存上传的文件
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)
        
        # 准备输出文件路径
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_csv = os.path.join(CSV_FOLDER, f'scene_analysis_{timestamp}.csv')
        
        # 启动异步处理
        thread = threading.Thread(target=lambda: detect_scene_changes(video_path, output_csv, task_id))
        thread.start()
        
        return jsonify({
            'success': True,
            'task_id': task_id,
            'message': '开始处理视频'
        })
        
    except Exception as e:
        error_msg = f'处理请求时出错: {str(e)}'
        logger.error(error_msg)
        return jsonify({'success': False, 'error': error_msg})

@app.route('/api/task_status/<task_id>')
def get_task_status(task_id):
    """获取任务状态"""
    try:
        status = processing_status.get(task_id, {})
        if not status:
            return jsonify({'status': 'unknown', 'error': '任务不存在'})
            
        # 添加调试日志
        logger.debug(f"Task status for {task_id}: {status}")
        return jsonify(status)
        
    except Exception as e:
        error_msg = f"获取任务状态失败: {str(e)}"
        logger.error(error_msg)
        return jsonify({'status': 'error', 'error': error_msg})

@app.route('/get_video_info', methods=['POST'])
def get_video_info():
    """获取视频信息"""
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({'error': '缺少URL参数'}), 400
            
        url = data['url']
        logger.info(f"获取频信息，URL: {url}")
        
        # 配置yt-dlp选项
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,  # 获取完整信息
            'format': None,  # 不限制格式
            'youtube_include_dash_manifest': True,  # 包含DASH格式
            'nocheckcertificate': True,
            'socket_timeout': 30,
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-us,en;q=0.5',
                'Sec-Fetch-Mode': 'navigate'
            },
            'ignoreerrors': True,
            'no_check_certificate': True,
            'retries': 5,
            'fragment_retries': 5,
            'skip_download': True
        }

        if os.path.exists(COOKIES_FILE):
            ydl_opts['cookiefile'] = COOKIES_FILE
            logger.info(f"使用cookies文件: {COOKIES_FILE}")
        else:
            logger.warning(f"Cookies文件不存在: {COOKIES_FILE}")

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                logger.info(f"开始获取视频信息: {url}")
                info = ydl.extract_info(url, download=False)
                if not info:
                    logger.error("无法获取视频信息")
                    return jsonify({'error': '无法获取视频信息'}), 400

                logger.info("成功获取视频信息")
                
                # 获取可用的格式
                formats = []
                if 'formats' in info:
                    seen_qualities = set()
                    for f in info['formats']:
                        # 记录格式信息用于调试
                        logger.debug(f"格式ID: {f.get('format_id')}, "
                                   f"视频编码: {f.get('vcodec')}, "
                                   f"音频编码: {f.get('acodec')}, "
                                   f"文件扩展名: {f.get('ext')}, "
                                   f"分辨率: {f.get('width')}x{f.get('height')}")
                        
                        # 跳过仅音频格式
                        if f.get('vcodec') == 'none':
                            continue
                            
                        height = f.get('height', 0)
                        if not height:
                            continue
                            
                        quality = f"{height}p"
                        if quality in seen_qualities:
                            continue
                            
                        seen_qualities.add(quality)
                        formats.append({
                            'format_id': f['format_id'],
                            'ext': f.get('ext', ''),
                            'height': height,
                            'quality': quality,
                            'filesize': f.get('filesize', 0),
                            'vcodec': f.get('vcodec', ''),
                            'acodec': f.get('acodec', '')
                        })
                
                # 按分辨率排序
                formats.sort(key=lambda x: x['height'], reverse=True)
                
                # 构建响应数据
                response_data = {
                    'title': info.get('title', '未知标题'),
                    'thumbnail': info.get('thumbnail', ''),
                    'duration': info.get('duration', 0),
                    'uploader': info.get('uploader', '未知作者'),
                    'formats': formats
                }
                
                logger.info(f"返回视频信息: {json.dumps(response_data, ensure_ascii=False)}")
                return jsonify(response_data)
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"获取视频信息失败: {error_msg}")
            return jsonify({'error': error_msg}), 500

    except Exception as e:
        error_msg = str(e)
        logger.error(f"获取视频信息失败: {error_msg}")
        return jsonify({'error': error_msg}), 500

@app.route('/download_youtube', methods=['POST'])
def download_youtube_route():
    """理YouTube视频下载请求"""
    try:
        url = request.json.get('url')
        requested_format = request.json.get('format')
        
        if not url:
            return jsonify({'error': '缺少URL参数'}), 400
            
        # 生成任务ID并初始化状态
        task_id = f"download_{int(time.time())}"
        processing_status[task_id] = {
            'status': 'starting',
            'progress': 0,
            'url': url,
            'format': requested_format
        }
        
        logger.info(f"开始下载任务 {task_id}: {url}")
        
        # 配置下载选项
        ydl_opts = {
            'format': requested_format if requested_format else 'bestvideo[height<=2160]+bestaudio/best',
            'merge_output_format': 'mp4',
            'outtmpl': os.path.join(YOUTUBE_FOLDER, '%(title)s_%(id)s.%(ext)s'),
            'quiet': False,
            'no_warnings': False,
            'nocheckcertificate': True,
            'socket_timeout': 60,
            'retries': 10,
            'fragment_retries': 10,
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-us,en;q=0.5',
                'Sec-Fetch-Mode': 'navigate'
            },
            'extractor_args': {
                'youtube': {
                    'player_client': ['android'],
                    'player_skip': ['webpage', 'configs']
                }
            },
            'progress_hooks': [lambda d: update_progress(task_id, d)],
            'postprocessor_hooks': [lambda d: update_progress(task_id, d)],
            'prefer_ffmpeg': True,
            'keepvideo': False
        }

        try:
            # 在后台线程中启动下载
            executor.submit(download_video, url, ydl_opts, task_id)
            return jsonify({
                'task_id': task_id,
                'status': 'started',
                'message': '下载已开始'
            })

        except Exception as e:
            error_msg = f"启动下载失败: {str(e)}"
            logger.error(error_msg)
            processing_status[task_id] = {
                'status': 'error',
                'error': error_msg
            }
            return jsonify({'error': error_msg}), 500
        
    except Exception as e:
        error_msg = f'处理下载请求时出错: {str(e)}'
        logger.error(error_msg)
        return jsonify({'error': error_msg}), 500

@app.route('/youtube_download/<filename>')
def serve_youtube_file(filename):
    """下载已完成的YouTube视频"""
    try:
        return send_from_directory(YOUTUBE_FOLDER, filename, as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 404

def download_video(url, ydl_opts, task_id):
    """在后台线程中下载视频"""
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.info(f"开始下载视频: {url}")
            ydl.download([url])
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"下载失败: {error_msg}")
        processing_status[task_id].update({
            'status': 'error',
            'error': error_msg
        })

def process_audio_chunk(audio_chunk, speech_client, chunk_index):
    """处理单个音频片段"""
    logger.info(f"开始理音频片段 {chunk_index+1}, 长度: {len(audio_chunk)}ms")
    
    # 保存为临时WAV文件
    temp_wav = os.path.join(UPLOAD_FOLDER, f'temp_chunk_{uuid.uuid4()}.wav')
    audio_chunk.export(temp_wav, format='wav')
    
    try:
        # 读取音频数据
        with open(temp_wav, 'rb') as f:
            audio_data = f.read()
        
        # 确保音频数据不超过10MB
        file_size_mb = len(audio_data) / (1024 * 1024)
        logger.info(f"音频片段 {chunk_index+1} 大小: {file_size_mb:.2f}MB")
        
        if len(audio_data) > 10 * 1024 * 1024:
            logger.error(f"音频片段 {chunk_index+1} 超过10MB限制")
            return None, "音频数据超过10MB限制"
        
        # 调用百度语识别API
        result = speech_client.asr(audio_data, 'pcm', 16000, {
            'dev_pid': 1537,  # 普通话(支持简单的英文识别)
        })
        
        if result['err_no'] == 0:
            recognized_text = ' '.join(result['result'])
            logger.info(f"音频片段 {chunk_index+1} 识别成功: {recognized_text[:50]}...")
            return result['result'], None
        else:
            error_codes = {
                3300: "输入参数不正确",
                3301: "音频质量过差",
                3302: "鉴权失败",
                3303: "语音服务器后端问题",
                3304: "用户的请求QPS超限",
                3305: "用户的日pv（日请求量）超限",
                3307: "语音服务器后端识别出错",
                3308: "音频过长",
                3309: "音频数据问题",
                3310: "输入的音频文件过大",
                3311: "采样率rate参数不在选项里",
                3312: "音频格式format参数不在选项里",
            }
            error_msg = error_codes.get(result['err_no'], result['err_msg'])
            logger.error(f"音频片段 {chunk_index+1} 识别失败: {error_msg}")
            return None, error_msg
    finally:
        # 清理临时文件
        try:
            os.remove(temp_wav)
        except:
            pass

@app.route('/audio2text', methods=['POST'])
def audio_to_text():
    if 'file' not in request.files:
        return jsonify({'error': '没有文件上传'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    if not file.filename.lower().endswith(('.mp3', '.wav', '.m4a', '.flac')):
        return jsonify({'error': '不支持的文件格式'}), 400
    
    try:
        # 保上传的文件
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        logger.info(f"开始处理音频文件: {filename}")
        
        # 使用pydub加载音频
        audio = AudioSegment.from_file(filepath)
        
        # 转换为16kHz采样率的单声道音频
        audio = audio.set_frame_rate(16000).set_channels(1)
        
        # 计算音频时长（毫秒）
        duration_ms = len(audio)
        logger.info(f"音频总长度: {duration_ms/1000:.2f}秒")
        
        # 将音频分割成30秒的片段（百度API建议不超过60秒）
        chunk_size_ms = 30 * 1000  # 30秒
        chunks = []
        
        # 计算需要分割的片段数
        num_chunks = (duration_ms + chunk_size_ms - 1) // chunk_size_ms
        logger.info(f"将音频分割成 {num_chunks} 个片段")
        
        # 添加1秒的重叠，以避免在分割点丢失语
        overlap_ms = 1000  # 1秒重叠
        
        for i in range(0, duration_ms, chunk_size_ms):
            start_ms = max(0, i - overlap_ms if i > 0 else i)
            end_ms = min(i + chunk_size_ms + overlap_ms, duration_ms)
            chunk = audio[start_ms:end_ms]
            chunks.append(chunk)
        
        # 处理每个音频片段
        all_results = []
        errors = []
        
        for i, chunk in enumerate(chunks):
            result, error = process_audio_chunk(chunk, speech_client, i)
            if result:
                # 确保每个片段的结果都是一个列表
                if isinstance(result, list):
                    all_results.extend(result)
                else:
                    all_results.append(result)
            if error:
                errors.append(f"片段{i+1}处理失败: {error}")
                # 如果是音频过长错误，尝试将该片段再分成两半处理
                if "音频过长" in error:
                    logger.info(f"尝试将片段 {i+1} 再次分割")
                    half_duration = len(chunk) // 2
                    first_half = chunk[:half_duration]
                    second_half = chunk[half_duration:]
                    
                    # 处理第一半
                    result1, error1 = process_audio_chunk(first_half, speech_client, f"{i+1}.1")
                    if result1:
                        if isinstance(result1, list):
                            all_results.extend(result1)
                        else:
                            all_results.append(result1)
                    
                    # 处理第二半
                    result2, error2 = process_audio_chunk(second_half, speech_client, f"{i+1}.2")
                    if result2:
                        if isinstance(result2, list):
                            all_results.extend(result2)
                        else:
                            all_results.append(result2)
        
        # 如果有任何结果
        if all_results:
            # 合并所有识别结果
            text = ' '.join(all_results)
            logger.info(f"所有片段处理完成，总共识别出 {len(text)} 个字符")
            
            # 保存识别结果到文本文件
            text_filename = os.path.splitext(filename)[0] + '.txt'
            text_filepath = os.path.join(TEXT_FOLDER, text_filename)
            with open(text_filepath, 'w', encoding='utf-8') as f:
                f.write(text)
            
            response = {
                'success': True,
                'text': text,
                'text_file': text_filename,
                'total_chunks': len(chunks),
                'successful_chunks': len([r for r in all_results if r]),
                'duration_seconds': duration_ms/1000
            }
            
            # 如果有错误，添加到响应中
            if errors:
                response['warnings'] = errors
                logger.warning(f"处理过程中出现以下警告: {'; '.join(errors)}")
            
            return jsonify(response)
        else:
            # 如果没有成功的结果
            error_msg = '音频识别失败: ' + '; '.join(errors)
            logger.error(error_msg)
            return jsonify({'error': error_msg}), 400
            
    except Exception as e:
        logger.error(f"音频识别错误: {str(e)}")
        return jsonify({'error': f'处理失败: {str(e)}'}), 500
    finally:
        # 清理原始上传文件
        try:
            os.remove(filepath)
        except:
            pass

@app.route('/download_text/<filename>')
def download_text(filename):
    return send_file(
        os.path.join(TEXT_FOLDER, filename),
        as_attachment=True
    )

@app.route('/api/extract_scenes', methods=['POST'])
def extract_scenes_route():
    try:
        if 'video' not in request.files:
            return jsonify({'error': '没有上传视频文件'}), 400
            
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
            
        # 生成唯一的任务ID
        task_id = str(uuid.uuid4())
        
        # 生成安全的文件名并保存视频
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}_{filename}")
        
        file.save(video_path)
        
        # 初始化任务状态
        processing_status[task_id] = {
            'status': 'processing',
            'progress': 0,
            'message': '正在处理...',
            'timestamp': timestamp  # 保存时间戳供后续使用
        }
        
        def process_scenes():
            try:
                # 获取保存的时间戳
                timestamp = processing_status[task_id]['timestamp']
                
                # 提取场景
                video = cv2.VideoCapture(video_path)
                if not video.isOpened():
                    raise Exception("无法打开视频文件")

                # 创建输出目录
                frames_folder = f"{timestamp}_frames"
                frames_path = os.path.join(FRAMES_FOLDER, frames_folder)
                os.makedirs(frames_path, exist_ok=True)

                # 场景检测参数
                total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = video.get(cv2.CAP_PROP_FPS)
                frame_interval = int(fps)  # 每秒取一帧
                threshold = 30.0  # 场景变化阈值

                prev_frame = None
                frame_count = 0
                scene_count = 0
                scenes = []

                while True:
                    ret, frame = video.read()
                    if not ret:
                        break

                    if frame_count % frame_interval == 0:
                        # 更新进度
                        progress = (frame_count / total_frames) * 100
                        processing_status[task_id].update({
                            'status': 'processing',
                            'progress': progress
                        })

                        # 检测场景变化
                        if prev_frame is not None:
                            # 计算帧差异
                            diff = cv2.absdiff(prev_frame, frame)
                            diff_mean = np.mean(diff)

                            if diff_mean > threshold:
                                # 保存关键帧
                                frame_filename = f"scene_{scene_count:04d}.jpg"
                                frame_path = os.path.join(frames_path, frame_filename)
                                cv2.imwrite(frame_path, frame)
                                
                                # 记录场景信息
                                scene_timestamp = frame_count / fps
                                scenes.append({
                                    'scene_id': scene_count,
                                    'timestamp': scene_timestamp,
                                    'frame_number': frame_count,
                                    'filename': frame_filename
                                })
                                scene_count += 1

                        prev_frame = frame.copy()
                    frame_count += 1

                video.release()

                # 生成CSV报告
                csv_filename = f"{timestamp}_scenes.csv"
                csv_path = os.path.join(CSV_FOLDER, csv_filename)
                df = pd.DataFrame(scenes)
                df.to_csv(csv_path, index=False)

                # 更新任务状态为完成
                processing_status[task_id].update({
                    'status': 'completed',  # 确保状态更新为completed
                    'progress': 100,
                    'frames_folder': frames_folder,
                    'csv_file': csv_filename,
                    'message': '场景提取完成'  # 添加完成消息
                })

            except Exception as e:
                logger.error(f"场景提取失败: {str(e)}")
                logger.error(traceback.format_exc())
                processing_status[task_id].update({
                    'status': 'error',
                    'message': f'处理失败: {str(e)}'
                })
            finally:
                # 清理临时文件
                if os.path.exists(video_path):
                    os.remove(video_path)

        # 启动后台处理
        executor.submit(process_scenes)
        
        return jsonify({
            'task_id': task_id,
            'message': '任务已提交，正在处理'
        })
        
    except Exception as e:
        return jsonify({
            'error': f'请求处理失败: {str(e)}'
        }), 500

@app.route('/download/subtitle/<filename>')
def download_subtitle(filename):
    """下载生成的字幕文件"""
    try:
        return send_file(
            os.path.join(TEXT_FOLDER, filename),
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({'error': f'文件下载失败: {str(e)}'}), 500

def update_progress(task_id, d):
    """更新下载进度"""
    try:
        if task_id not in processing_status:
            logger.warning(f"Task {task_id} not found in processing_status")
            return
            
        current_time = time.time()
        current_status = processing_status[task_id]
        
        # 添加最后更新时间
        if 'last_update' not in current_status:
            current_status['last_update'] = current_time
            
        # 检查是否过30秒没有实际进度更新
        if current_time - current_status['last_update'] > 30:
            if current_status.get('status') == 'downloading' and current_status.get('progress', 0) > 0:
                logger.warning(f"Download progress stalled for task {task_id}")
                processing_status[task_id].update({
                    'status': 'error',
                    'error': '下载停滞，请重试'
                })
                return
                
        if d['status'] == 'downloading':
            downloaded = d.get('downloaded_bytes', 0)
            total = d.get('total_bytes') or d.get('total_bytes_estimate', 0)
            
            if downloaded and total:
                progress = (float(downloaded) / float(total)) * 100
                # 只有当进度有实际变化时才更新
                if abs(progress - current_status.get('progress', 0)) > 0.1:
                    processing_status[task_id].update({
                        'status': 'downloading',
                        'progress': round(progress, 2),
                        'speed': d.get('speed', 0),
                        'eta': d.get('eta', 0),
                        'filename': d.get('filename', ''),
                        'downloaded': downloaded,
                        'total': total,
                        'last_update': current_time
                    })
                    
                    logger.info(
                        f"下载进度更新 - Task: {task_id}, "
                        f"Progress: {progress:.1f}%, "
                        f"Speed: {d.get('speed', 0)/1024/1024:.2f}MB/s, "
                        f"ETA: {d.get('eta', 0)}s"
                    )
            
        elif d['status'] == 'finished':
            processing_status[task_id].update({
                'status': 'finished',
                'progress': 100,
                'filename': d.get('filename', ''),
                'last_update': current_time
            })
            logger.info(f"下载完成 - Task: {task_id}, File: {d.get('filename', '')}")
            
        elif d['status'] == 'error':
            error_msg = str(d.get('error', '未知错误'))
            processing_status[task_id].update({
                'status': 'error',
                'error': error_msg,
                'last_update': current_time
            })
            logger.error(f"下载错误 - Task: {task_id}, Error: {error_msg}")
            
    except Exception as e:
        error_msg = f"更新进度时出错: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        if task_id in processing_status:
            processing_status[task_id].update({
                'status': 'error',
                'error': error_msg,
                'last_update': time.time()
            })

@app.route('/analyze_image', methods=['POST'])
def handle_image_analysis():
    """处理单张图片分析请求"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': '没有上传图片'}), 400
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
            
        if not allowed_image_file(file.filename):
            return jsonify({'error': '不支持的图片格式'}), 400
            
        # 生成安全的文件名并保存图片
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}_{filename}")
        
        file.save(image_path)
        
        try:
            # 分析图片
            analysis_result = analyze_image(image_path)
            
            # 返回分析结果
            return jsonify({
                'success': True,
                'analysis': analysis_result
            })
            
        finally:
            # 清理临时文件
            try:
                if os.path.exists(image_path):
                    os.remove(image_path)
            except Exception as e:
                logger.error(f"清理临时文件失败: {str(e)}")
                
    except Exception as e:
        error_msg = f"处理图片分析请求失败: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500

def preprocess_frame(frame, target_width=960):
    """预处理视频帧"""
    height, width = frame.shape[:2]
    # 保持宽高比缩放
    ratio = target_width / width
    new_height = int(height * ratio)
    frame = cv2.resize(frame, (target_width, new_height))
    return frame

def update_task_status(task_id, status, progress=None, message=None, output_file=None):
    """更新任务状态"""
    if task_id in processing_status:
        processing_status[task_id].update({
            'status': status,
            'last_update': time.time()
        })
        
        if progress is not None:
            processing_status[task_id]['progress'] = progress
            
        if message is not None:
            processing_status[task_id]['message'] = message
            
        if output_file is not None:
            processing_status[task_id]['output_file'] = output_file

def generate_srt(subtitles, video_path):
    """生成SRT格式字幕文件"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.basename(video_path)
    base_name = os.path.splitext(filename)[0]
    srt_path = os.path.join(TEXT_FOLDER, f"{timestamp}_{base_name}.srt")
    
    with open(srt_path, 'w', encoding='utf-8') as f:
        for i, subtitle in enumerate(subtitles, 1):
            # 转换时间戳为SRT格式
            start_time = subtitle['timestamp']
            # 假设每个字幕显示3秒
            end_time = start_time + 3
            
            start_str = time.strftime('%H:%M:%S,000', time.gmtime(start_time))
            end_str = time.strftime('%H:%M:%S,000', time.gmtime(end_time))
            
            # 写入SRT格式
            f.write(f"{i}\n")
            f.write(f"{start_str} --> {end_str}\n")
            f.write(f"{subtitle['text']}\n\n")
    
    return srt_path

@app.route('/api/open_folder')
def open_output_folder():
    """打开输出文件夹"""
    try:
        # 从查询参数获取文件夹路径
        folder_path = request.args.get('folder')
        if not folder_path:
            return jsonify({
                'success': False,
                'error': '未指定文件夹路径'
            })

        # 构建完整路径
        full_path = os.path.join(FRAMES_FOLDER, folder_path)
        
        # 确保文件夹存在且在允许的目录内
        if not os.path.exists(full_path) or not os.path.realpath(full_path).startswith(os.path.realpath(FRAMES_FOLDER)):
            return jsonify({
                'success': False,
                'error': '文件夹不存在或路径无效'
            })

        # 获取绝对路径
        abs_path = os.path.abspath(full_path)
        
        # 获取操作系统类型
        system = platform.system()
        
        if system == 'Windows':
            os.startfile(abs_path)
        elif system == 'Darwin':  # macOS
            subprocess.Popen(['open', abs_path])
        elif system == 'Linux':
            subprocess.Popen(['xdg-open', abs_path])
        else:
            return jsonify({
                'success': False, 
                'error': f'不支持的操作系统: {system}'
            })
            
        return jsonify({
            'success': True,
            'message': '文件夹已打开'
        })
        
    except Exception as e:
        logger.error(f"打开文件夹失败: {str(e)}")
        return jsonify({
            'success': False, 
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
