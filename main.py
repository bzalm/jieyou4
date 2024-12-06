import os
import asyncio
from typing import Optional
from fastapi import FastAPI, Request, UploadFile, File, Form, Body
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import yt_dlp
from pathlib import Path
import aiofiles
import uuid
import logging
from moviepy.editor import VideoFileClip

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建必要的目录
BASE_DIR = Path(__file__).resolve().parent
DOWNLOAD_DIR = BASE_DIR / "downloads"
AUDIO_DIR = BASE_DIR / "audio"
STATIC_DIR = BASE_DIR / "static"

# 确保目录存在
for dir_path in [DOWNLOAD_DIR, AUDIO_DIR, STATIC_DIR]:
    dir_path.mkdir(exist_ok=True)

# 创建 FastAPI 应用实例
app = FastAPI(title="YouTube Downloader")

# 挂载静态文件
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/downloads", StaticFiles(directory=str(DOWNLOAD_DIR)), name="downloads")
app.mount("/audio", StaticFiles(directory=str(AUDIO_DIR)), name="audio")

# 配置模板
templates = Jinja2Templates(directory="templates")

class VideoDownloader:
    def __init__(self):
        self.ydl_opts = {
            'format': 'best',
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'nocheckcertificate': True,
            'ignoreerrors': True,
            'outtmpl': str(DOWNLOAD_DIR / '%(title)s.%(ext)s')
        }

    async def get_video_info(self, url: str) -> dict:
        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return {
                    'title': info.get('title', ''),
                    'duration': info.get('duration', 0),
                    'thumbnail': info.get('thumbnail', ''),
                    'formats': [
                        {
                            'format_id': f['format_id'],
                            'ext': f['ext'],
                            'format': f['format'],
                            'filesize': f.get('filesize', 0),
                            'resolution': f.get('resolution', 'unknown')
                        }
                        for f in info['formats']
                        if f.get('resolution') != 'audio only'
                    ]
                }
        except Exception as e:
            logger.error(f"Error extracting video info: {str(e)}")
            raise Exception(f"获取视频信息失败: {str(e)}")

    async def download_video(self, url: str, format_id: Optional[str] = None) -> dict:
        try:
            # 生成唯一文件名
            video_filename = f"{uuid.uuid4()}"
            self.ydl_opts['outtmpl'] = str(DOWNLOAD_DIR / f"{video_filename}.%(ext)s")
            
            if format_id:
                self.ydl_opts['format'] = format_id

            # 下载视频
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                ext = info['ext']
                video_path = DOWNLOAD_DIR / f"{video_filename}.{ext}"
                
                if not video_path.exists():
                    raise Exception(f"视频下载失败: 文件 {video_path} 不存在")

                # 提取音频
                audio_filename = f"{video_filename}_audio.mp3"
                audio_path = AUDIO_DIR / audio_filename
                
                video = VideoFileClip(str(video_path))
                video.audio.write_audiofile(str(audio_path))
                video.close()

                return {
                    "video_url": f"/downloads/{video_filename}.{ext}",
                    "audio_url": f"/audio/{audio_filename}",
                    "title": info.get('title', '未知标题')
                }
                
        except Exception as e:
            logger.error(f"Error downloading video: {str(e)}")
            raise Exception(f"下载视频失败: {str(e)}")

# 创建下载器实例
downloader = VideoDownloader()

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.post("/get_video_info")
async def get_video_info(url: str = Body(..., embed=True)):
    try:
        info = await downloader.get_video_info(url)
        return JSONResponse(content=info)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/download")
async def download_video(url: str = Form(...), format_id: Optional[str] = Form(None)):
    try:
        result = await downloader.download_video(url, format_id)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
