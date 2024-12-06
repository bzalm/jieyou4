import os

# 创建cookies目录
cookies_dir = os.path.join(os.path.dirname(__file__), 'cookies')
if not os.path.exists(cookies_dir):
    os.makedirs(cookies_dir)
    print(f"Created directory: {cookies_dir}")
else:
    print(f"Directory already exists: {cookies_dir}")
