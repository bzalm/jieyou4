from paddleocr import PaddleOCR
import os

def test_paddleocr():
    print("开始测试PaddleOCR...")
    try:
        # 初始化PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang='ch')
        print("PaddleOCR初始化成功!")
        
        # 获取当前目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"当前目录: {current_dir}")
        print("PaddleOCR安装和初始化测试完成!")
        return True
    except Exception as e:
        print(f"发生错误: {str(e)}")
        return False

if __name__ == "__main__":
    test_paddleocr()
