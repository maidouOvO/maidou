import os
from main import process_image

def test_with_sample():
    input_path = os.path.expanduser("~/attachments/7cce3879-dd5b-4323-b4ea-27f2bb77fa46/test1.jpg")
    output_dir = os.path.expanduser("~/repos/maidou/text_remover/results")
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/test1_result.jpg"
    
    # 使用较低的检测阈值处理图像
    process_image(input_path, output_path, text_threshold=0.5, link_threshold=0.3, low_text=0.3)
    
    # 使用优化的中空文字检测和较低的检测阈值
    output_path_optimized = f"{output_dir}/test1_result_optimized.jpg"
    process_image(input_path, output_path_optimized, text_threshold=0.5, link_threshold=0.3, low_text=0.3, optimize_hollow=True)
    
    print(f"测试完成，结果保存在: {output_dir}")

if __name__ == "__main__":
    test_with_sample()
