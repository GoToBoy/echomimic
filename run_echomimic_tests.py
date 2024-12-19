import subprocess
import time
import pandas as pd
import numpy as np
from itertools import product
import datetime
import yaml
import re
import sys
import json
from pathlib import Path
from threading import Thread
from queue import Queue, Empty
import argparse

class TestProgress:
    def __init__(self, total_combinations, progress_file="test_progress.json"):
        # TODO: change path
        self.progress_file = Path("/root/autodl-fs/echomimic_output/progress") / progress_file
        self.progress_file.parent.mkdir(exist_ok=True, parents=True)
        self.total = total_combinations
        self.completed = self.load_progress()

    def load_progress(self):
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return set(json.load(f))
        return set()

    def save_progress(self):
        with open(self.progress_file, 'w') as f:
            json.dump(list(self.completed), f)

    def mark_completed(self, combination_key):
        self.completed.add(combination_key)
        self.save_progress()

    def is_completed(self, combination_key):
        return combination_key in self.completed

def run_echomimic(steps, cfg, facemusk_ratio, facecrop_ratio):
    start_time = time.time()
    
    command = [
        "python", "-u", "infer_audio2vid_acc_upscale.py",
        "--steps", str(steps),
        "--cfg", str(cfg),
        "--facemusk_dilation_ratio", str(facemusk_ratio),
        "--facecrop_dilation_ratio", str(facecrop_ratio)
    ]
    
    try:
        # 创建进程
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            universal_newlines=False
        )

        # 创建队列存储输出
        stdout_queue = Queue()
        stderr_queue = Queue()

        # 创建线程读取输出
        stdout_thread = Thread(target=stream_output, args=(process.stdout, stdout_queue))
        stderr_thread = Thread(target=stream_output, args=(process.stderr, stderr_queue))

        # 启动线程
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        stdout_thread.start()
        stderr_thread.start()

        # 存储完整输出
        complete_output = []

        # 实时显示输出
        while process.poll() is None:
            # 处理标准输出
            try:
                while True:
                    line = stdout_queue.get_nowait()
                    complete_output.append(line)
                    print(line.strip())
                    sys.stdout.flush()
            except Empty:
                pass

            # 处理标准错误
            try:
                while True:
                    line = stderr_queue.get_nowait()
                    print(line.strip(), file=sys.stderr)
                    sys.stderr.flush()
            except Empty:
                pass

            time.sleep(0.1)

        # 确保所有输出都被读取
        stdout_thread.join()
        stderr_thread.join()

        # 获取剩余的输出
        while not stdout_queue.empty():
            line = stdout_queue.get()
            complete_output.append(line)
            print(line.strip())

        while not stderr_queue.empty():
            line = stderr_queue.get()
            print(line.strip(), file=sys.stderr)

        end_time = time.time()
        total_duration = end_time - start_time
        
        # 将完整输出合并成字符串
        complete_output_str = ''.join(complete_output)
        
        # 提取视频路径
        video_paths = extract_video_paths(complete_output_str)
        
        # 获取测试用例数量
        test_cases = load_test_cases()
        total_cases = sum(len(audios) for audios in test_cases.values())
        
        # 计算平均时间
        avg_duration = total_duration / total_cases if total_cases > 0 else 0
        
        return {
            'status': 'success' if process.returncode == 0 else 'failed',
            'total_duration': total_duration,
            'avg_duration': avg_duration,
            'video_count': len(video_paths),
            'video_paths': video_paths
        }
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'status': 'error',
            'total_duration': -1,
            'avg_duration': -1,
            'video_count': 0,
            'video_paths': [str(e)]
        }

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-start', type=int, default=0,
                       help='Starting batch index')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Number of combinations per batch')
    parser.add_argument('--max-batches', type=int, default=None,
                       help='Maximum number of batches to process')
    return parser.parse_args()

# [保持原有的辅助函数不变]
def stream_output(pipe, queue):
    """实时读取并输出管道内容"""
    for line in iter(pipe.readline, b''):
        queue.put(line.decode('utf-8'))
    pipe.close()

def load_test_cases(yaml_path="configs/prompts/animation_acc_upScale.yaml"):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    test_cases = config.get('test_cases', {})
    return test_cases

def extract_video_paths(stdout):
    base_path = "/root/autodl-fs/echomimic_output"
    video_paths = re.findall(f'{base_path}/.*?\.mp4', stdout)
    video_paths = [path for path in video_paths if not 'temp_' in path]
    return video_paths

def check_disk_space(min_gb=20):
    """检查磁盘空间"""
    #TODO change output path
    import shutil
    path = "/root/autodl-fs" 
    total, used, free = shutil.disk_usage(path)
    free_gb = free // (2**30)
    print(f"\nTotal parameter used: {used} free: {free_gb}")
    return free_gb >= min_gb

def save_batch_results(results, batch_idx):
    """保存每个批次的结果"""
    #TODO change output path
    results_dir = Path("/root/autodl-fs/echomimic_output/results")
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # 保存批次结果
    df = pd.DataFrame(results)
    df.to_csv(results_dir / f"batch_{batch_idx}_results.csv", index=False)
    
    # 合并所有已完成批次的结果
    all_results = []
    for result_file in results_dir.glob("batch_*_results.csv"):
        batch_df = pd.read_csv(result_file)
        all_results.append(batch_df)
    
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_df.to_csv(results_dir / "combined_results.csv", index=False)
        
        # 保存Excel版本
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_path = results_dir / f"combined_results_{timestamp}.xlsx"
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            combined_df.to_excel(writer, index=False, sheet_name='Results')
            
            workbook = writer.book
            worksheet = writer.sheets['Results']
            
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'bg_color': '#D7E4BC',
                'border': 1
            })
            
            for col_num, value in enumerate(combined_df.columns.values):
                worksheet.write(0, col_num, value, header_format)

def main():
    args = parse_args()
    
    # 参数范围定义
    steps_range = range(6, 32, 2)
    cfg_range = [round(x, 1) for x in np.arange(1.0, 2.6, 0.1)]
    facemusk_range = [round(x, 1) for x in np.arange(0.1, 0.3, 0.1)]
    facecrop_range = [round(x, 1) for x in np.arange(0.5, 1.1, 0.1)]

    # 生成所有组合
    all_combinations = list(product(steps_range, cfg_range, facemusk_range, facecrop_range))
    total_combinations = len(all_combinations)
    
    # 初始化进度跟踪器
    progress = TestProgress(total_combinations)
    
    # 计算批次信息
    total_batches = (total_combinations + args.batch_size - 1) // args.batch_size
    start_batch = args.batch_start
    end_batch = total_batches if args.max_batches is None else min(start_batch + args.max_batches, total_batches)

    print(f"\nTotal parameter combinations: {total_combinations}")
    print(f"Processing batches {start_batch + 1} to {end_batch} (batch size: {args.batch_size})")

    for batch_idx in range(start_batch, end_batch):
        batch_start = batch_idx * args.batch_size
        batch_end = min(batch_start + args.batch_size, total_combinations)
        batch_combinations = all_combinations[batch_start:batch_end]
        
        print(f"\nProcessing batch {batch_idx + 1}/{total_batches}")
        
        batch_results = []
        for idx, (steps, cfg, facemusk, facecrop) in enumerate(batch_combinations, 1):
            # 生成组合的唯一标识
            combination_key = f"{steps}_{cfg}_{facemusk}_{facecrop}"
            
            if progress.is_completed(combination_key):
                print(f"Skipping completed combination: {combination_key}")
                continue

            print(f"\n{'='*80}")
            print(f"Combination {batch_start + idx}/{total_combinations}")
            print(f"Parameters: steps={steps}, cfg={cfg}, facemusk={facemusk}, facecrop={facecrop}")
            print(f"{'='*80}\n")

            try:
                result = run_echomimic(steps, cfg, facemusk, facecrop)
                
                result_data = {
                    'steps': steps,
                    'cfg': cfg,
                    'facemusk_ratio': facemusk,
                    'facecrop_ratio': facecrop,
                    'total_duration': result['total_duration'],
                    'avg_duration': result['avg_duration'],
                    'video_count': result['video_count'],
                    'status': result['status'],
                    'video_paths': '\n'.join(result['video_paths'])
                }
                
                batch_results.append(result_data)
                progress.mark_completed(combination_key)
                
                # 打印当前组合的结果摘要
                print(f"\nCombination Results:")
                print(f"Total Duration: {result['total_duration']:.2f} seconds")
                print(f"Average Duration per video: {result['avg_duration']:.2f} seconds")
                print(f"Videos Generated: {result['video_count']}")
                print(f"Status: {result['status']}")
                
            except Exception as e:
                print(f"Error processing combination {combination_key}: {str(e)}")
                continue

        # 保存批次结果
        save_batch_results(batch_results, batch_idx)
        
        # 检查磁盘空间
        if not check_disk_space():
            print("Low disk space. Please free up space before continuing.")
            break

if __name__ == "__main__":
    main()

# 
# 正常运行（处理100个组合）： 
# python run_echomimic_tests.py --batch-size 100 --max-batches 1
# 
# 从特定批次开始
# python run_echomimic_tests.py --batch-start 5 --batch-size 100 --max-batches 1
#
# 一次处理多个批次：
# python run_echomimic_tests.py --batch-size 100 --max-batches 5



