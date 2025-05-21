# main.py - 主程序入口

import os
import sys
import time
import threading
import argparse

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入配置和模块
import config
from controller import InterpreterController
from utils.debug_utils import log, LOG_CATEGORY_GENERAL, enable_file_logging


def print_header():
    """打印程序头信息"""
    print("=" * 80)
    print("实时口译系统 - 智能句子管理版")
    print("=" * 80)
    print(f"支持的语言: {', '.join([f'{name}({code})' for code, name in config.LANGUAGE_DISPLAY_NAMES.items()])}")
    print("-" * 80)


def print_instructions():
    """打印使用说明"""
    print("\n使用说明:")
    print("1. 选择目标语言")
    print("2. 启动系统后，对着麦克风说话")
    print("3. 系统会智能识别句子并进行翻译和朗读")
    print("4. 按Enter键可以停止系统")
    print("-" * 80)


def command_line_interface(controller):
    """简单的命令行界面"""
    # 打印语言选项
    print("\n请选择目标语言:")
    for i, (code, name) in enumerate(config.LANGUAGE_DISPLAY_NAMES.items(), 1):
        if not hasattr(config, 'ALLOW_SAME_SOURCE_TARGET') or not config.ALLOW_SAME_SOURCE_TARGET:
            if code == config.DEFAULT_SOURCE_LANG:  # 不显示源语言
                continue
        print(f"{i - 1}. {name} ({code})")

    # 获取用户选择
    while True:
        try:
            choice = input("\n请输入目标语言编号 (默认为英语): ").strip()

            # 默认选择英语
            if not choice:
                target_lang = "en"
                break

            choice = int(choice)
            target_langs = list(config.LANGUAGE_DISPLAY_NAMES.keys())

            if 0 <= choice < len(target_langs):
                target_lang = target_langs[choice]
                break
            else:
                print("无效的选择，请重试")
        except ValueError:
            print("请输入数字")

    print(f"\n已选择目标语言: {config.LANGUAGE_DISPLAY_NAMES[target_lang]}")

    # 启动系统
    print("\n正在启动实时口译系统...")
    controller.start(target_lang)

    print("\n系统已启动，对着麦克风说话即可...")
    print("按Enter键停止系统")

    # 显示线程 - 实时显示识别和翻译结果
    def display_thread():
        last_source = ""
        last_translation = ""

        while controller.is_running:
            try:
                # 处理事件
                while not controller.event_queue.empty():
                    event = controller.event_queue.get()

                    if event["type"] == "source_text_update":
                        new_text = event["data"]["text"]
                        if new_text != last_source:
                            print(f"\r原文: {new_text}", end="")
                            last_source = new_text

                    elif event["type"] == "translated_text_update":
                        new_text = event["data"]["text"]
                        if new_text != last_translation:
                            print(f"\n译文: {new_text}")
                            last_translation = new_text

                    elif event["type"] == "tts_play":
                        print(f"\n[正在朗读译文...]")

                    elif event["type"] == "tts_stop":
                        print(f"\n[语音播放已停止]")

                # 短暂休眠
                time.sleep(0.1)
            except Exception as e:
                print(f"显示线程错误: {e}")

    # 启动显示线程
    display_thr = threading.Thread(target=display_thread, daemon=True)
    display_thr.start()

    # 等待用户按Enter停止
    input()

    # 停止系统
    controller.stop()
    print("\n实时口译系统已停止")


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="实时口译系统 - 智能句子管理版")
    parser.add_argument("--gui", action="store_true", help="使用图形用户界面")
    parser.add_argument("--debug", action="store_true", help="启用调试日志")
    parser.add_argument("--log-file", help="设置日志文件")
    args = parser.parse_args()

    # 设置调试选项
    if args.debug:
        from utils.debug_utils import set_log_level, LOG_LEVEL_DEBUG
        set_log_level(LOG_LEVEL_DEBUG)

    # 设置日志文件
    if args.log_file:
        enable_file_logging(args.log_file)

    # 显示程序头信息
    print_header()

    if args.gui:
        # 启动图形用户界面
        try:
            from gui import InterpreterGUI
            log("启动图形用户界面", LOG_CATEGORY_GENERAL)
            app = InterpreterGUI()
            app.mainloop()
        except ImportError as e:
            print(f"启动图形界面失败: {e}")
            print("可能是缺少必要的GUI库，尝试使用命令行界面。")
            # 如果GUI启动失败，回退到命令行界面
            run_cli()
    else:
        # 使用命令行界面
        run_cli()


def run_cli():
    """运行命令行界面"""
    # 创建控制器
    controller = InterpreterController()

    # 初始化控制器
    if not controller.initialize():
        print("初始化失败，程序退出")
        return

    print_instructions()

    # 启动命令行界面
    command_line_interface(controller)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n程序发生错误: {e}")
        import traceback

        traceback.print_exc()

    print("\n感谢使用实时口译系统!")