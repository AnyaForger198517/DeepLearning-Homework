import logging
import argparse


def create_logger(logger_file_name):
    """
    :param logger_file_name:
    :return:  日志记录器对象
    """
    logger = logging.getLogger()         # 设定日志对象
    logger.setLevel(logging.INFO)        # 设定日志等级

    file_handler = logging.FileHandler(logger_file_name)   # 文件输出
    console_handler = logging.StreamHandler()              # 控制台输出

    # 输出格式
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s "
    )

    file_handler.setFormatter(formatter)       # 设置文件输出格式
    console_handler.setFormatter(formatter)    # 设施控制台输出格式
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='configTemplates')
    parser.add_argument('-log_path', default='./train.log', type=str, help='log file path to save result')

    args = parser.parse_args()

    logger = create_logger(args.log_path)

    logger.info('Begin Training Model...')