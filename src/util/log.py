import datetime
from logging import (getLogger, FileHandler, StreamHandler,
                     Formatter, INFO, DEBUG)
import pathlib
import time


"""
# ログ出力について

* ロガーは各モジュール別に logging.getLogger(モジュール名) によって作成することで、
  それぞれのログがどのモジュールから出力されたかわかりやすくする (root ロガーは利用しない)

* ログ出力先は ログ出力ディレクトリ以下に作成
  その中で、通常のログとスタックトレースのダンプに分けて出力する

  * log_{yyyymmdd}.log : 全レベルのログ
  * log_{yyyymmdd}.trace : スタックトレースのダンプ (未実装)

# レベル定義

- CRITICAL: サービス全体を停止するレベルの例外が発生した場合 (アプリケーションレイヤではおそらく存在しない)
- ERROR: ある処理を停止＆スキップするレベルの例外が発生した場合
- WARNING: なんらかの意図しない状態になったが、復旧/処理継続ができる場合
- INFO: 処理状況を出力する場合 (障害発生時にモジュールが特定できるレベルの出力をする)
- DEBUG: numpy など利用パッケージのデバッグに出力が必要な場合
"""

_TIME_FMT = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
_FILE_FMT = 'log_{}.log'

_FORMATTER = Formatter('%(asctime)s [%(name)s] [%(levelname)s] %(message)s')

# TODO: Read log file
# _LOG_DIR = pathlib.Path(CONFIG['log']['log_dir'])
_LOG_DIR = pathlib.Path('log')

if not _LOG_DIR.exists():
    _LOG_DIR.mkdir()


def get_logger(name):
    """
    Logger 名を元に Logger インスタンスを返す。

    :param name: ロガーの名称、通常はモジュール名
    :return: ロガー
    """

    log_file = _LOG_DIR / _FILE_FMT.format(_TIME_FMT)

    logger = getLogger(name)

    # for file output
    fout = FileHandler(filename=str(log_file), mode='w')
    fout.setFormatter(_FORMATTER)
    logger.addHandler(fout)

    # for stdout
    stdout = StreamHandler()
    stdout.setFormatter(_FORMATTER)
    logger.addHandler(stdout)

    # level = CONFIG['log']['level'].upper()
    level = 'DEBUG'
    if level == 'INFO':
        logger.setLevel(INFO)
    elif level == 'DEBUG':
        logger.setLevel(DEBUG)
    else:
        msg = 'log level must be either INFO or DEBUG, given: {}'
        raise ValueError(msg.format(level))
    return logger
