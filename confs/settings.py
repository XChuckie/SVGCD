from utils import IDUtil
from utils import PathUtil
import os

ID = IDUtil.get_random_id_bytime()  # Run ID

WORK_DIR = os.getcwd()

ARCHIVE_FOLDER_PATH = f"{WORK_DIR}/archive"
DATA_FOLDER_PATH = f"{WORK_DIR}/data"
TEMP_FOLDER_PATH = f"{WORK_DIR}/temp"

PathUtil.auto_create_folder_path(
    TEMP_FOLDER_PATH,
    ARCHIVE_FOLDER_PATH,
    DATA_FOLDER_PATH,
)

DISABLE_TQDM_BAR = False
LOG_WITHOUT_DATE = False
TQDM_NCOLS = 100