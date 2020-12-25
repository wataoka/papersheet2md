import os
from glob import glob

# dir
HOME_DIR : str = os.path.expanduser('~')
BASE_DIR : str = os.path.dirname(os.path.abspath(__file__))
NOTE_DIR : str = os.path.join(BASE_DIR, 'note')

# path
APIKEY_PATH       : str  = os.path.join(HOME_DIR, '.apikeys', 'google-api-key.json')
SHEETS_KEY        : str  = "1aIJFzaYehW3n0c-U4mvkxztyXE8xqqxa1_E29AzlMtw"
SCOPE             : list = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']

SPREADSHEETS_NAME : str  = "paper"
SHEETS_NAME       : str  = "machine learning"

# note
NOTE_FILEPATH_LIST : list = glob(f'{NOTE_DIR}/*.txt')
NOTE_FILENAME_LIST : list = []
NOTE_FILEID_LIST   : list = []
for filepath in NOTE_FILEPATH_LIST:
    NOTE_FILENAME_LIST.append(filepath.split('/')[-1])
for filename in NOTE_FILENAME_LIST:
    NOTE_FILEID_LIST.append(int(filename.split('.')[0]))

# ignore
IGNORE_ID_LIST : list = [ 69,  70,  72,  75,  80,  84,  92,  93, 105, 107,
                         108, 115, 116, 122, 166, 167, 178, 134, 136, 182]