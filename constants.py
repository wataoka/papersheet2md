import os
from glob import glob

# dir
HOME_DIR : str = os.path.expanduser('~')
BASE_DIR : str = os.path.dirname(os.path.abspath(__file__))
NOTE_DIR : str = os.path.join(BASE_DIR, 'note')

# path
APIKEY_PATH : str = "<PATH to Your Google API KEY>"
SHEETS_KEY  : str = "<KEY of Your Google Speadsheet>"
SCOPE             : list = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']

SPREADSHEETS_NAME : str  = "<Your Spreadsheet File Name>"
SHEETS_NAME       : str  = "<Your Sheet Name>"

# note
NOTE_FILEPATH_LIST : list = glob(f'{NOTE_DIR}/*.txt')
NOTE_FILENAME_LIST : list = []
NOTE_FILEID_LIST   : list = []
for filepath in NOTE_FILEPATH_LIST:
    NOTE_FILENAME_LIST.append(filepath.split('/')[-1])
for filename in NOTE_FILENAME_LIST:
    NOTE_FILEID_LIST.append(filename.split('.')[0])