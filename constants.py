import os

# dir
HOME_DIR : str = os.path.expanduser('~')
BASE_DIR : str = os.path.dirname(os.path.abspath(__file__))

# path
APIKEY_PATH : str = "<PATH to Your Google API KEY>"
SHEETS_KEY  : str = "<KEY of Your Google Speadsheet>"
SCOPE             : list = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']

SPREADSHEETS_NAME : str  = "<Your Spreadsheet File Name>"
SHEETS_NAME       : str  = "<Your Sheet Name>"