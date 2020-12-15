import os
import sys
import json

import gspread
from oauth2client.service_account import ServiceAccountCredentials

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE))
from constants import (
    APIKEY_PATH,
    SPREADSHEETS_NAME,
    SHEETS_NAME, 
    SCOPE,
)



def get_sheet(sheet_name):
    credentials = ServiceAccountCredentials.from_json_keyfile_name(APIKEY_PATH, SCOPE)
    client = gspread.authorize(credentials)
    if credentials.access_token_expired:
        client.login()

    spreadsheets = client.open(SPREADSHEETS_NAME)
    sheet = spreadsheets.worksheet(sheet_name)
    return sheet


def main():
    sheet = get_sheet(SHEETS_NAME)


if __name__ == "__main__":
    main()