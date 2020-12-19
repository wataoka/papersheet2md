import os
import sys
import json
import pandas as pd

import gspread
from oauth2client.service_account import ServiceAccountCredentials

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE))
from constants import (
    APIKEY_PATH,
    SPREADSHEETS_NAME,
    SHEETS_NAME, 
    SCOPE,
    NOTE_FILEID_LIST,
    IGNORE_ID_LIST 
)


def get_sheet(sheet_name):
    credentials = ServiceAccountCredentials.from_json_keyfile_name(APIKEY_PATH, SCOPE)
    client = gspread.authorize(credentials)
    if credentials.access_token_expired:
        client.login()

    spreadsheets = client.open(SPREADSHEETS_NAME)
    sheet = spreadsheets.worksheet(sheet_name)
    return sheet


def check_note(row):
    # check note
    if not str(row+2) in NOTE_FILEID_LIST:
        return None
    # read note
    with open(f'note/{row+2}.txt', 'r') as f:
        data = f.read()
    return data


def main(filename='output.md'):

    print('loading google sheet...')
    sheet = get_sheet(SHEETS_NAME)
    sheet_df = pd.DataFrame(sheet.get_all_records())

    row = 61
    print('writing markdown...')
    with open(filename, 'w') as f:

        print('この記事は私, wataokaが一人で2020年の**1年間をかけて**作り続けた論文要約の**超大作記事**です.\n', file=f)

        print('# 論文100本解説\n', file=f)
        while True:
            row += 1

            # get row
            row_df = sheet_df[row:row+1]
            # check end
            if len(row_df) == 0:
                break
            # check ignore
            if row+2 in IGNORE_ID_LIST:
                continue

            # set values
            title_en = row_df['論文名'].values[0]
            title_ja = row_df['論文名(日本語)'].values[0]
            tag      = row_df['タグ'].values[0]
            conf     = row_df['学会'].values[0]
            url      = row_df['リンク'].values[0]
            date     = row_df['投稿日付'].values[0]
            abst     = row_df['概要'].values[0]
            method   = row_df['手法'].values[0]
            result   = row_df['結果'].values[0]
            comment  = row_df['コメント'].values[0]

            assert not title_en == ''
            print(f'## {title_en}\n', file=f)
            if not title_ja == '':
                print(f'wataokaの日本語訳「{title_ja}」', file=f)
            if not tag == '':
                print(f'- 種類: {tag}', file=f)
            if not conf == '':
                print(f'- 学会: {conf}', file=f)
            if not date == '':
                print(f'- 日付: {date}', file=f)
            if not url == '':
                print(f'- URL: [{url}]({url})', file=f)
            print('\n', file=f)

            # check note
            note = check_note(row)
            if note is not None:
                print(note, file=f)
                continue

            if not abst == '':
                print(f'### 概要\n', file=f)
                print(f'{abst}\n', file=f)
            if not method == '':
                print(f'### 手法\n', file=f)
                print(f'{method}\n', file=f)
            if not result == '':
                print(f'### 結果\n', file=f)
                print(f'{result}\n', file=f)
            if not comment == '':
                print(f'### wataokaのコメント\n', file=f)
                print(f'{comment}\n', file=f)
    print('Done!')


if __name__ == "__main__":
    main()