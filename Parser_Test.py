import argparse

parser = argparse.ArgumentParser(description='このプログラムの説明')

parser.add_argument('--N',help='試行回数')
parser.add_argument('--decktype1',help='プレイヤー1のデッキタイプ')
parser.add_argument('--decktype2',help='プレイヤー2のデッキタイプ')
parser.add_argument('--filename',help='ファイル名')

args = parser.parse_args()

print('N='+args.N)
print('decktype1='+args.decktype1)
print('decktype2='+args.decktype2)
print('filename='+args.filename)