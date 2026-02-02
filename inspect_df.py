from data_fetch import build_base_dataframe

if __name__ == '__main__':
    df = build_base_dataframe()
    print('rows:', len(df))
    if len(df) == 0:
        print('empty df')
    else:
        print('index type:', type(df.index))
        print('first 10:', list(df.index[:10]))
        print('last 10:', list(df.index[-10:]))
        try:
            print('min index:', df.index.min())
            print('max index:', df.index.max())
        except Exception as e:
            print('min/max error', e)
