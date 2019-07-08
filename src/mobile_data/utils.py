import os
import pandas as pd


def hash_phone_numbers(folder, new_folder=None):
    files = os.listdir(folder)

    cnt = 0
    for f in files:
        try:
            if f.endswith('csv'):
                fpath = os.path.join(folder, f)
                fpath2 = os.path.join(new_folder, f)
                if os.path.exists(fpath2):
                    continue
                df = pd.read_csv(fpath)
                keep = ['cdr type',
                        'cdr datetime',
                        'call duration',
                        'calling phonenumber',
                        'last calling cellid']
                df = df[keep]

                df['calling phonenumber2'] = df.apply(lambda x: abs(hash(str(x['calling phonenumber']))), axis=1)
                df.drop(labels=['calling phonenumber'], axis=1, inplace=True)
                df.to_csv(fpath2, index=False)
                cnt += 1
                if cnt % 1000 == 0:
                    print('Done with {} files so far'.format(cnt))

        except Exception as e:
            print(e)
            continue


if __name__ == '__main__':
    raw_cdrs = '/Users/dmatekenya/Google-Drive/gigs/aims-dakar-2019/day5-case-studies/africell/'
    hashed_cdrs = '/Users/dmatekenya/Google-Drive/gigs/aims-dakar-2019/day5-case-studies/cdrs/'
    hash_phone_numbers(folder=raw_cdrs, new_folder=hashed_cdrs)


