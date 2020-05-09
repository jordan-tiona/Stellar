from finviz.screener import Screener

filters = ['exch_nyse', 'sh_avgvol_o300', 'sh_price_u30']
symbols = Screener(order='-perf52w', filters=filters)

with open('./training-set.txt', 'w') as f:
    for row in symbols.data:
        print(row)
        f.write(row['Ticker'] + '\n')