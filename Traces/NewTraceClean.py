import pandas

def clean(file_name):


    file = pandas.read_csv(file_name + '.csv', sep=',', header=None)
    res = []

    for i in range(len(file[2])):
        value, digit = file[2][i], file[3][i]
        if digit == 'Kbits/sec':
            res.append(str(round(value / 1000.0, 3)) + '\n')
        else:
            res.append(str(value) + '\n')
    # print(res)
    with open(file_name + '_clean.txt', 'w', encoding='utf-8') as writer:
        writer.writelines(res)


clean('Ntrain(wipeQ)')

