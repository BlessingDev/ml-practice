import csv

def process_credit() :
    f = open('../credit.csv')
    r = csv.reader(f)

    result = []
    for row in r :
        result.append(row)

    return result

def process_housing() :
    f = open('../housing.data')
    r = csv.reader(f, delimiter=' ')

    result = []
    for row in r :
        blank_num = 0
        for i, d in enumerate(row) :
            if d == '' :
                blank_num += 1
            else :
                row[i] = float(d.strip())
        for i in range(blank_num) :
            row.remove('')

        result.append(row)

    return result