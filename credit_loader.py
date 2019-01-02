import csv

def process_credit() :
    f = open('../credit.csv')
    r = csv.reader(f)

    result = []
    for row in r :
        result.append(row)

    return result