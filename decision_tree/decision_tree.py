import credit_loader
import math
from collections import Counter

def cal_entropy(list_of_prob_num) :
    print(list_of_prob_num)

    entire_num = sum(list_of_prob_num)

    result = 0.0
    for i in range(len(list_of_prob_num)) :
        p1 = list_of_prob_num[i] / entire_num
        result += -1 * p1 * (math.log(p1) / math.log(len(list_of_prob_num)))

    return result

if __name__ == "__main__" :
    credit = credit_loader.process_credit()

    print(len(credit))
    #print(credit)
    probs = []
    for i in range(len(credit[0])) :
        prob = Counter([case[i] for case in credit]).values()
        probs.append(list(prob))

    for prob in probs :
        entropy = cal_entropy(prob)

        print(entropy)