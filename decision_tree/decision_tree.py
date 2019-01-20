import credit_loader
import math
from collections import defaultdict

def cal_entropy(list_of_prob_num) :
    print(list_of_prob_num)

    entire_num = sum(list_of_prob_num)

    result = 0.0
    for i in range(len(list_of_prob_num)) :
        p1 = list_of_prob_num[i] / entire_num
        result += -1 * p1 * (math.log(p1) / math.log(len(list_of_prob_num)))

    return result

def conditional_entropy(bef_prob, joint_prob) :
    result = 0.0
    for i in range(len(joint_prob)) :
        p1 = bef_prob[i]
        for j in range(len(joint_prob[0])) :
            p2 = joint_prob[i][j] / p1
            try :
                # p2가 0일 경우 log 에러 발생
                result += p1 * -1 * p2 * (math.log(p2) / math.log(len(joint_prob[0])))
            except Exception as e:
                print(e)
                return 1.0

    return result

def make_conditional_set(factor_idx, datas) :
    total_case_num = len(datas)

    # x와 y로 묶어서 데이터 정리
    a = [(d['x'][factor_idx], d['y']) for d in datas]

    print(a)

    a_counter = {}
    y_list = [] # y 값의 라벨링 유지


    for d in a:
        if d[0] not in a_counter.keys():
            a_counter[d[0]] = []

        a_counter[d[0]].append(d)

        # y 값의 라벨링 유지
        for y in d[1] :
            if y not in y_list :
                y_list.append(y)


    bef_prob = []

    # x 값의 라벨링 유지
    k_list = list(a_counter.keys())
    for k in k_list :
        bef_prob.append(len(a_counter[k]) / total_case_num) # x 값에 대한 MLE 계산
    
    print(bef_prob)

    # (x 팩터의 개수, y 팩터의 개수) 행렬의 교집합 확률 변수
    joint_prob = []

    for i, x in enumerate(k_list) :
        joint_prob.append([])
        for j, y in enumerate(y_list) :
            y_num = len([d for d in a_counter[x] if d[1] == y])

            joint_prob[i].append(y_num / total_case_num) # P(Y=y 교 X=x)의 확률

    print(joint_prob)

    return k_list, y_list, bef_prob, joint_prob

def inf_gain(factor_idx, datas) :
    y_dic = defaultdict(lambda : 0)

    for d in datas :
        y_dic[d['y']] += 1

    ori_entropy = cal_entropy(list(y_dic.values())) # H(Y)

    #print(ori_entropy)

    x_list, y_list, bef_prob, joint_prob = make_conditional_set(factor_idx, datas)

    af_entropy = conditional_entropy(bef_prob, joint_prob) # H(Y|A_i)

    # IG(Y, A_i) = H(Y) - H(Y|A_i)
    return ori_entropy - af_entropy

class DecisionTree :
    def __init__(self, datas) :
        self.datas = datas
        self.tree = {}
        self.av_x = [] # 범주형 자료인 x의 idx

    def make_tree(self) :

        # 루트 노드 만들기
        root_x = None
        max_ig = 0
        for i in range(len(datas[0]['x'])) :
            ig = inf_gain(i, self.datas)
            print(ig)

            if ig > 0 :
                self.av_x.append(i)

                if ig > max_ig :
                    max_ig = ig
                    root_x = i


        print(root_x)
        print(self.av_x)


if __name__ == "__main__" :
    credit = credit_loader.process_credit()

    print(len(credit))
    #print(credit)
    datas = []
    for i in range(len(credit)) :

        datas.append({'x': credit[i][:-1], 'y': credit[i][len(credit[0]) - 1:][0]})

    dt = DecisionTree(datas)

    dt.make_tree()