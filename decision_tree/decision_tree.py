import credit_loader
import math
import random
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
        self.cate_x = [] # 범주형 자료인 x의 idx

    def _get_x(self, x_idx, cur_datas) :
        x_dic = defaultdict(lambda : 0)
        for d in cur_datas :
            x_dic[d['x'][x_idx]] += 1

        return list(x_dic.keys())

    def _get_y(self) :
        y_dic = defaultdict(lambda : 0)
        for d in self.datas :
            y_dic[d['y']] += 1

        return list(y_dic.keys())

    def _determine_leaf_node(self, x_idx, cur_datas):
        y_count_dic = defaultdict(lambda : defaultdict(lambda : 0)) # 각 x에 대한 y의 각 값의 개수를 셀 dictionary

        for d in cur_datas :
            x_val = d['x'][x_idx]
            y_count_dic[x_val][d['y']] += 1

        x_list = self._get_x(x_idx, cur_datas)
        for x in x_list :
            if len(y_count_dic[x].keys()) > 1 :
                return (False, y_count_dic)

        return (True, y_count_dic)

    def _get_sub_set(self, x_idx, x_val, cur_datas):
        # x_idx의 값이 x_val인 data를 cur_datas에서 찾아서 반환한다
        new_data = []
        for d in cur_datas :
            if d['x'][x_idx] == x_val :
                new_data.append(d)

        return new_data

    def _make_leaf_node(self, tree, x_list, y_count_dic):
        for x in x_list:
            y_sorted_list = sorted([(k, y_count_dic[x][k]) for k in y_count_dic[x].keys()], key=lambda x: x[1],
                                   reverse=True)
            tree[x] = y_sorted_list[0][0]

    def _make_node(self, tree, x_idx, cur_datas, av_x) :
        av_x.remove(x_idx)

        is_leaf, y_count_dic = self._determine_leaf_node(x_idx, cur_datas)

        print()
        print('tree log', x_idx, dict(y_count_dic))
        print(self.tree)
        print()

        if not is_leaf\
            or len(av_x) > 0 :
            # 먼저 현재 노드가 단말 노드가 되는지 확인한다
            # 그리고 탐색가능한 노드가 비어있지 않은지 확인한다
            x_list = self._get_x(x_idx, cur_datas)

            for x in x_list :
                x_datas = self._get_sub_set(x_idx, x, cur_datas)

                max_gain = 0.0
                ig_idx = -1
                for de_x in av_x :
                    # 현재 탐색 가능한 x에 대해 가장 높은 ig를 얻을 수 있는 x를 찾는다
                    ig = inf_gain(de_x, x_datas)

                    if ig > max_gain :
                        max_gain = ig
                        ig_idx = de_x
                
                if ig_idx == -1 :
                    # ig를 얻을 수 없었다면
                    self._make_leaf_node(tree, x_list, y_count_dic)
                else :
                    # ig를 얻었다면
                    # 최종적으로 가장 많은 ig를 얻은 factor에 대해 node를 구성
                    tree[x] = (ig_idx, {})
                    self._make_node(tree[x][1], ig_idx, x_datas, list(av_x))
        else :
            x_list = self._get_x(x_idx, cur_datas)

            self._make_leaf_node(tree, x_list, y_count_dic)


    def make_tree(self) :

        # 루트 노드 만들기
        root_x = None
        max_ig = 0
        for i in range(len(datas[0]['x'])) :
            ig = inf_gain(i, self.datas)
            print(ig)

            if ig > 0 :
                self.cate_x.append(i)

                if ig > max_ig :
                    max_ig = ig
                    root_x = i


        print(root_x)
        print(self.cate_x)

        av_x = list(self.cate_x)
        self.tree = (root_x, {})
        self._make_node(self.tree[1], root_x, self.datas, av_x)

        print(self.tree)

    def determine(self, x_factors) :
        cur_node = self.tree

        while str(type(cur_node)) == "<class 'tuple'>" :
            cur_node = cur_node[1][x_factors[cur_node[0]]]

        return cur_node

if __name__ == "__main__" :
    credit = credit_loader.process_credit()

    print(len(credit))
    #print(credit)
    datas = []
    for i in range(len(credit)) :

        datas.append({'x': credit[i][:-1], 'y': credit[i][len(credit[0]) - 1:][0]})

    random.shuffle(datas)

    div_idx = int(len(datas) * 0.8)
    train_data, test_data = datas[:div_idx], datas[div_idx:]

    dt = DecisionTree(train_data)

    dt.make_tree()

    coll_num = 0
    for d in test_data :
        prediction = dt.determine(d['x'])
        if prediction == d['y'] :
            coll_num += 1

    print('accuracy:', float(coll_num) / len(test_data))