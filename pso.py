'''
author: longxin
date: 2019-11-03
descprtion:
version: 0.0
changedescrption:

'''

class PSO:

    def __init__(self, D, M, fit_func, V_max, Max_inter, Opt_bound):
        '''
        D: 数据维度
        M：粒子数量
        :param D:
        :param M:
        '''
        import numpy as np
        self.c1 = 2
        self.c2 = 2
        self.w = 1
        # self.episi = 0.2
        # self.eta = 0.8
        self.M = M
        self.D = D
        self.max_inter = Max_inter
        self.Opt_bound = Opt_bound
        self.fit_func = fit_func
        self.local_inter = 0

        # 进行粒子初始化
        self.X = np.random.choice([0, 1], (M, D)) # 粒子初始化的解
        self.old_X = np.copy(self.X) # 保存下一步的解
        self.pid_list = np.copy(self.X) # 粒子历史最优解
        self.vid_list = np.random.random((M, D))*V_max # 每个粒子的初始速度

        # 计算每个粒子的适应值, 然后获得领域最优的位置
        self.best_fitness = [fit_func(self.X[i]) for i in range(np.shape(self.X)[0])]
        self.pgd = np.copy(self.X[np.argmax(self.best_fitness)])


    def computer_next_v_id(self, vk_id, pk_id, pk_gd, xk_id):
        import random
        return vk_id + self.c1*random.random()*(pk_id-xk_id) + self.c2*random.random()*(pk_gd-xk_id)

    def computer_next_x_id(self, v_id_next):
        import math
        import random
        s_next = 1 / (1+math.exp(-v_id_next))
        if random.random() < s_next:
            return 1
        else:
            return 0

    def update(self, X):
        import numpy as np
        import math

        # 计算每个粒子的适应值
        fitness_list = [0 for i in range(self.M)]
        for i in range(self.M):
            fitness_list[i] = self.fit_func(X[i])

        # 根据适应值更新pid_list 和 pg
        for i in range(self.M):
            if self.best_fitness[i] < fitness_list[i]:
                self.pid_list[i] = np.copy(X[i])
                self.best_fitness[i] = fitness_list[i]

                if self.best_fitness[i] == max(self.best_fitness):
                    self.pgd = np.copy(X[i])
        # max_p = np.argmax(self.best_fitness)
        # self.pgd = np.copy(X[max_p])

        # 更新粒子速度
        for i in range(self.M):
            for d in range(self.D):
                self.vid_list[i][d] = self.computer_next_v_id(self.vid_list[i][d], self.pid_list[i][d], self.pgd[d], X[i][d])

        # 更行粒子位置
        self.old_X = np.copy(self.X)
        for i in range(self.M):
            for d in range(self.D):
                self.X[i][d] = self.computer_next_x_id(self.vid_list[i][d])

class package:

    def __init__(self):
        import numpy as np
        # 最大权重
        self.Max_weight = 100
        # 物品数量
        self.D = 20
        # 物品价值
        self.Value = np.random.random(self.D) * 20
        # 物品重量
        self.Weight = np.random.random(self.D) * (self.Max_weight/self.D) * 3

        # pso
        self.pso = None

        # 粒子数量
        self.p_count = 6

    def fit_func(self, x):
        import numpy as np
        tmpx = np.array(x)
        weigth = np.sum(tmpx*self.Weight)

        if weigth > self.Max_weight:
            return -1000
        else:
            return np.sum(tmpx*self.Value)


    def run(self):
        import numpy as np
        import matplotlib.pyplot as plt

        max_inter = 100
        self.pso = PSO(self.D, self.p_count, self.fit_func, 2, max_inter, 999)
        fit_list = []
        for inter in range(max_inter+1):
            self.pso.update(self.pso.X)
            fit_list.append(np.max(self.pso.best_fitness))
            print("Inter {0} Max value: {1} Max weight: {2}".format(inter+1, np.max(self.pso.best_fitness),
                                                                    np.sum(self.Weight*self.pso.pgd)))

        plt.figure()
        plt.plot([i for i in range(max_inter+1)], fit_list)
        plt.show()

        print("The best solutions is, ", self.pso.pgd)

    def test_run(self):
        import numpy as np
        import matplotlib.pyplot as plt
        import turtle
        import copy
        import time

        # turtle.Screen(800, 400, 0, 0)
        turtle.setup(800, 400)
        turtle.speed(0)
        turtle.delay(0)
        turtle.hideturtle()
        start_x = -380
        start_y = 180

        best_gd_start_place = [start_x, start_y]

        # 初始化粒子所在位置
        Sum_weights = np.sum(self.Weight)
        Sum_values = np.sum(self.Value)

        Max_p_width = (np.max(self.Weight) / Sum_weights) * Max_width + 10
        Max_p_height = (np.max(self.Value) / Sum_values) * Max_height + 10

        p_start_x = []
        p_start_y = []
        for i in range(self.p_count):
            p_start_x.append(start_x)
            p_start_y.append(start_y-(i+1)*(Max_p_height+20))

        max_inter = 100
        self.pso = PSO(self.D, self.p_count, self.fit_func, 2, max_inter, 999)
        fit_list = []

        # 对解进行初始化显示
        plot_x(self.Weight, self.Value, self.pso.pgd, best_gd_start_place, "全局最有解： ")
        for i in range(self.p_count):
            plot_x(self.Weight, self.Value, self.pso.X[i], [p_start_x[i], p_start_y[i]],
                   "粒子 {0} 当前解".format(i+1))
        plt.ion()
        # plt.figure(1)
        fig, ax = plt.subplots(1, 2)
        # fig1 = plt.figure(2)

        point_list = []
        anocation_list = []

        tmppoint, tmpanocation = plot_point(ax[1], 1, self.pso.pgd, None)
        point_list.append(tmppoint)
        # anocation_list.append(tmpanocation)
        for m in range(self.p_count):
            tmpoint, tmpanocation = plot_point(ax[1], 0, self.pso.X[m], self.pso.old_X[m])
            point_list.append(tmpoint)
            anocation_list.append(tmpanocation)

        # fig, ax = plt.subplots(figsize=(5, 3))
        for inter in range(max_inter+ 1):
            self.pso.update(self.pso.X)
            fit_list.append(np.max(self.pso.best_fitness))

            # 画出最优解的收敛性能
            ax[0].plot([i for i in range(inter+1)], fit_list,c='r',ls='-', marker='o', mec='b',mfc='w')
            ax[0].set_ylim(0, np.sum(self.Value))
            ax[0].set_xlabel("Interation Num")
            ax[0].set_ylabel("Local best solutions")

            # fig1.show()
            print("Inter {0} Max value: {1} Max weight: {2}".format(inter + 1, np.max(self.pso.best_fitness),
                                                                    np.sum(self.Weight * self.pso.pgd)))

            # 画出当前粒子所在的背包
            plot_x(self.Weight, self.Value, self.pso.pgd, best_gd_start_place, "全局最有解： ")
            for i in range(self.p_count):
                plot_x(self.Weight, self.Value, self.pso.X[i], [p_start_x[i], p_start_y[i]], "粒子 {0} 当前解".format(i+1))

            # 画出当前粒子的位置
            # ax[1].lines.remove(point_list[0])
            plt.cla()
            tmppoint, tmpanocation = plot_point(ax[1], 1, self.pso.pgd, None)
            point_list[0] = tmppoint
            anocation_list.append(tmpanocation)
            for m in range(self.p_count):
                # ax[1].lines.remove(point_list[m+1])
                # ax[1].anocations.remove(anocation_list[m+1])
                tmpoint, tmpanocation = plot_point(ax[1], 0, self.pso.X[m], self.pso.old_X[m])
                point_list[m+1] = tmpoint
                anocation_list[m+1] = tmpanocation

            fig.show()
            plt.pause(1)
            # fig.pause(0.1)
        # plt.figure()
        # plt.plot([i for i in range(max_inter + 1)], fit_list)
        # plt.show()

        print("The best solutions is, ", self.pso.pgd)

        plot_x(self.Weight, self.Value, self.pso.pgd, [0, 0])
        turtle.done()
def bool2int(x):
    y = 0
    for i,j in enumerate(x):
        y += j<<i
    return y

def plot_point(ax, color, x, x_next):
    import matplotlib.pyplot as plt
    import numpy as np

    half_long_x = int(np.shape(x)[0]/2)
    point_x_start = bool2int(x[:half_long_x])
    point_y_start = bool2int(x[half_long_x:])

    point = None
    anotion = None

    tmpyones = np.ones(shape=(int(np.shape(x)[0])-half_long_x), dtype=np.int).tolist()
    tmpxones = np.ones(shape=(half_long_x), dtype=np.int).tolist()
    if color == 1:
        point = ax.plot([point_x_start], [point_y_start], marker='o', color='red', markersize=10)
        ax.set_ylim(0, bool2int(tmpyones[:]))
        ax.set_xlim(0, bool2int(tmpxones[:]))

    else:
        point_x_next = bool2int(x_next[:half_long_x])
        point_y_next = bool2int(x_next[half_long_x:])
        point = ax.plot([point_x_next], [point_y_next], marker='o', color='blue')

        vec_dire = [0, 0]
        if point_x_start != point_x_next:
            vec_dire[0] = 2*(-point_x_start+point_x_next)/((point_x_start-point_x_next)**2+(point_y_start-point_y_next)**2)
        if point_y_next != point_y_start:
            vec_dire[1] = 2*(-point_y_start+point_y_next)/((point_x_start-point_x_next)**2+(point_y_start-point_y_next)**2)

        # vec_dire = [2*(point_x_start-point_x_next)/((point_x_start-point_x_next)**2+(point_y_start-point_y_next)**2),
        #             2*(point_y_start-point_y_next)/((point_x_start-point_x_next)**2+(point_y_start-point_y_next)**2)]

        anotion = ax.annotate('', xy=(point_x_next, point_y_next), xytext=(point_x_next+2*vec_dire[0], point_y_next+2*vec_dire[1]),
                    arrowprops=dict(facecolor='black', shrink=0.05))
        ax.set_ylim(0, bool2int(tmpyones[:]))
        ax.set_xlim(0, bool2int(tmpxones[:]))
    return point, anotion


def text(x, y, words, size):
    import turtle
    turtle.penup()
    turtle.goto(x, y)
    turtle.pendown()
    turtle.write(words, font=('微软雅黑', size, 'bold'))

def goods(x, y, color, width,  height):
    import turtle

    turtle.penup()
    turtle.goto(x, y)
    turtle.begin_fill()
    turtle.fillcolor(color)
    for i in range(4):
        if i%2 == 0:
            turtle.forward(width)
            turtle.right(90)
        else:
            turtle.forward(height)
            turtle.right(90)
    turtle.end_fill()


def bag(x,y):
    import turtle

    turtle.penup()
    turtle.goto(x,y)
    turtle.begin_fill()
    turtle.fillcolor("#FFEBCD")
    turtle.forward(400)
    turtle.right(90)
    turtle.forward(20)
    turtle.right(90)
    turtle.forward(400)
    turtle.right(90)
    turtle.forward(40)
    turtle.right(90)
    turtle.pendown()
    turtle.end_fill()


Max_width = 200
Max_height = 200

def plot_x(Weights, Values, X, X_place, forfix):
    import numpy as np

    Sum_weights = np.sum(Weights)
    Sum_values = np.sum(Values)

    Min_width = (np.max(Weights)/Sum_weights)*Max_width + 10
    Min_height = (np.max(Values)/Sum_values)*Max_height + 10

    for i in range(np.shape(X)[0]):
        x = X_place[0]+Min_width*i
        y = X_place[1]
        if X[i] == 0:
            goods(x, y, "LightSkyBlue", int((Weights[i]/Sum_weights)*Max_width),
                  int((Values[i]/Sum_values)*Max_height))
        else:
            goods(x, y, "LightPink", int((Weights[i] / Sum_weights) * Max_width),
                  int((Values[i] / Sum_values) * Max_height))

        if i == 0:
            bag(x, y - Min_height+10)
            ss = forfix + " Total value= " + str(int(np.sum(np.array(X) * np.array(Values)))) + " Total weigth= " + \
                 str(int(np.sum(np.array(X) * np.array(Weights))))
            # W =
            text(x+10, y - Min_height - 10, ss, 10)
            # text(x+10, y - Min_height - 30, V, 10)



def plot_gd(gd_place, gd, fitness_list):
    import numpy as np
    bag(gd_place[0], gd_place[1])
    text(gd_place[0])
def test():
    import turtle
    turtle.setup(800, 400, 0, 0)

    goods(0, 0, "LightPink", 100, 200)
    turtle.done()

if __name__ == "__main__":
    pk = package()
    # pk.run()
    pk.test_run()
    # test()
