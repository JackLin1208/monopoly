import matplotlib.pyplot as plt
from collections import Counter
import re
import random
import itertools
import numpy as np
import pandas as pd
from IPython.display import display
from tabulate import tabulate

players = ["player1", "player2", "player3"]

# 使用 permutations 生成所有排列組合
players_permutations = list(itertools.permutations(players))
# 打印結果
# for perm in players_permutations:
#     print(perm)

factors = {'sequence':[], 'high_land_num':[], 'conti_land_num':[], "land_holding":[], "ranking":[]
}

DF = pd.DataFrame(factors)

class Monopoly():
    # 參數 : 玩家數量, 每位玩家初始金額, 地圖上土地數量, 每塊土地價格(list), 過路費倍率, 升級費倍率   投擲骰子(前進)之順序
    def __init__(self, numberOfPlayer, money, numberOfLand, PriceOfLand, Fee_Weight, Upgrade_Weight, StartSequence):

        self.numberOfPlayer = numberOfPlayer

        self.StartSequence = StartSequence #再用另一個def設計

        print("玩家投擲順序: ", self.StartSequence)

        self.numberOfLand = numberOfLand # 土地數量

        # 依據玩家數量生成一個串列紀錄, eg. [player1, player2, ....]
        players = []
        for i in range(1, numberOfPlayer + 1):
            player = f'player{i}'
            players.append(player)
        self.Players = players 
        print("所有玩家: ", self.Players)
       
        # 玩家金錢紀錄, eg. [3000, 3000, 3000, ....]
        self.Money = [int(money) for i in range(numberOfPlayer)]
        print("每位玩家初始金錢: ", self.Money)
            
        # 玩家土地持有紀錄, eg. [[2,7,5], [6,1,9], [3,4], ....]
        self.players_holding = [[] for i in range(numberOfPlayer)]
        print("每位玩家土地持有紀錄: ", self.players_holding)

        # 土地持有狀況 [p2, p3 , p1 , ...], 未被購入初始值為0，被購入則變為該買家之編號
        self.land_status = [int(0) for i in range(self.numberOfLand)]
        print("每塊土地被持有狀況: ", self.land_status)

        # 土地價格 , eg. [500, 200, 700, ...]
        self.PriceOfLand = PriceOfLand
        print("每塊土地價格: ", self.PriceOfLand)

        # 土地等級
        self.Level = [int(0) for i in range(self.numberOfLand)]
        print("土地等級 : ", self.Level)

        # 過路費
        self.Pass_Fee = [int(0) for i in range(self.numberOfLand)]
        print("過路費 : ", self.Pass_Fee)

        # 土地過路費倍率
        self.Fee_W = Fee_Weight

        # 土地升及倍率
        self.Upgrade_W = Upgrade_Weight

        # 每位玩家在地圖上的位置，0為起點，固起始值為-1
        self.Position = [int(-1) for i in range(numberOfPlayer)]
        print("每位玩家目前位置: ", self.Position)

    def Parameter(self):
        # 用於紀錄該回合玩家淘汰之回合，若"NaN"代表獲勝
        self.Losing_Round = ["NaN" for i in range(self.numberOfPlayer)]
        # 玩家資訊
        self.players_info = {'money':self.Money, 'land':self.players_holding, "location":self.Position}
        self.players_index_labels = ["player1", "player2", "player3"]
        self.DF_players_info = pd.DataFrame(self.players_info, index = self.players_index_labels)
        print(self.DF_players_info)

        # show玩家資訊
        print(tabulate(self.DF_players_info, headers = 'keys', tablefmt = 'psql'))
        # print(self.DF_players_info.to_markdown())

        # 地圖資訊
        self.lands_info = {'owner':self.land_status, 'price':self.PriceOfLand, "level":self.Level, "fee":self.Pass_Fee}
        # self._lands_index_labels = ["player1", "player2", "player3"]
        self.DF_lands_info = pd.DataFrame(self.lands_info, index = range(0, self.numberOfLand))
        print(self.DF_lands_info)

        # sohw土地資訊
        print(tabulate(self.DF_lands_info, headers = 'keys', tablefmt = 'psql'))
        # print(self.DF_lands_info.to_markdown()

        

    def result(self):
        print(f"各玩家淘汰回合 : {self.Losing_Round}")
         # 挑出該回合蛋黃區之擁有玩家
        self.Yolk_Area_holding = [self.land_status[i] for i in range(len(self.land_status)) if (i+1)%3 == 0]
        # 該回合玩家持有土地數
        self.num_holding = [len(x) for x in self.players_holding]
        # 該回合持有最多土地之玩家
        temp = max(self.num_holding)
        self.most_holding_player = self.num_holding.index(temp) #最多土地玩家之index, player1 = 0
     

    def Playing(self):

        self.count = 0 # 計算回合數
        # 決定玩家投擲順序
        # eg. 順序為[player1, player3, player2, ....]
        pattern = re.compile(r'player(\d+)')
        self.Sequence = [int(pattern.search(i).group(1)) for i in self.StartSequence]
        print("投擲順序 : ", self.Sequence)
        # 即得到[1, 3, 2]

        # 用來計算名次
        self.rank = [0, 0, 0]
        last = 3

        # 若只剩剩一位玩家有錢(即該位玩家勝出)，則遊戲結束
        while sum(1 for x in self.Money if x > 0) != 1 :

            # 根據上面決定的順序擲骰子開始
            for i in self.Sequence:  
                if sum(1 for x in self.Money if x > 0) == 1 :  ## 若剩一個玩家即結束，不再進行投擲
                    break
                PlayerLabel = i
                i -= 1 # 假如該回合是玩家2，則他所有東西的位置會在1
                player_name = self.players_index_labels[i] #玩家在df裡之名稱
                self.count += 1 # 擲骰子，回合數+1
                print("以下為回合", self.count)
                # 兩顆骰子
                dice_1, dice_2 = random.randint(1, 6), random.randint(1, 6)
                step = dice_1 + dice_2 # 擲骰子前進之步數
                self.Position[i] += step
                if self.Position[i] > (self.numberOfLand - 1) : # 總共只有18格，若跑完整條則繼續從原點開始
                    self.Position[i] %= self.numberOfLand
                current = self.Position[i]  # 現在位置
                self.DF_players_info.at[player_name, 'location'] = current

                print(f"玩家{PlayerLabel}擲出", step, "前進到位置", self.Position[i])

                # 該土地沒人擁有
                if self.land_status[current] == 0:
                    if self.Money[i] >= self.PriceOfLand[current]:
                        self.Money[i] -= self.PriceOfLand[current] # 購入土地，扣該玩家錢
                        self.land_status[current] = PlayerLabel  # 被玩家i買下，土地狀態記為他的編號
                        self.Level[current] += 1        # 被購入，等級從0升為1
                        self.players_holding[i].append(current)
                        print(f"玩家{PlayerLabel}花", self.PriceOfLand[current], "購入土地" )
                        ## table紀錄
                        self.DF_players_info.at[player_name, 'money'] = self.Money[i]
                        self.DF_players_info.at[player_name, 'land'] = self.players_holding[i]
                        self.DF_lands_info.at[current, 'owner'] = PlayerLabel
                        ## 開始有過路費
                        self.Pass_Fee[current] = self.PriceOfLand[current] * self.Fee_W
                        self.DF_lands_info.at[current, 'fee'] = self.Pass_Fee[current]

                else:
                    # 土地是該玩家的
                    if self.land_status[current] == PlayerLabel: 
                        # 若此土地是你的且錢足夠升級，升級費為當前土地價格*升級倍率
                        Upgrade_Price = self.PriceOfLand[current] * (self.Upgrade_W)
                        if self.Money[i] >= Upgrade_Price :     #還有錢升級
                            self.Money[i] -= Upgrade_Price
                            self.Level[current] += 1 # 土地升級，等級+1
                            self.DF_lands_info.at[current, 'level'] = self.Level[current]

                            # 土地升級變貴
                            self.PriceOfLand[current] *= (1 + self.Upgrade_W) 
                            self.DF_lands_info.at[current, 'price'] = self.PriceOfLand[current]

                            ## 土地升級過路費也要升級(土地價格*fee權重)
                            self.Pass_Fee[current] = self.PriceOfLand[current] * self.Fee_W

                            self.DF_lands_info.at[current, 'fee'] = self.Pass_Fee[current]

                            self.DF_players_info.at[player_name, 'money'] = self.Money[i]
                            print(f"玩家{PlayerLabel}花", Upgrade_Price, "升級他的土地", current)
                            
                         ## table紀錄
                    # 土地是別人的要付過路費    
                    else: 
                        Fee = self.Pass_Fee[current]
                        # Fee = self.PriceOfLand[current] * self.Fee_W
                        self.Money[i] -= Fee
                        if self.Money[i] >= 0: # 付完還有錢
                            print(f"玩家{PlayerLabel}花", Fee, "付過路費，現在他剩", self.Money[i], "元")

                            print(self.Money)

                            self.DF_players_info.at[player_name, 'money'] = self.Money[i]

                    
                        else: # 付完<0 破產
                            self.Money[i] += Fee 

                            print(f"玩家{PlayerLabel}剩", self.Money[i], "不足以支付", Fee, "元的過路費，只好給土地擁有者剩餘的錢")

                            Fee = self.Money[i]  # 因為他沒錢了，土地擁有者只能拿到該玩家剩餘的錢
                           
                            self.Money[i] = 0 # 沒錢了歸0
                            self.DF_players_info.at[player_name, 'money'] = 0

                            if self.Money[i] <= 0:
                                print(f"Player{PlayerLabel} ", "is broken", "他在第", self.count, "回合出局")
                                self.Sequence.remove(PlayerLabel)  # 將該玩家移出投擲順序
                                self.rank[i] = last
                                last -= 1
                                self.Losing_Round[i] = self.count # 紀錄該玩家第幾回合淘汰
                                

                        # 看這塊current是誰的
                        current_holding = self.land_status[current]
                        current_holding_index = current_holding - 1
                        recieve_player = self.players_index_labels[current_holding_index] # 收錢人之id

                        # for j , player in enumerate(self.players_holding):
                        #     if current in player:
                                # current_holding = j

                        if self.Money[current_holding_index ] != 0:  # 該玩家若沒錢則代表已淘汰，不能再收取過路費
                            self.Money[current_holding_index] += Fee

                            self.DF_players_info.at[recieve_player, 'money'] = self.Money[current_holding_index]

                            print(f"玩家{(current_holding )}從玩家{PlayerLabel}那裡收取{Fee}元" )
                            
                # print(self.DF_lands_info)
                # print(self.DF_players_info)
                # print("剩餘玩家 : ", self.Sequence)
    
                # print("當前每位玩家持有金錢: ", self.Money)
                # print("每塊土地被持有狀況: ", self.land_status)
                # print("當前每位玩家土地持有紀錄: ", self.players_holding)
                # print("當前每塊土地價格: ", self.PriceOfLand)
                # print("當前每塊土地等級", self.Level)
                # print("每位玩家目前位置: ", self.Position)
        self.rank = [ 1 if x == 0 else x for x in self.rank]
        self.winner = self.Sequence[0] # 僅存玩家為獲勝者
        print(f"Game over , winner is player{self.winner}")
        self.winner_hold_land = self.players_holding[self.winner - 1]
        print("在第", self.count, "回合所有玩家皆破產，遊戲結束")
        # return winner


def simulating(times, numberOfplayer, DF):
    player_land_hold_most = [0, 0, 0] #記錄每回合玩家土地持有數，該回合最高者+1
    # A_all_Win_freq_per = []
    # B_all_Win_freq_per = []
    # C_all_Win_freq_per = []
    all_winner_hold_land = [[] for i in range(3)]  ## 贏家擁有土地之紀錄
    conti_num_land = [0, 0, 0]
    winner_conti_num_land = [[] for i in range(3)]
    
    for p in range(0, len(players_permutations)-5):
    # for p in range(len(players_permutations)):
        Losing_Round_Count = [[] for i in range(numberOfplayer)]
        Winner_Round_Count = [[] for i in range(numberOfplayer)]
        Yolk_hoding_count = []
        Yolk_ornot = [0, 0, 0]

        for i in range(1, times + 1):

            ## 3之倍數土地為高級區
            # Game = Monopoly(3, 3000, 18, [int(500) if (x%3) == 0 else int(300) for x in range(1, 19)], 0.3, 0.3, 
            #                 players_permutations[p])
            Game = Monopoly(3, 3000, 15, [int(500) if (x%3) == 0 else int(300) for x in range(1, 16)], 0.3, 0.3, 
                            players_permutations[p])
            Game.Parameter()
            Game.Playing()
            Game.result()
            # print(f"第{i}次模擬結束")

        # 淘汰及獲勝之回合數分布圖表
            for j in range(Game.numberOfPlayer):
                if Game.Losing_Round[j] != "NaN":#紀錄每次模擬輸家在第幾回合淘汰
                    Losing_Round_Count[j].append(Game.Losing_Round[j])
                else:                               # 紀錄每次模擬贏家在第幾回合勝出
                    Winner_Round_Count[j].append(Game.count)
            
            ## 每回合紀錄各玩家之蛋黃區持有狀況，若持有一塊編號出現一次，最後計算每位玩家所有回合持有數
            # 紀錄每位玩家每次遊戲擁有之蛋黃
            [Yolk_hoding_count.append(i) for i in Game.Yolk_Area_holding if  i != 0]
            Yolk_hoding_count_this_round = [i for i in Game.Yolk_Area_holding if  i != 0]
            ## 判斷該玩家此次遊戲是否擁有蛋黃區
            for x in [0, 1, 2]:
                if (x+1) in Game.Yolk_Area_holding:
                    Yolk_ornot[x] += 1

            # 計算該次模擬土地持有數最高之玩家
            player_land_hold_most[Game.most_holding_player] += 1
            ## 所有回合玩家土地計算
            all_winner_hold_land[Game.winner - 1 ].extend(Game.players_holding[Game.winner - 1])
            # 紀錄最長連續土地數
            this_round_conti = [longest_consecutive_sequence(i) for i in Game.players_holding]
            conti_num_land  = [(conti_num_land[i] + this_round_conti[i]) for i in range(3)]
            # 紀錄winner時最長連續土地數
            winner_conti_num_land[Game.winner - 1].append(longest_consecutive_sequence(Game.players_holding[Game.winner - 1]))

       
            
        ## DATAFRAME
            for i in range(3):
                Yolk_hoding_count_this_round_count = Counter(Yolk_hoding_count_this_round)
                high_land_num_i = Yolk_hoding_count_this_round_count[i+1]
                # new_row = pd.DataFrame({'sequence':Game.Sequence[i], 'high_land_num':high_land_num_i, 'conti_land_num':this_round_conti[i], 
                #                      "land_holding": one_hot_encode(Game.players_holding[i], Game.numberOfLand)
                # }) 
                print("資料室:", [i+1, high_land_num_i, this_round_conti[i], 
                                one_hot_encode(Game.players_holding[i], Game.numberOfLand)])
                DF.loc[len(DF)] = [i+1, high_land_num_i, this_round_conti[i], 
                                one_hot_encode(Game.players_holding[i], Game.numberOfLand), Game.rank[i]]
        
            # DF = pd.concat([DF, new_row], ignore_index=True)
        print("名次", Game.rank)
        print("連續地: ", conti_num_land)
#---------------------------------------------------
# -
        total_Losing_Round_Count = [] # 所有玩家淘汰之回合紀錄
        color = ['lightcoral', 'lightblue' , 'lightgreen']
        for i in range(Game.numberOfPlayer):
            Player_Label = i + 1
            # 該玩家淘汰回合紀錄長條圖
            print(f"玩家{Player_Label}之結束輪次:")       
            Round_Count = Counter(Losing_Round_Count[i])
            x = list(int(x) for x in Round_Count.keys())
            y = list(int(y) for y in Round_Count.values())
            # print(x)
            # print(y)
            plt.grid(linestyle = '--' )
            plt.bar(x, y, color = color[i],  zorder=3)
            plt.title(f'Losing roubd of the player{Player_Label}')
            plt.xlabel('Losing Round')
            plt.ylabel('Times')
            plt.show()
            total_Losing_Round_Count.extend(Losing_Round_Count[i])

        ## 所有玩家淘汰回合紀錄長條圖
        Round_Count = Counter(total_Losing_Round_Count)
        x = list(int(x) for x in Round_Count.keys())
        y = list(int(y) for y in Round_Count.values())
        # print(x)
        # print(y)
        plt.grid(linestyle = '--' )
        plt.bar(x, y, color = 'darkorange',  zorder=3)
        plt.title('Losing roubd of all the player')
        plt.xlabel('Losing Round')
        plt.ylabel('Times')
        plt.show()
    #--------------------------------------------------------
        # 繪製ABC當win時之最長連續土地計數
        color = ['lightcoral', 'lightblue' , 'lightgreen']
        for i in range(3):
            counted_numbers = Counter(winner_conti_num_land[i])
            # 拆解成標籤和計數
            labels, counts = zip(*counted_numbers.items())
            # 繪製長條圖
            plt.grid(linestyle = '--' )
            plt.bar(labels, counts, color=color[i],  zorder=3)
            # 添加標題和軸標籤
            plt.xticks([int(x) for x in labels])
            plt.xlabel('number of conti land ')
            plt.ylabel('times')
            plt.title(f'player{i+1} number of conti land when winning')
            plt.grid()
            # 顯示圖表
            plt.show()
#---------------------------------------------------------------------
        # 蛋黃區統計
        yolk_count = Counter(Yolk_hoding_count)
        print("蛋黃:", yolk_count)
        x_yolk = list(yolk_count.keys())
        y_yolk = list(yolk_count.values())
        # plt.grid(linestyle = '--' )
        # plt.bar(x_yolk, y_yolk, color = ['lightcoral', 'lightblue' , 'lightgreen'], zorder=3)
        # plt.xticks([1, 2, 3])
        # plt.xlabel('players')
        # plt.ylabel('yolk_holding_count')
        # plt.title('Frequency of yolk holding')
        # # plt.show()
        # 是否擁有蛋黃統計
        # print("是否有蛋黃區: ", Yolk_ornot)
        # # x_ornot = ['1', '2', '3']
        # y_ornot = Yolk_ornot
        # # plt.bar(x_ornot, y_ornot, color = ['lightcoral', 'lightblue' , 'lightgreen'])
        # # plt.title('yolk_ornot')
        # # plt.show()
        # print(players_permutations[p])
        # # 獲勝淘汰統計
        # # print("淘汰回合總計:", Losing_Round_Count)
        # print("獲勝回合總計:", Winner_Round_Count)
        Win_freq_per = [len(Winner_Round_Count[0]), len(Winner_Round_Count[1]), len(Winner_Round_Count[2])]
        print("獲勝次數: ", Win_freq_per)
        # 每次模擬ABC之獲勝次數統計，1*6，6種組合，計算每種組合之獲勝數
        # A_all_Win_freq_per.append(Win_freq_per[0])
        # B_all_Win_freq_per.append(Win_freq_per[1])
        # C_all_Win_freq_per.append(Win_freq_per[2])
        # print(Win_freq_per)
        A_all_Win_freq_per = (Win_freq_per[0])
        B_all_Win_freq_per = (Win_freq_per[1])
        C_all_Win_freq_per = (Win_freq_per[2])
     # -------------------------------------------------------------------   
        # 繪製土地持有因子長條圖
        winner_x = np.arange(1, 4)
        winner_y = [len(Winner_Round_Count[i]) for i in range(len(Winner_Round_Count))]
        bar_width = 0.2
        r1 = winner_x - bar_width # for number of most lands times
        r2 = winner_x
        r3 = winner_x + bar_width
        plt.figure(figsize=(20, 10))
        plt.grid(linestyle = '--' )
        # 土地持有最多之比例
        print("持有土地most: ", player_land_hold_most)
        ratio_player_land_hold_most = [i/times for i in player_land_hold_most]
        # ABC獲勝場次平均最高土地連續數
        conti_num_land_ave = [k/times for i, k in enumerate(conti_num_land)]
        ratio_conti_num_land =[ i / sum(conti_num_land_ave) for i in conti_num_land_ave]
        # 蛋黃數ratio 
        ratio_yolk = [ i / sum(y_yolk) for i in y_yolk]
        plt.bar(r1, ratio_player_land_hold_most, color = ['rosybrown', 'lightsteelblue' , 'lightgreen'], width=bar_width, label = [f'ratio of the most holding {i+1}' for i in range(3)],  zorder=3)
        plt.bar(r2, ratio_yolk, color = ['brown', 'royalblue' , 'forestgreen'], width=bar_width, label = [f'ratio of the high price land holding{i+1}' for i in range(3)],  zorder=3)
        plt.bar(r3, ratio_conti_num_land, color = ['red', 'navy' , 'darkgreen'], width=bar_width, label = [f'conti_ratio{i+1}' for i in range(3)],  zorder=3)
        plt.xticks([1, 2, 3])
        plt.xlabel('player')
        plt.ylabel("ratio")
        plt.title('statistic of player land holding')
        plt.legend()
        plt.show()
#------------------------------------------------------------------------
        ## 勝率圓餅圖
        # 計算每個玩家的總獲勝回合
        total_wins = sum(winner_y)
        # 計算每個玩家的勝率百分比
        win_percentage = [(wins / total_wins) * 100 for wins in winner_y]
        # 設定玩家標籤
        labels = ['player1', 'player2', 'player3']
        # 畫圓餅圖
        plt.figure(figsize=(7, 7))  # 設定圖形大小
        plt.pie(win_percentage, labels=labels, autopct='%1.1f%%', startangle=90, colors = ['lightcoral', 'lightblue' , 'lightgreen'])
        plt.title('winning rate')  # 添加標題
        plt.axis('equal')  # 確保餅圖為正圓形
        plt.show()
#-----------------------------------------------------------------------------------
# 每位玩家獲勝持有土地統計
    color = ['lightcoral', 'lightblue' , 'lightgreen']
    for i in range(3):
        counted_numbers = Counter(all_winner_hold_land[i])
        # 拆解成標籤和計數
        labels, counts = zip(*counted_numbers.items())
        # 繪製長條圖
        plt.grid(linestyle = '--' )
        plt.bar(labels, counts, color=color[i],  zorder=3)
        # 添加標題和軸標籤
        plt.xlabel('land')
        plt.ylabel('Cumulative times')
        plt.title(f'player{i+1} number of land holding when winning')
        # 顯示圖表
        plt.show()
    print(DF)
# --------------------------------------------------------------------------------------------
    return DF
    
# G1 = Monopoly(3, 3000, 18, [int(300) for x in range(18)], 0.3, 0.3, ["player1", "player2", "player3"])
# G1.Parameter()
# G1.Playing()
# G1.result()

# 計算最長連續子序列
def longest_consecutive_sequence(lst):
    # 將列表轉換為集合來去重並加速查找操作
    num_set = set(lst)
    longest_streak = 0

    # 遍歷列表中的每個數字
    for num in lst:
        # 只在當前數字的前一個數字不在集合中時開始計數，確保只從序列的起點計算
        if num - 1 not in num_set:
            current_num = num
            current_streak = 1

            # 連續數字檢查
            while current_num + 1 in num_set:
                current_num += 1
                current_streak += 1

            # 更新最長連續長度
            longest_streak = max(longest_streak, current_streak)

    return longest_streak

def one_hot_encode(lst, length):
    # 初始化长度为 length 的全 0 列表
    one_hot = [0] * length
    
    # 将出现在 lst 中的数字对应位置设置为 1
    for num in lst:
        if num < length:
            one_hot[num] = 1
            
    return one_hot

Df_out = simulating(3000, 3, DF)
Df_out.to_csv('output.csv')
