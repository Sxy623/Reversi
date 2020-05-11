from func_timeout import func_timeout, FunctionTimedOut
from copy import deepcopy
import random
import datetime
import math

class AIPlayer:
    """
    AI 玩家
    """

    def __init__(self, color):
        """
        玩家初始化
        :param color: 下棋方，'X' - 黑棋，'O' - 白棋
        """

        self.color = color
        self.op_color = 'O' if color == 'X' else 'X'
        self.corner_score = 30 # 每个角的分值
        self.c = 2 # UCB 中的权重
        self.policy = 'Monte Carlo' # 'Minimax', 'Alpha-Beta'， ’Monte Carlo‘

    def get_move(self, board):
        """
        根据当前棋盘状态获取最佳落子位置
        :param board: 棋盘
        :return: action 最佳落子位置, e.g. 'A1'
        """
        if self.color == 'X':
            player_name = '黑棋'
        else:
            player_name = '白棋'
        print("请等一会，对方 {}-{} 正在思考中...".format(player_name, self.color))

        # -----------------请实现你的算法代码--------------------------------------

        if self.policy == 'Minimax':
            action, _ = self.max_min(board, 5)
        elif self.policy == 'Alpha-Beta':
            action, _ = self.alpha_beta(board, 5, float("-inf"), float("inf"), True)
        else:
            action = self.monte_carlo(board, 10)
        
        # ------------------------------------------------------------------------

        return action
    
    # 评估函数，得分越高对自己越有利
    def evaluate(self, board):
        score = board.count(self.color)
        # 检查四个角
        rows = [0, 7]
        cols = [0, 7]
        for row in rows:
            for col in cols:
                if board._board[row][col] == self.color:
                    score += self.corner_score
                elif board._board[row][col] == self.op_color:
                    score -= self.corner_score
        return score
    
    # 最小最大搜索——极大节点（自己）
    def max_min(self, board, depth):
        # 搜索到叶子节点
        if depth == 0:
            return None, self.evaluate(board)
        else:
            action_list = list(board.get_legal_actions(self.color))
            # 无子可走的情况
            if len(action_list) == 0:
                return None, self.evaluate(board)
            max_score = float('-inf') # 最大值
            max_action = None
            for action in action_list:
                flipped_pos = board._move(action, self.color)
                next_action, score = self.min_max(board, depth-1)
                # 更新最大值
                if score > max_score:
                    max_score = score
                    max_action = action
                # 回溯
                board.backpropagation(action, flipped_pos, self.color)
            return max_action, max_score
        
    # 最小最大搜索——极小节点（对手）
    def min_max(self, board, depth):
        # 搜索到叶子节点
        if depth == 0:
            return None, self.evaluate(board)
        else:
            action_list = list(board.get_legal_actions(self.op_color))
            # 无子可走的情况
            if len(action_list) == 0:
                return None, self.evaluate(board)
            min_score = float('inf') # 最小值
            min_action = None
            for action in action_list:
                flipped_pos = board._move(action, self.op_color)
                next_action, score = self.max_min(board, depth-1)
                # 更新最小值
                if score < min_score:
                    min_score = score
                    min_action = action
                # 回溯
                board.backpropagation(action, flipped_pos, self.op_color)
            return min_action, min_score
        
    # Alpha-Beta 剪枝搜索
    def alpha_beta(self, board, depth, alpha, beta, is_max_player):
        # 搜索到叶子节点
        if depth == 0:
            return None, self.evaluate(board)
        else:
            # 极大节点
            if is_max_player:
                action_list = list(board.get_legal_actions(self.color))
                # 无子可走的情况
                if len(action_list) == 0:
                    return None, self.evaluate(board)
                max_action = None
                for action in action_list:
                    flipped_pos = board._move(action, self.color)
                    next_action, score = self.alpha_beta(board, depth-1, alpha, beta, not is_max_player)
                    if score > alpha:
                        alpha = score
                        max_action = action
                    # 回溯
                    board.backpropagation(action, flipped_pos, self.color)
                    # 剪枝
                    if alpha >= beta:
                        break
                return max_action, alpha
            # 极小节点
            else:
                action_list = list(board.get_legal_actions(self.op_color))
                # 无子可走的情况
                if len(action_list) == 0:
                    return None, self.evaluate(board)
                min_action = None
                for action in action_list:
                    flipped_pos = board._move(action, self.op_color)
                    next_action, score = self.alpha_beta(board, depth-1, alpha, beta, not is_max_player)
                    if score < beta:
                        beta = score
                        min_action = action
                    # 回溯
                    board.backpropagation(action, flipped_pos, self.op_color)
                    # 剪枝
                    if alpha >= beta:
                        break
                return min_action, beta
    
    # Monte Carlo 树节点
    class Node:
        
         def __init__(self, color):
            self.childs = {} # 子节点
            self.color = color # 待落子一方的颜色
            self.n = 0 # 总访问次数
            self.t = 0 # 总获胜次数
            self.is_leaf = True # 是否为叶子节点
    
    # Monte Carlo 树搜索
    def monte_carlo(self, board, time):
        # 记录开始时间
        start_time = datetime.datetime.now()
        # 根节点
        root = self.Node(self.color)
        action_list = list(board.get_legal_actions(self.color))
        # 无子可走的情况
        if len(action_list) == 0:
            return None
        # 扩展根节点
        for action in action_list:
            root.childs[action] = self.Node(self.op_color)
        root.is_leaf = False
        # 迭代直到超时
        count = 0
        current_time = datetime.datetime.now()
        while (current_time - start_time).seconds < time:
            self.iter(board, root, count)
            count += 1
            current_time = datetime.datetime.now()
        # 返回访问次数最多的结点
        best_action = None
        max_n = 0
        for action in action_list:
            print(action, ' ', root.childs[action].t, '/', root.childs[action].n)
            if root.childs[action].n > max_n:
                max_n = root.childs[action].n
                best_action = action
        return best_action
    
    # Monte Carlo 树迭代
    def iter(self, board, node, count):
        # 对手颜色
        op_color = 'O' if node.color == 'X' else 'X'
        # 非叶子节点，计算UCB向下搜索
        if node.is_leaf == False:
            best_action = None
            best_node = None
            max_ucb = float('-inf')
            # 遍历子节点
            for action, child_node in node.childs.items():
                ucb = self.cal_ucb(child_node, count)
                # 找到了更大的 UCB
                if ucb > max_ucb:
                    max_ucb = ucb
                    best_action = action
                    best_node = child_node
            # 落子
            flipped_pos = board._move(best_action, node.color)
            # 继续搜索
            result = self.iter(board, best_node, count)
            # 回溯
            board.backpropagation(best_action, flipped_pos, node.color)
            # 更新非叶子节点的统计信息
            if result == True:
                node.t += 1
            node.n += 1
            return result
        # 叶子节点
        else:
            # 判断是否被访问，若是则对节点展开，否则开始 Rollout
            if node.n != 0:
                action_list = list(board.get_legal_actions(node.color))
                # 无子可走
                if len(action_list) == 0:
                    # 如果对面可以走，让对面走
                    action_list = list(board.get_legal_actions(op_color))
                    if len(action_list) != 0:
                        node.color = op_color
                        self.iter(board, node, count)
                    # 对面也走不了，游戏结束
                    else:
                        result = board.count(self.color) > board.count(self.op_color)
                        if result == True:
                            node.t += 1
                        node.n += 1
                        return result
                # 展开未访问节点
                else:
                    for action in action_list:
                        node.childs[action] = self.Node(op_color)
                    node.is_leaf = False
                    # 继续搜索
                    self.iter(board, node, count)
            else:
                result = self.rollout(deepcopy(board), node)
                # 更新叶子节点的统计信息
                if result == True:
                    node.t += 1
                node.n += 1
                return result
    
    # 计算 UCB
    def cal_ucb(self, node, count):
        # 从未被访问过的节点
        if node.n == 0:
            return float('inf')
        else:
            return node.t / node.n + self.c * math.sqrt(math.log(count) / node.n)
        
    # 从当前节点随机进行到游戏结束
    def rollout(self, board, node):
        # 对手颜色
        op_color = 'O' if node.color == 'X' else 'X'
        action_list1 = list(board.get_legal_actions(node.color))
        action_list2 = list(board.get_legal_actions(op_color))
        # 只要还有一方能落子，就继续进行
        while len(action_list1) != 0 or len(action_list2) != 0:
            if len(action_list1) != 0:
                action1 = random.choice(action_list1)
                board._move(action1, node.color)
                action_list1 = list(board.get_legal_actions(node.color))
                action_list2 = list(board.get_legal_actions(op_color))
            if len(action_list2) != 0:
                action2 = random.choice(action_list2)
                board._move(action2, op_color)
                action_list1 = list(board.get_legal_actions(node.color))
                action_list2 = list(board.get_legal_actions(op_color))
        # 判断胜者
        winner, _ = board.get_winner()
        if winner == 0 and self.color == 'X':
            return True
        elif winner == 1 and self.color == 'O':
            return True
        else:
            return False
        