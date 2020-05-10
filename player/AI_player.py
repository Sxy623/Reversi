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
        self.op_color = "O" if color == "X" else "X"

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

        action, _ = self.max_min(board, 3)
        
        # ------------------------------------------------------------------------

        return action
    
    # 评估函数，得分越高对自己越有利
    def evaluate(self, board):
        return board.count(self.color)
    
    # 极大节点（自己）
    def max_min(self, board, depth):
        if depth == 0:
            return None, self.evaluate(board)
        else:
            action_list = list(board.get_legal_actions(self.color))
            if len(action_list) == 0:
                return None, self.evaluate(board)
            max_score = -10000 # 最大值
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
        
    # 极小节点（对手）
    def min_max(self, board, depth):
        if depth == 0:
            return None, self.evaluate(board)
        else:
            action_list = list(board.get_legal_actions(self.op_color))
            if len(action_list) == 0:
                return None, self.evaluate(board)
            min_score = 10000 # 最小值
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
        