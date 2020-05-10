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
        self.corner_score = 30 # 每个角的分值
        self.policy = 'Alpha-Beta' # 'Minimax', 'Alpha-Beta'

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
        else:
            action, _ = self.alpha_beta(board, 5, float("-inf"), float("inf"), True)
        
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
            max_score = float("-inf") # 最大值
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
            min_score = float("inf") # 最小值
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
            