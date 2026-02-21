import pygame
import sys
import time

# 初始化
pygame.init()
WIDTH, HEIGHT = 800, 900
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)  # 可调整窗口，按 F11 切全屏
pygame.display.set_caption("中国象棋 - 带 AI 延迟")

# 全屏控制
fullscreen = False

# 颜色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (230, 40, 40)
BLUE = (60, 120, 240)
WOOD = (210, 180, 140)
LINE_COLOR = (0, 0, 0)
SELECTED_COLOR = (255, 255, 0)

# 棋盘参数
BOARD_ROWS = 10
BOARD_COLS = 9
MARGIN = 50
SQUARE_SIZE = 70

# 字体
font = pygame.font.SysFont("SimHei", 28, bold=True)

# 初始棋子布局（红方在下，黑方在上）
initial_board = [
    ["黑車", "黑馬", "黑象", "黑士", "黑將", "黑士", "黑象", "黑馬", "黑車"],
    [None, None, None, None, None, None, None, None, None],
    [None, "黑砲", None, None, None, None, None, "黑砲", None],
    ["黑卒", None, "黑卒", None, "黑卒", None, "黑卒", None, "黑卒"],
    [None, None, None, None, None, None, None, None, None],
    [None, None, None, None, None, None, None, None, None],
    ["红兵", None, "红兵", None, "红兵", None, "红兵", None, "红兵"],
    [None, "红炮", None, None, None, None, None, "红炮", None],
    [None, None, None, None, None, None, None, None, None],
    ["红车", "红马", "红相", "红仕", "红帅", "红仕", "红相", "红马", "红车"]
]

class ChessGame:
    def __init__(self):
        self.board = [row[:] for row in initial_board]  # 深拷贝
        self.selected = None  # (row, col)
        self.turn = "红"      # 红方先走
        self.ai_pending = False
        self.ai_move = None
        self.ai_schedule_time = 0
        self.AI_DELAY_MS = 1500  # 1.5秒延迟

    def draw_board(self, screen):
        w, h = screen.get_size()
        board_w = BOARD_COLS * SQUARE_SIZE
        board_h = BOARD_ROWS * SQUARE_SIZE
        offset_x = (w - board_w) // 2
        offset_y = (h - board_h) // 2

        # 画棋盘背景
        pygame.draw.rect(screen, WOOD, (offset_x - 10, offset_y - 10, board_w + 20, board_h + 20))

        # 画线
        for i in range(BOARD_ROWS):
            pygame.draw.line(screen, LINE_COLOR, (offset_x, offset_y + i * SQUARE_SIZE),
                             (offset_x + board_w, offset_y + i * SQUARE_SIZE))
        for j in range(BOARD_COLS):
            pygame.draw.line(screen, LINE_COLOR, (offset_x + j * SQUARE_SIZE, offset_y),
                             (offset_x + j * SQUARE_SIZE, offset_y + board_h))

        # 画九宫格斜线
        # 上方（黑将）
        pygame.draw.line(screen, LINE_COLOR, (offset_x + 3 * SQUARE_SIZE, offset_y),
                         (offset_x + 5 * SQUARE_SIZE, offset_y + 2 * SQUARE_SIZE))
        pygame.draw.line(screen, LINE_COLOR, (offset_x + 5 * SQUARE_SIZE, offset_y),
                         (offset_x + 3 * SQUARE_SIZE, offset_y + 2 * SQUARE_SIZE))
        # 下方（红帅）
        pygame.draw.line(screen, LINE_COLOR, (offset_x + 3 * SQUARE_SIZE, offset_y + 7 * SQUARE_SIZE),
                         (offset_x + 5 * SQUARE_SIZE, offset_y + 9 * SQUARE_SIZE))
        pygame.draw.line(screen, LINE_COLOR, (offset_x + 5 * SQUARE_SIZE, offset_y + 7 * SQUARE_SIZE),
                         (offset_x + 3 * SQUARE_SIZE, offset_y + 9 * SQUARE_SIZE))

        # 画棋子
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                piece = self.board[row][col]
                if piece:
                    color = RED if piece.startswith("红") else BLUE
                    text = font.render(piece[1], True, color)
                    rect = text.get_rect(center=(offset_x + col * SQUARE_SIZE + SQUARE_SIZE // 2,
                                                 offset_y + row * SQUARE_SIZE + SQUARE_SIZE // 2))
                    # 选中高亮
                    if self.selected == (row, col):
                        pygame.draw.circle(screen, SELECTED_COLOR,
                                           (offset_x + col * SQUARE_SIZE + SQUARE_SIZE // 2,
                                            offset_y + row * SQUARE_SIZE + SQUARE_SIZE // 2),
                                           SQUARE_SIZE // 2 - 5, 3)
                    screen.blit(text, rect)

    def get_square_under_mouse(self, pos):
        w, h = screen.get_size()
        board_w = BOARD_COLS * SQUARE_SIZE
        board_h = BOARD_ROWS * SQUARE_SIZE
        offset_x = (w - board_w) // 2
        offset_y = (h - board_h) // 2

        x, y = pos
        if x < offset_x or x > offset_x + board_w or y < offset_y or y > offset_y + board_h:
            return None
        col = (x - offset_x) // SQUARE_SIZE
        row = (y - offset_y) // SQUARE_SIZE
        if 0 <= row < BOARD_ROWS and 0 <= col < BOARD_COLS:
            return (row, col)
        return None

    def is_valid_move(self, from_pos, to_pos):
        fr, fc = from_pos
        tr, tc = to_pos
        piece = self.board[fr][fc]
        target = self.board[tr][tc]

        if not piece:
            return False
        if target and ((piece.startswith("红") and target.startswith("红")) or
                       (piece.startswith("黑") and target.startswith("黑"))):
            return False  # 不能吃同色

        name = piece[1]
        side = "红" if piece.startswith("红") else "黑"

        dr, dc = tr - fr, tc - fc

        # 简化规则：只实现 将/帅 和 兵/卒 的基本移动（其他可扩展）
        if name == "帥" or name == "將":
            if abs(dr) + abs(dc) != 1:
                return False
            # 九宫格限制
            if side == "红":
                if not (7 <= tr <= 9 and 3 <= tc <= 5):
                    return False
            else:
                if not (0 <= tr <= 2 and 3 <= tc <= 5):
                    return False
            return True

        elif name == "兵" or name == "卒":
            if side == "红":
                if dr > 0: return False  # 红兵不能后退
                if dr == 0 and abs(dc) != 1: return False
                if dr == -1 and dc != 0: return False
                if fr >= 5 and dr == 0 and abs(dc) == 1:  # 过河可横走
                    return True
                if dr == -1 and dc == 0:
                    return True
            else:  # 黑卒
                if dr < 0: return False
                if dr == 0 and abs(dc) != 1: return False
                if dr == 1 and dc != 0: return False
                if fr <= 4 and dr == 0 and abs(dc) == 1:
                    return True
                if dr == 1 and dc == 0:
                    return True
            return False

        # 其他棋子：允许任意移动（简化！实际应加规则）
        return True

    def make_move(self, from_pos, to_pos):
        if self.is_valid_move(from_pos, to_pos):
            fr, fc = from_pos
            tr, tc = to_pos
            self.board[tr][tc] = self.board[fr][fc]
            self.board[fr][fc] = None
            self.turn = "黑" if self.turn == "红" else "红"
            return True
        return False

    def get_ai_move(self):
        # 极简AI：随机找一个红方棋子走一步（仅演示）
        import random
        pieces = []
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                p = self.board[r][c]
                if p and p.startswith("红"):
                    pieces.append((r, c))
        random.shuffle(pieces)
        for fr, fc in pieces:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    tr, tc = fr + dr, fc + dc
                    if 0 <= tr < BOARD_ROWS and 0 <= tc < BOARD_COLS:
                        if self.is_valid_move((fr, fc), (tr, tc)):
                            return (fr, fc), (tr, tc)
        return None

    def handle_click(self, pos):
        square = self.get_square_under_mouse(pos)
        if not square:
            return

        if self.selected is None:
            # 选择棋子
            r, c = square
            piece = self.board[r][c]
            if piece and piece.startswith(self.turn):
                self.selected = square
        else:
            # 尝试移动
            fr, fc = self.selected
            tr, tc = square
            if (fr, fc) == (tr, tc):
                self.selected = None  # 取消选择
            elif self.make_move(self.selected, square):
                self.selected = None
                # 如果是玩家（黑方）走完，轮到 AI（红方）
                if self.turn == "红":
                    self.ai_pending = True
                    self.ai_schedule_time = pygame.time.get_ticks()
                    self.ai_move = self.get_ai_move()
            else:
                # 无效移动，可能重新选子
                r, c = square
                piece = self.board[r][c]
                if piece and piece.startswith(self.turn):
                    self.selected = square
                else:
                    self.selected = None

    def update_ai(self):
        if self.ai_pending and self.turn == "红":
            current = pygame.time.get_ticks()
            if current - self.ai_schedule_time >= self.AI_DELAY_MS:
                if self.ai_move:
                    (fr, fc), (tr, tc) = self.ai_move
                    self.make_move((fr, fc), (tr, tc))
                self.ai_pending = False
                self.ai_move = None


# 主程序
def main():
    global screen
    global fullscreen
    game = ChessGame()
    clock = pygame.time.Clock()

    while True:
        screen.fill(WHITE)
        game.draw_board(screen)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F11:
                    fullscreen = not fullscreen
                    if fullscreen:
                        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                    else:
                        screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1 and not game.ai_pending and game.turn == "黑":
                    game.handle_click(event.pos)

        game.update_ai()
        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()