import CONFIG
import COUNT5
import PRIOR
import STATUS1
from AIHash import HashTable
# import numpy as np
# import numba as nb
# from numba import jit


class OXSpace:
    EMPTY = 2
    WRONG = 3


DX = [1, 0, 1, 1]
DY = [0, 1, 1, -1]

class OXPoint:

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

class OXMove:
    """
    class OXMove
    """

    def __init__(self, x=0, y=0, value=0):
        """
        @type self: OXMove
        @type x: np.uint8
        @type y: np.uint8
        @type value: np.int32
        """
        self.x = x
        self.y = y
        self.value = value

    def point(self):
        """
        @type: OXMove
        """
        return OXPoint(self.x, self.y)

class Cell:

    def __init__(self):
        self.piece = 0
        # self.pattern = np.zeros([4, 2], dtype=np.uint8)
        self.pattern = [[0, 0], [0, 0], [0, 0], [0, 0]]
        # self.shape1 = np.zeros([4, 2], dtype=np.uint8)
        self.shape1 = [[0, 0], [0, 0], [0, 0], [0, 0]]
        # self.shape4 = np.zeros(2, dtype=np.uint8)
        self.shape4 = [0, 0]
        # self.adj1 = np.uint8(0)
        self.adj1 = 0
        # self.adj2 = np.uint8(0)
        self.adj2 = 0

    def prior(self, PRIOR):
        return PRIOR[self.pattern[0][0]][self.pattern[0][1]] +\
            PRIOR[self.pattern[1][0]][self.pattern[1][1]] +\
            PRIOR[self.pattern[2][0]][self.pattern[2][1]] +\
            PRIOR[self.pattern[3][0]][self.pattern[3][1]] +\
            PRIOR[self.pattern[0][1]][self.pattern[0][0]] +\
            PRIOR[self.pattern[1][1]][self.pattern[1][0]] +\
            PRIOR[self.pattern[2][1]][self.pattern[2][0]] +\
            PRIOR[self.pattern[3][1]][self.pattern[3][0]] +\
            (self.adj1 != 0)

    def update4(self, STATUS4):
        self.shape4[0] = STATUS4[self.shape1[0][0]][self.shape1[1][0]][
            self.shape1[2][0]
        ][self.shape1[3][0]]
        self.shape4[1] = STATUS4[self.shape1[0][1]][self.shape1[1][1]][
            self.shape1[2][1]
        ][self.shape1[3][1]]

    def update1(self, k):
        self.shape1[k][0] = STATUS1.content[self.pattern[k][0]][self.pattern[k][1]]
        self.shape1[k][1] = STATUS1.content[self.pattern[k][1]][self.pattern[k][0]]


class AI:
    WIN_MIN = 25000
    WIN_MAX = 30000
    INF = 32000
    A = 8
    B = 7
    C = 6
    E = 4
    D = 5
    F = 3
    G = 2
    H = 1
    FORBID = 9

    def __init__(self):
        # self.STATUS4 = np.zeros([10, 10, 10, 10], dtype=np.uint8)
        self.STATUS4 = [[[[0 for _ in range(10)] for _ in range(10)] for _ in range(10)] for _ in range(10)]
        # self.RANK = np.zeros(107, dtype=np.uint8)
        self.RANK = [0 for _ in range(107)]
        # self.PRIOR = np.zeros([256, 256], dtype=np.uint8)
        self.PRIOR = [[0 for _ in range(256)] for _ in range(256)]
        self.space = [[Cell() for _ in range(30)] for _ in range(30)]
        self.OP = 0
        self.XP = 1
        for a in range(10):
            for b in range(10):
                for c in range(10):
                    for d in range(10):
                        self.STATUS4[a][b][c][d] = self.getStatus4(a, b, c, d)
        for a in range(107):
            self.RANK[a] = self.getRANK(a)
        for a in range(256):
            for b in range(256):
                self.PRIOR[a][b] = self.getPrior(a, b)
        self.remMove = [None for _ in range(400)]
        self.remCell = [None for _ in range(400)]
        self.remULCand = [None for _ in range(400)]
        self.remLRCand = [None for _ in range(400)]
        self.MAX_CAND = 30
        self.table = HashTable()
        self.premove = 0, 0

    def getStatus4(self, s0, s1, s2, s3):
        """
        @type self: AI
        @type s0: np.uint8
        @type s1: np.uint8
        @type s2: np.uint8
        @type s3: np.uint8
        """
        # n = np.zeros(10, dtype=np.uint8)
        n = [0 for _ in range(10)]
        n[s0] += 1
        n[s1] += 1
        n[s2] += 1
        n[s3] += 1

        if n[9] >= 1:
            return self.A
        if n[8] >= 1:
            return self.B
        if n[7] >= 2:
            return self.B
        if n[7] >= 1 and n[6] >= 1:
            return self.C
        if n[7] >= 1 and n[5] >= 1:
            return self.D
        if n[7] >= 1 and n[4] >= 1:
            return self.D
        if n[7] >= 1:
            return self.E
        if n[6] >= 2:
            return self.F
        if n[6] >= 1 and n[5] >= 1:
            return self.G
        if n[6] >= 1 and n[4] >= 1:
            return self.G
        if n[6] >= 1:
            return self.H
        return 0

    def getRANK(self, cfg):
        """
        @type cdg: np.uint8
        """
        content = COUNT5.content
        return content[cfg][4] * 19 + content[cfg][3] * 15 + \
            content[cfg][2] * 11 + content[cfg][1] * 7 + content[cfg][0] * 3

    def getPrior(self, a, b):
        """
        @type a: np.uint8
        @type b: np.uint8
        """
        return PRIOR.content[a][b]

    def evaluate(self):
        # p = np.zeros(2, dtype=np.int64)
        p = [0, 0]
        for i in range(self.remCount):
            c = self.remCell[i]
            a = c.piece
            for k in range(4):
                # print([c.pattern[k][a], c.pattern[k][1 - a]])
                # p[a] += self.RANK[CONFIG.content[c.pattern[k][a]][c.pattern[k][1 - a]]]
                p[a] += self.getRANK(CONFIG.content[c.pattern[k][a]][c.pattern[k][1 - a]])
        return p[self.who] - p[self.opp]

    def start(self, width, height):
        """
        @type self: AI
        @type width: np.uint8
        @type height: np.uint8
        """
        self.boardWidth = width
        self.boardHeight = height
        for y in range(height + 8):
            for x in range(width + 8):
                if (x < 4 or y < 4 or x >= width + 4 or y >= height + 4):
                    self.space[x][y].piece = OXSpace.WRONG
                else:
                    self.space[x][y].piece = OXSpace.EMPTY
                for k in range(4):
                    self.space[x][y].pattern[k][0] = self.space[x][y].pattern[k][
                        1
                    ] = 0
        # unit = np.uint8(1)
        unit = 1
        for y in range(4, height + 4):
            for x in range(4, width + 4):
                for k in range(0, 4):
                    xx = x - DX[k]
                    yy = y - DY[k]
                    # p = np.uint8(8)
                    p = 8
                    while p != 0:
                        if self.space[xx][yy].piece == OXSpace.WRONG:
                            self.space[x][y].pattern[k][0] |= p
                            self.space[x][y].pattern[k][1] |= p
                        xx -= DX[k]
                        yy -= DY[k]
                        p >>= unit
                    xx = x + DX[k]
                    yy = y + DY[k]
                    # p = np.uint8(16)
                    p = 16
                    while p != 0:
                        if self.space[xx][yy].piece == OXSpace.WRONG:
                            self.space[x][y].pattern[k][0] |= p
                            self.space[x][y].pattern[k][1] |= p
                        xx += DX[k]
                        yy += DY[k]
                        p <<= unit
                        p = p % 256
        for y in range(4, height + 4):
            for x in range(4, height + 4):
                self.space[x][y].update1(0)
                self.space[x][y].update1(1)
                self.space[x][y].update1(2)
                self.space[x][y].update1(3)
                self.space[x][y].update4(self.STATUS4)
                self.space[x][y].adj1 = self.space[x][y].adj2 = 0
        # self.CountShape4 = np.zeros((2, 10), dtype=np.int64)
        self.CountShape4 = [[0 for _ in range(10)] for _ in range(2)]
        self.totalSearched = 0
        self.who = self.OP
        self.opp = self.XP
        self.moveCount = self.remCount = 0
        self.upperLeftCand = OXPoint(99, 99)
        self.lowerRightCand = OXPoint(0, 0)
        self.nSearched = 0
        self.table.clear()
        return

    def move(self, x, y):
        """
        @type x: np.uint8
        @type y: np.uint8
        """
        self.premove = x, y
        self.table.clear()
        if self.space[x + 4][y + 4].piece == OXSpace.EMPTY:
            self._move(x + 4, y + 4)
        # else:
            # eprintf("wrong move")

    def _move(self, xp, yp):
        """
        @type xp: np.uint8
        @type yp: np.uint8
        """
        # print(["move", xp, yp, updateHash])
        # logline = ""
        # for m in range(2):
        #     for n in range(10):
        #         logline += (str(self.CountShape4[m][n]) + " ")
        # print(logline)
        self.nSearched += 1
        self.CountShape4[0][self.space[xp][yp].shape4[0]] -= 1
        self.CountShape4[1][self.space[xp][yp].shape4[1]] -= 1

        self.space[xp][yp].piece = self.who
        self.remCell[self.remCount] = self.space[xp][yp]
        self.remMove[self.moveCount] = OXPoint(xp, yp)
        self.remULCand[self.remCount] = self.upperLeftCand
        self.remLRCand[self.remCount] = self.lowerRightCand
        self.moveCount += 1
        self.remCount += 1

        if (xp - 2 < self.upperLeftCand.x):
            self.upperLeftCand.x = max(xp - 2, 4)
        if (yp - 2 < self.upperLeftCand.y):
            self.upperLeftCand.y = max(yp - 2, 4)
        if (xp + 2 > self.lowerRightCand.x):
            self.lowerRightCand.x = min(xp + 2, self.boardWidth + 3)
        if (yp + 2 > self.lowerRightCand.y):
            self.lowerRightCand.y = min(yp + 2, self.boardHeight + 3)
        # unit = np.uint8(1)
        unit = 1
        for k in range(4):
            x = xp
            y = yp
            # p = np.uint8(16)
            p = 16
            while p != 0:
                x -= DX[k]
                y -= DY[k]
                self.space[x][y].pattern[k][self.who] |= p
                if self.space[x][y].piece == OXSpace.EMPTY:
                    self.space[x][y].update1(k)
                    self.CountShape4[0][self.space[x][y].shape4[0]] -= 1
                    self.CountShape4[1][self.space[x][y].shape4[1]] -= 1
                    self.space[x][y].update4(self.STATUS4)
                    self.CountShape4[0][self.space[x][y].shape4[0]] += 1
                    self.CountShape4[1][self.space[x][y].shape4[1]] += 1
                p <<= unit
                p = p % 256
            x = xp
            y = yp
            # p = np.uint8(8)
            p = 8
            while p != 0:
                x += DX[k]
                y += DY[k]
                self.space[x][y].pattern[k][self.who] |= p
                if self.space[x][y].piece == OXSpace.EMPTY:
                    self.space[x][y].update1(k)
                    self.CountShape4[0][self.space[x][y].shape4[0]] -= 1
                    self.CountShape4[1][self.space[x][y].shape4[1]] -= 1
                    self.space[x][y].update4(self.STATUS4)
                    self.CountShape4[0][self.space[x][y].shape4[0]] += 1
                    self.CountShape4[1][self.space[x][y].shape4[1]] += 1
                p >>= unit

        self.space[xp - 1][yp - 1].adj1 += 1
        self.space[xp][yp - 1].adj1 += 1
        self.space[xp + 1][yp - 1].adj1 += 1
        self.space[xp - 1][yp].adj1 += 1
        self.space[xp + 1][yp].adj1 += 1
        self.space[xp - 1][yp + 1].adj1 += 1
        self.space[xp][yp + 1].adj1 += 1
        self.space[xp + 1][yp + 1].adj1 += 1
        self.space[xp - 2][yp - 2].adj2 += 1
        self.space[xp][yp - 2].adj2 += 1
        self.space[xp + 2][yp - 2].adj2 += 1
        self.space[xp - 2][yp].adj2 += 1
        self.space[xp + 2][yp].adj2 += 1
        self.space[xp - 2][yp + 2].adj2 += 1
        self.space[xp][yp + 2].adj2 += 1
        self.space[xp + 2][yp + 2].adj2 += 1

        # if updateHash:
        #     self.table.move(xp, yp, self.who)

        self.who = self.OPPNENT(self.who)
        self.opp = self.OPPNENT(self.opp)
        # logline = ""
        # for m in range(2):
        #     for n in range(10):
        #         logline += (str(self.CountShape4[m][n]) + " ")
        # print(logline)

    def block(self, x, y):
        xp = x + 4
        yp = y + 4

        self.CountShape4[0][self.space[xp][yp].shape4[0]] -= 1
        self.CountShape4[1][self.space[xp][yp].shape4[1]] -= 1

        self.space[xp][yp].piece = OXSpace.WRONG
        self.remMove[self.moveCount] = OXPoint(xp, yp)
        self.moveCount += 1

        # unit = np.uint8(1)
        unit = 1
        for k in range(4):
            x = xp
            y = yp
            # p = np.uint8(16)
            p = 16
            while p != 0:
                x -= DX[k]
                y -= DY[k]
                self.space[x][y].pattern[k][0] |= p
                self.space[x][y].pattern[k][1] |= p
                if self.space[x][y].piece == OXSpace.EMPTY:
                    self.space[x][y].update1(k)
                    self.CountShape4[0][self.space[x][y].shape4[0]] -= 1
                    self.CountShape4[1][self.space[x][y].shape4[1]] -= 1
                    self.space[x][y].update4(self.STATUS4)
                    self.CountShape4[0][self.space[x][y].shape4[0]] += 1
                    self.CountShape4[1][self.space[x][y].shape4[1]] += 1
                p <<= unit
                p = p % 256
            x = xp
            y = yp
            # p = np.uint8(8)
            p = 8
            while p != 0:
                x += DX[k]
                y += DY[k]
                self.space[x][y].pattern[k][0] |= p
                self.space[x][y].pattern[k][1] |= p
                if self.space[x][y].piece == OXSpace.EMPTY:
                    self.space[x][y].update1(k)
                    self.CountShape4[0][self.space[x][y].shape4[0]] -= 1
                    self.CountShape4[1][self.space[x][y].shape4[1]] -= 1
                    self.space[x][y].update4(self.STATUS4)
                    self.CountShape4[0][self.space[x][y].shape4[0]] += 1
                    self.CountShape4[1][self.space[x][y].shape4[1]] += 1
                p >>= unit

        self.who = self.OPPNENT(self.who)
        self.opp = self.OPPNENT(self.opp)

    def undo(self):
        logline = ""
        for m in range(2):
            for n in range(10):
                logline += (str(self.CountShape4[m][n]) + " ")
        # print(logline)
        self.moveCount -= 1
        self.remCount -= 1
        xp = self.remMove[self.moveCount].x
        yp = self.remMove[self.moveCount].y
        self.upperLeftCand = self.remULCand[self.remCount]
        self.lowerRightCand = self.remLRCand[self.remCount]
        c = self.remCell[self.remCount]
        c.update1(0)
        c.update1(1)
        c.update1(2)
        c.update1(3)
        c.update4(self.STATUS4)

        self.CountShape4[0][c.shape4[0]] += 1
        self.CountShape4[1][c.shape4[1]] += 1

        c.piece = OXSpace.EMPTY

        self.who = self.OPPNENT(self.who)
        self.opp = self.OPPNENT(self.opp)

        # self.table.undo(xp, yp, self.who)

        logline = ""
        for m in range(2):
            for n in range(10):
                logline += (str(self.CountShape4[m][n]) + " ")
        # print(logline)
        # unit = np.uint8(1)
        unit = 1
        for k in range(4):
            x = xp
            y = yp
            # p = np.uint8(16)
            p = 16
            while p != 0:
                x -= DX[k]
                y -= DY[k]
                self.space[x][y].pattern[k][self.who] ^= p
                if self.space[x][y].piece == OXSpace.EMPTY:
                    self.space[x][y].update1(k)
                    self.CountShape4[0][self.space[x][y].shape4[0]] -= 1
                    self.CountShape4[1][self.space[x][y].shape4[1]] -= 1
                    self.space[x][y].update4(self.STATUS4)
                    self.CountShape4[0][self.space[x][y].shape4[0]] += 1
                    self.CountShape4[1][self.space[x][y].shape4[1]] += 1
                p <<= unit
                p = p % 256
            x = xp
            y = yp
            # p = np.uint8(8)
            p = 8
            while p != 0:
                x += DX[k]
                y += DY[k]
                self.space[x][y].pattern[k][self.who] ^= p
                if self.space[x][y].piece == OXSpace.EMPTY:
                    self.space[x][y].update1(k)
                    self.CountShape4[0][self.space[x][y].shape4[0]] -= 1
                    self.CountShape4[1][self.space[x][y].shape4[1]] -= 1
                    self.space[x][y].update4(self.STATUS4)
                    self.CountShape4[0][self.space[x][y].shape4[0]] += 1
                    self.CountShape4[1][self.space[x][y].shape4[1]] += 1
                p >>= unit

        self.space[xp - 1][yp - 1].adj1 -= 1
        self.space[xp][yp - 1].adj1 -= 1
        self.space[xp + 1][yp - 1].adj1 -= 1
        self.space[xp - 1][yp].adj1 -= 1
        self.space[xp + 1][yp].adj1 -= 1
        self.space[xp - 1][yp + 1].adj1 -= 1
        self.space[xp][yp + 1].adj1 -= 1
        self.space[xp + 1][yp + 1].adj1 -= 1
        self.space[xp - 2][yp - 2].adj2 -= 1
        self.space[xp][yp - 2].adj2 -= 1
        self.space[xp + 2][yp - 2].adj2 -= 1
        self.space[xp - 2][yp].adj2 -= 1
        self.space[xp + 2][yp].adj2 -= 1
        self.space[xp - 2][yp + 2].adj2 -= 1
        self.space[xp][yp + 2].adj2 -= 1
        self.space[xp + 2][yp + 2].adj2 -= 1
        logline = ""
        for m in range(2):
            for n in range(10):
                logline += (str(self.CountShape4[m][n]) + " ")
        # print(logline)

    def _undo(self, x, y):
        if self.moveCount > 0 and\
                self.remMove[self.moveCount - 1].x == x + 4 and\
                self.remMove[self.moveCount - 1].y == y + 4:
            self.undo()
            return 0
        return 1

    def setWho(self, who):
        self.who = who
        self.opp = self.OPPNENT(who)
        if self.moveCount == 0:
            self.firstPlayer = who

    def OPPNENT(self, who):
        return 1 - who

    def yourTurn(self, depth=0, time=0):
        # self.table.clear()
        if depth > 0:
            self.nSearched = 0
            best = self.minmax(depth, True, -self.INF, self.INF)
            turnSearched = self.nSearched
            x = best.x - 4
            y = best.y - 4
            self.totalSearched += turnSearched
            if x < 0 or y < 0:
                return 10, 10
            return (x, y)

    def minmax(self, depth, root, alpha, beta):
        """
        @type depth: int64
        @type root: bool
        @type alpha: int64
        @type beta: int 64
        """
        if alpha > beta + 1:
            return OXMove(0, 0, beta + 1)
        best = OXMove(0, 0, alpha - 1)

        q = self.quickWinSearch()
        # print("answerto QWS{}".format(q))
        if q != 0:
            if not root:
                if q > 0:
                    return OXMove(0, 0, +self.WIN_MAX - q)
                else:
                    return OXMove(0, 0, -self.WIN_MAX - q)
            if q == 1:
                for y in range(
                    self.upperLeftCand.y, self.lowerRightCand.y + 1
                ):
                    for x in range(
                        self.upperLeftCand.x, self.lowerRightCand.x + 1
                    ):
                        if self.space[x][y].piece == OXSpace.EMPTY and\
                                (self.space[x][y].adj1 or self.space[x][y].adj2):
                            if self.space[x][y].shape4[self.who] == self.A:
                                return OXMove(x, y, self.WIN_MAX - 1)

        if depth == 0:
            return OXMove(0, 0, self.evaluate())
        depth -= 1

        cnd = self.generateCand()
        nCnd = len(cnd)
        if nCnd > 1:
            cnd.sort(key=(lambda a: (-a.value, -a.y, -a.x)))
            # print([(c.x, c.y, c.value) for c in cnd])
        elif nCnd == 1:
            if root:
                return OXMove(cnd[0].x, cnd[0].y, 0)
        else:
            for y in range(self.upperLeftCand.y, self.lowerRightCand.y + 1):
                for x in range(
                    self.upperLeftCand.x, self.lowerRightCand.x + 1
                ):
                    if self.space[x][y].piece == OXSpace.EMPTY and\
                            (self.space[x][y].adj1 or self.space[x][y].adj2):
                        if nCnd < self.MAX_CAND:
                            cnd.append(OXMove(x, y, 0))
                            nCnd += 1
            if nCnd == 0:
                best.value = 0
        for i in range(nCnd):
            # self.table.move(cnd[i].x, cnd[i].y, self.who)
            # if self.table.present() and\
            #         self.table.depth() >= depth and\
            #         (((int(self.table.depth()) ^ int(depth)) & 1) == 0 or
            #             abs(self.table.value() >= self.WIN_MIN)):
            #     self.nSearched += 1
            #     value = self.table.value()
            #     self.table.undo(cnd[i].x, cnd[i].y, self.who)
            # else:
            self._move(cnd[i].x, cnd[i].y)
            vA = -beta
            vB = -(best.value + 1)
            if vB >= self.WIN_MIN:
                vB += 1
            if vA <= -self.WIN_MIN:
                vA -= 1
            m = self.minmax(depth, False, vA, vB)
            # print(['minmax', m.x, m.y, m.value])

            value = -m.value

            if value >= self.WIN_MIN:
                value -= 1
            if value <= -self.WIN_MIN:
                value += 1
            # if -vB <= value and value <= -vA:
            #     self.table.update(value, depth, m.x, m.y)

            self.undo()
            if value > best.value:
                best = OXMove(cnd[i].x, cnd[i].y, value)
                if value > beta:
                    return OXMove(best.x, best.y, beta + 1)
        return best

    def generateCand(self):
        cnd = [OXMove() for _ in range(self.MAX_CAND)]
        nCnd = 0
        cnd[0].x = -1

        # if self.table.present(
        # ) and self.table.depth() >= 0 and self.table.best()[0] != 0:
        #     cnd[0].x = self.table.best()[0]
        #     cnd[0].y = self.table.best()[1]
        #     cnd[0].value = 10000
        #     nCnd = 1

        for y in range(self.upperLeftCand.y, self.lowerRightCand.y + 1):
            for x in range(self.upperLeftCand.x, self.lowerRightCand.x + 1):
                if self.space[x][y].piece == OXSpace.EMPTY and\
                        (self.space[x][y].adj1 or self.space[x][y].adj2):
                    if x != cnd[0].x or y != cnd[0].y:
                        cnd[nCnd].x = x
                        cnd[nCnd].y = y
                        cnd[nCnd].value = self.space[x][y].prior(self.PRIOR)
                        if cnd[nCnd].value > 1:
                            nCnd += 1
        if self.CountShape4[self.who][self.A] > 0:
            i = 0
            while self.space[cnd[i].x][cnd[i].y].shape4[self.who] != self.A:
                i += 1
            cnd[0] = cnd[i]
            nCnd = 1
            return cnd[:nCnd]

        if self.CountShape4[self.opp][self.A] > 0:
            i = 0
            while self.space[cnd[i].x][cnd[i].y].shape4[self.opp] != self.A:
                i += 1
            cnd[0] = cnd[i]
            nCnd = 1
            return cnd[:nCnd]

        if self.CountShape4[self.who][self.B] > 0:
            i = 0
            while self.space[cnd[i].x][cnd[i].y].shape4[self.who] != self.B:
                i += 1
            cnd[0] = cnd[i]
            nCnd = 1
            return cnd[:nCnd]

        if self.CountShape4[self.opp][self.B] > 0:
            nCnd = 0
            for y in range(self.upperLeftCand.y, self.lowerRightCand.y + 1):
                for x in range(
                    self.upperLeftCand.x, self.lowerRightCand.x + 1
                ):
                    if self.space[x][y].piece == OXSpace.EMPTY and\
                            (self.space[x][y].adj1 or self.space[x][y].adj2):
                        if self.space[x][y].shape4[self.who] >= self.E and self.space[x][y].shape4[self.who] != self.FORBID or\
                                self.space[x][y].shape4[self.opp] >= self.E and self.space[x][y].shape4[self.opp] != self.FORBID:
                            cnd[nCnd].x = x
                            cnd[nCnd].y = y
                            cnd[nCnd].value = self.space[x][y].prior(self.PRIOR)
                            if cnd[nCnd].value > 0:
                                nCnd += 1

        return cnd[:nCnd]

    def whowin(self):
        if self.CountShape4[self.who][self.A]:
            return self.who
        elif self.CountShape4[self.opp][self.A]:
            return self.opp
        return -1

    def quickWinSearch(self):
        # logline = ""
        # for m in range(2):
        #     for n in range(10):
        #         logline += (str(self.CountShape4[m][n]) + " ")
        # print(logline)
        if self.CountShape4[self.who][self.A] >= 1:
            return 1
        if self.CountShape4[self.opp][self.A] >= 2:
            return -2
        if self.CountShape4[self.opp][self.A] == 1:
            for y in range(self.upperLeftCand.y, self.lowerRightCand.y + 1):
                for x in range(
                    self.upperLeftCand.x, self.lowerRightCand.x + 1
                ):
                    if self.space[x][y].piece == OXSpace.EMPTY and\
                            (self.space[x][y].adj1 or self.space[x][y].adj2):
                        if self.space[x][y].shape4[self.opp] == self.A:
                            self._move(x, y)
                            q = -self.quickWinSearch()
                            # print(['qws', q])
                            self.undo()
                            if q < 0:
                                q -= 1
                            elif q > 0:
                                q += 1
                            return q
        if self.CountShape4[self.who][self.B] >= 1:
            return 3
        if self.CountShape4[self.who][self.C] >= 1:
            if self.CountShape4[self.opp][self.B] == 0 and\
                    self.CountShape4[self.opp][self.C] == 0 and\
                    self.CountShape4[self.opp][self.D] == 0 and\
                    self.CountShape4[self.opp][self.E] == 0:
                return 5
            for y in range(self.upperLeftCand.y, self.lowerRightCand.y + 1):
                for x in range(
                    self.upperLeftCand.x, self.lowerRightCand.x + 1
                ):
                    if self.space[x][y].piece == OXSpace.EMPTY and\
                            (self.space[x][y].adj1 or self.space[x][y].adj2):
                        if self.space[x][y].shape4[self.who] == self.C:
                            self._move(x, y)
                            q = -self.quickWinSearch()
                            self.undo()
                            if q > 0:
                                return q + 1
        if self.CountShape4[self.who][self.F] >= 1:
            if self.CountShape4[self.opp][self.B] == 0 and\
                    self.CountShape4[self.opp][self.C] == 0 and\
                    self.CountShape4[self.opp][self.D] == 0 and\
                    self.CountShape4[self.opp][self.E] == 0:
                return 5
        return 0

    # def log(self):
    #     logfile = open("board.log", "w")
    #     for i in range(self.boardWidth):
    #         for j in range(self.boardHeight):
    #             logfile.write("({}, {}): pattern: ".format(i, j))
    #             for m in range(4):
    #                 for n in range(2):
    #                     logfile.write(
    #                         str(self.space[i + 4][j + 4].pattern[m][n]) + " "
    #                     )
    #             logfile.write("shape1: ")
    #             for m in range(4):
    #                 for n in range(2):
    #                     logfile.write(
    #                         str(self.space[i + 4][j + 4].shape1[m][n]) + " "
    #                     )
    #             logfile.write("shape4: ")
    #             logfile.write(str(self.space[i + 4][j + 4].shape4[0]) + " ")
    #             logfile.write(str(self.space[i + 4][j + 4].shape4[1]) + "\n")
    #     for m in range(2):
    #         for n in range(10):
    #             logfile.write(str(self.CountShape4[m][n]) + " ")

    # def __str__(self):
    #     board = [[space.piece for space in row][4:24] for row in self.space][4:24]
    #     symbols = {0: '⚈', 1: '⚆', 2: '⋅', 10: '⭓', 11: '⭔'}
    #     colnums = ''.join(['  '] + ['{:^3d}'.format(num) for num in range(20)])

    #     def s(x, i, j):
    #         if (i, j) == self.premove:
    #             x += 10
    #         return '{:^3}'.format(symbols.get(x, '⋅'))

    #     return '\n'.join(
    #         [colnums] + [
    #             ''.join(
    #                 ['%2d' % i] + [s(x, i, j)
    #                                for x, j in zip(row, range(20))] +
    #                 ['%-2d' % i]
    #             )
    #             for row, i in zip(board, range(20))
    #         ] + [colnums]
    #     )


def main():
    ai = AI()
    ai.start(20, 20)
    ai.setWho(0)
    ai.move(10, 10)
    # ai.setWho(0)
    # ai.move(10, 11)
    # ai.setWho(0)
    # ai.move(10, 12)
    # ai.setWho(0)
    # ai.move(10, 11)
    # print(ai.space[14][18].pattern)
    print(ai.space[14][19].shape1)
    print(ai.space[14][18].shape1)
    print(ai.space[14][17].shape1)
    print(ai.space[14][16].shape1)
    print(ai.space[14][15].shape1)
    print(ai.space[14][14].shape1)
    # ai.setWho(0)
    # ai.move(10, 12)
    # print(ai.space[14][18].pattern)
    # print(ai.space[14][18].shape1)
    # ai.setWho(0)
    # ai.move(10, 13)
    # print(ai.space[14][18].pattern)
    # print(ai.space[14][18].shape1)
    # ai.move(10, 11)
    # ai.move(9, 10)
    # ai.move(9, 11)
    # ai.move(8, 10)
    ai.evaluate()
    # ai.move(10, 10)
    #x, y = ai.yourTurn(2)
    #eprint([x, y])
    #ai.move(x, y)
    # ai.move(11, 11)
    # ai.move(8, 12)
    #    x, y = ai.yourTurn(2)
    #    eprint([x, y])
    #    ai.move(x, y)
    # ai.move(9, 11)
    # ai.move(11, 10)
    #    x, y = ai.yourTurn(2)
    #    eprint([x, y])
    #    ai.move(x, y)
    #    x, y = ai.yourTurn(2)
    #    eprint([x, y])
    #    ai.move(x, y)
    # x, y = ai.yourTurn(6)
    # ai.move(x, y)


if __name__ == "__main__":
    main()
