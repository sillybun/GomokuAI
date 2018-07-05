import numpy as np

import CONFIG
import COUNT5
import PRIOR
import STATUS1
import numba as nb
from numba import jit, jitclass, int8, uint8, int64, int32
from zytutil import timethis


class OXCell:
    EMPTY = 2
    WRONG = 3


DX = [1, 0, 1, 1]
DY = [0, 1, 1, -1]

@jitclass([('x', uint8), ('y', uint8)])
class OXPoint:

    def __init__(self, x=uint8(0), y=uint8(0)):
        self.x = x
        self.y = y

@jitclass([('x', uint8), ('y', uint8), ('value', int32)])
class OXMove:
    """
    class OXMove
    """

    def __init__(self, x=uint8(0), y=uint8(0), value=int32(0)):
        """
        @type self: OXMove
        @type x: np.uint8
        @type y: np.uint8
        @type y: np.int32
        """
        self.x = x
        self.y = y
        self.value = value

    def point(self):
        """
        @type: OXMove
        """
        return OXPoint(self.x, self.y)

spec = [
    ('piece', int64),               # a simple scalar field
    ('pattern', uint8[:, :]),          # an array field
    ('status1', uint8[:, :]),
    ('status4', uint8[:]),
    ('adj1', uint8),
    ('adj2', uint8)
]

@jitclass(spec)
class Cell:

    def __init__(self):
        self.piece = int64(0)
        self.pattern = np.zeros((4, 2), dtype=np.uint8)
        self.status1 = np.zeros((4, 2), dtype=np.uint8)
        self.status4 = np.zeros(2, dtype=np.uint8)
        self.adj1 = uint8(0)
        self.adj2 = uint8(0)

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
        self.status4[0] = STATUS4[self.status1[0][0]][self.status1[1][0]][
            self.status1[2][0]
        ][self.status1[3][0]]
        self.status4[1] = STATUS4[self.status1[0][1]][self.status1[1][1]][
            self.status1[2][1]
        ][self.status1[3][1]]

    def update1(self, STATUS1, k):
        self.status1[k][0] = STATUS1[self.pattern[k][0]][self.pattern[k][1]]
        self.status1[k][1] = STATUS1[self.pattern[k][1]][self.pattern[k][0]]

# @jitclass
class AI:
    # WIN_MIN = 25000
    # WIN_MAX = 30000
    # INF = 32000
    # A = 8
    # B = 7
    # C = 6
    # E = 4
    # D = 5
    # F = 3
    # G = 2
    # H = 1
    # FORBID = 9

    def __init__(self, _STATUS1):
        self.STATUS4 = np.zeros([10, 10, 10, 10], dtype=np.uint8)
        self.RANK = np.zeros(107, dtype=np.uint8)
        self.PRIOR = np.zeros([256, 256], dtype=np.uint8)
        self.cell = [[Cell() for _ in range(30)] for _ in range(30)]
        self.OP = 0
        self.XP = 1
        self.STATUS1 = _STATUS1
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
        self.MAX_CAND = 256
        self.premove = 0, 0

    def getStatus4(self, s0, s1, s2, s3):
        n = np.zeros(10, dtype=np.uint8)
        n[s0] += 1
        n[s1] += 1
        n[s2] += 1
        n[s3] += 1

        if n[9] >= 1:
            return 8
        if n[8] >= 1:
            return 7
        if n[7] >= 2:
            return 7
        if n[7] >= 1 and n[6] >= 1:
            return 6
        if n[7] >= 1 and n[5] >= 1:
            return 5
        if n[7] >= 1 and n[4] >= 1:
            return 5
        if n[7] >= 1:
            return 4
        if n[6] >= 2:
            return 3
        if n[6] >= 1 and n[5] >= 1:
            return 2
        if n[6] >= 1 and n[4] >= 1:
            return 2
        if n[6] >= 1:
            return 1
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
        p = np.zeros(2, dtype=np.int64)
        for i in range(self.remCount):
            c = self.remCell[i]
            a = c.piece
            for k in range(4):
                # print([c.pattern[k][a], c.pattern[k][1 - a]])
                p[a] += self.RANK[CONFIG.content[c.pattern[k][a]][c.pattern[k]
                                                                  [1 - a]]]
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
                    self.cell[x][y].piece = OXCell.WRONG
                else:
                    self.cell[x][y].piece = OXCell.EMPTY
                for k in range(4):
                    self.cell[x][y].pattern[k][0] = self.cell[x][y].pattern[k][
                        1
                    ] = 0
        unit = np.uint8(1)
        for y in range(4, height + 4):
            for x in range(4, width + 4):
                for k in range(0, 4):
                    xx = x - DX[k]
                    yy = y - DY[k]
                    p = np.uint8(8)
                    while p != 0:
                        if self.cell[xx][yy].piece == OXCell.WRONG:
                            self.cell[x][y].pattern[k][0] |= p
                            self.cell[x][y].pattern[k][1] |= p
                        xx -= DX[k]
                        yy -= DY[k]
                        p >>= unit
                    xx = x + DX[k]
                    yy = y + DY[k]
                    p = np.uint8(16)
                    while p != 0:
                        if self.cell[xx][yy].piece == OXCell.WRONG:
                            self.cell[x][y].pattern[k][0] |= p
                            self.cell[x][y].pattern[k][1] |= p
                        xx += DX[k]
                        yy += DY[k]
                        p <<= unit
        for y in range(4, height + 4):
            for x in range(4, height + 4):
                self.cell[x][y].update1(self.STATUS1, 0)
                self.cell[x][y].update1(self.STATUS1, 1)
                self.cell[x][y].update1(self.STATUS1, 2)
                self.cell[x][y].update1(self.STATUS1, 3)
                self.cell[x][y].update4(self.STATUS4)
                self.cell[x][y].adj1 = self.cell[x][y].adj2 = 0
        self.nSt = np.zeros((2, 10), dtype=np.int64)
        self.totalSearched = 0
        self.who = self.OP
        self.opp = self.XP
        self.moveCount = self.remCount = 0
        self.upperLeftCand = OXPoint(99, 99)
        self.lowerRightCand = OXPoint(0, 0)
        self.nSearched = 0
        return

    def move(self, x, y):
        """
        @type x: np.uint8
        @type y: np.uint8
        """
        self.premove = x, y
        if self.cell[x + 4][y + 4].piece == OXCell.EMPTY:
            self._move(x + 4, y + 4)
        # else:
            # eprintf("wrong move")

    def _move(self, xp, yp, updateHash=True):
        """
        @type xp: np.uint8
        @type yp: np.uint8
        """
        # print(["move", xp, yp, updateHash])
        logline = ""
        for m in range(2):
            for n in range(10):
                logline += (str(self.nSt[m][n]) + " ")
        # print(logline)
        self.nSearched += 1
        self.nSt[0][self.cell[xp][yp].status4[0]] -= 1
        self.nSt[1][self.cell[xp][yp].status4[1]] -= 1

        self.cell[xp][yp].piece = self.who
        self.remCell[self.remCount] = self.cell[xp][yp]
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
        unit = np.uint8(1)
        for k in range(4):
            x = xp
            y = yp
            p = np.uint8(16)
            while p != 0:
                x -= DX[k]
                y -= DY[k]
                self.cell[x][y].pattern[k][self.who] |= p
                if self.cell[x][y].piece == OXCell.EMPTY:
                    self.cell[x][y].update1(self.STATUS1, k)
                    self.nSt[0][self.cell[x][y].status4[0]] -= 1
                    self.nSt[1][self.cell[x][y].status4[1]] -= 1
                    self.cell[x][y].update4(self.STATUS4)
                    self.nSt[0][self.cell[x][y].status4[0]] += 1
                    self.nSt[1][self.cell[x][y].status4[1]] += 1
                p <<= unit
            x = xp
            y = yp
            p = np.uint8(8)
            while p != 0:
                x += DX[k]
                y += DY[k]
                self.cell[x][y].pattern[k][self.who] |= p
                if self.cell[x][y].piece == OXCell.EMPTY:
                    self.cell[x][y].update1(self.STATUS1, k)
                    self.nSt[0][self.cell[x][y].status4[0]] -= 1
                    self.nSt[1][self.cell[x][y].status4[1]] -= 1
                    self.cell[x][y].update4(self.STATUS4)
                    self.nSt[0][self.cell[x][y].status4[0]] += 1
                    self.nSt[1][self.cell[x][y].status4[1]] += 1
                p >>= unit

        self.cell[xp - 1][yp - 1].adj1 += 1
        self.cell[xp][yp - 1].adj1 += 1
        self.cell[xp + 1][yp - 1].adj1 += 1
        self.cell[xp - 1][yp].adj1 += 1
        self.cell[xp + 1][yp].adj1 += 1
        self.cell[xp - 1][yp + 1].adj1 += 1
        self.cell[xp][yp + 1].adj1 += 1
        self.cell[xp + 1][yp + 1].adj1 += 1
        self.cell[xp - 2][yp - 2].adj2 += 1
        self.cell[xp][yp - 2].adj2 += 1
        self.cell[xp + 2][yp - 2].adj2 += 1
        self.cell[xp - 2][yp].adj2 += 1
        self.cell[xp + 2][yp].adj2 += 1
        self.cell[xp - 2][yp + 2].adj2 += 1
        self.cell[xp][yp + 2].adj2 += 1
        self.cell[xp + 2][yp + 2].adj2 += 1

        self.who = self.OPPNENT(self.who)
        self.opp = self.OPPNENT(self.opp)
        logline = ""
        for m in range(2):
            for n in range(10):
                logline += (str(self.nSt[m][n]) + " ")
        # print(logline)

    def block(self, x, y):
        xp = x + 4
        yp = y + 4

        self.nSt[0][self.cell[xp][yp].status4[0]] -= 1
        self.nSt[1][self.cell[xp][yp].status4[1]] -= 1

        self.cell[xp][yp].piece = OXCell.WRONG
        self.remMove[self.moveCount] = OXPoint(xp, yp)
        self.moveCount += 1

        unit = np.uint8(1)
        for k in range(4):
            x = xp
            y = yp
            p = np.uint8(16)
            while p != 0:
                x -= DX[k]
                y -= DY[k]
                self.cell[x][y].pattern[k][0] |= p
                self.cell[x][y].pattern[k][1] |= p
                if self.cell[x][y].piece == OXCell.EMPTY:
                    self.cell[x][y].update1(self.STATUS1, k)
                    self.nSt[0][self.cell[x][y].status4[0]] -= 1
                    self.nSt[1][self.cell[x][y].status4[1]] -= 1
                    self.cell[x][y].update4(self.STATUS4)
                    self.nSt[0][self.cell[x][y].status4[0]] += 1
                    self.nSt[1][self.cell[x][y].status4[1]] += 1
                p <<= unit
            x = xp
            y = yp
            p = np.uint8(8)
            while p != 0:
                x += DX[k]
                y += DY[k]
                self.cell[x][y].pattern[k][0] |= p
                self.cell[x][y].pattern[k][1] |= p
                if self.cell[x][y].piece == OXCell.EMPTY:
                    self.cell[x][y].update1(self.STATUS1, k)
                    self.nSt[0][self.cell[x][y].status4[0]] -= 1
                    self.nSt[1][self.cell[x][y].status4[1]] -= 1
                    self.cell[x][y].update4(self.STATUS4)
                    self.nSt[0][self.cell[x][y].status4[0]] += 1
                    self.nSt[1][self.cell[x][y].status4[1]] += 1
                p >>= unit

        self.who = self.OPPNENT(self.who)
        self.opp = self.OPPNENT(self.opp)

    def undo(self):
        logline = ""
        for m in range(2):
            for n in range(10):
                logline += (str(self.nSt[m][n]) + " ")
        # print(logline)
        self.moveCount -= 1
        self.remCount -= 1
        xp = self.remMove[self.moveCount].x
        yp = self.remMove[self.moveCount].y
        self.upperLeftCand = self.remULCand[self.remCount]
        self.lowerRightCand = self.remLRCand[self.remCount]
        c = self.remCell[self.remCount]
        c.update1(self.STATUS1, 0)
        c.update1(self.STATUS1, 1)
        c.update1(self.STATUS1, 2)
        c.update1(self.STATUS1, 3)
        c.update4(self.STATUS4)

        self.nSt[0][c.status4[0]] += 1
        self.nSt[1][c.status4[1]] += 1

        c.piece = OXCell.EMPTY

        self.who = self.OPPNENT(self.who)
        self.opp = self.OPPNENT(self.opp)

        # print(logline)
        unit = np.uint8(1)
        for k in range(4):
            x = xp
            y = yp
            p = np.uint8(16)
            while p != 0:
                x -= DX[k]
                y -= DY[k]
                self.cell[x][y].pattern[k][self.who] ^= p
                if self.cell[x][y].piece == OXCell.EMPTY:
                    self.cell[x][y].update1(self.STATUS1, k)
                    self.nSt[0][self.cell[x][y].status4[0]] -= 1
                    self.nSt[1][self.cell[x][y].status4[1]] -= 1
                    self.cell[x][y].update4(self.STATUS4)
                    self.nSt[0][self.cell[x][y].status4[0]] += 1
                    self.nSt[1][self.cell[x][y].status4[1]] += 1
                p <<= unit
            x = xp
            y = yp
            p = np.uint8(8)
            while p != 0:
                x += DX[k]
                y += DY[k]
                self.cell[x][y].pattern[k][self.who] ^= p
                if self.cell[x][y].piece == OXCell.EMPTY:
                    self.cell[x][y].update1(self.STATUS1, k)
                    self.nSt[0][self.cell[x][y].status4[0]] -= 1
                    self.nSt[1][self.cell[x][y].status4[1]] -= 1
                    self.cell[x][y].update4(self.STATUS4)
                    self.nSt[0][self.cell[x][y].status4[0]] += 1
                    self.nSt[1][self.cell[x][y].status4[1]] += 1
                p >>= unit

        self.cell[xp - 1][yp - 1].adj1 -= 1
        self.cell[xp][yp - 1].adj1 -= 1
        self.cell[xp + 1][yp - 1].adj1 -= 1
        self.cell[xp - 1][yp].adj1 -= 1
        self.cell[xp + 1][yp].adj1 -= 1
        self.cell[xp - 1][yp + 1].adj1 -= 1
        self.cell[xp][yp + 1].adj1 -= 1
        self.cell[xp + 1][yp + 1].adj1 -= 1
        self.cell[xp - 2][yp - 2].adj2 -= 1
        self.cell[xp][yp - 2].adj2 -= 1
        self.cell[xp + 2][yp - 2].adj2 -= 1
        self.cell[xp - 2][yp].adj2 -= 1
        self.cell[xp + 2][yp].adj2 -= 1
        self.cell[xp - 2][yp + 2].adj2 -= 1
        self.cell[xp][yp + 2].adj2 -= 1
        self.cell[xp + 2][yp + 2].adj2 -= 1
        logline = ""
        for m in range(2):
            for n in range(10):
                logline += (str(self.nSt[m][n]) + " ")
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

    @timethis
    def yourTurn(self, depth=0, time=0):
        if depth > 0:
            self.nSearched = 0
            best = self.minmax(depth, True, -32000, 32000)
            turnSearched = self.nSearched
            x = best.x - 4
            y = best.y - 4
            self.totalSearched += turnSearched
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
                    return OXMove(0, 0, +30000 - q)
                else:
                    return OXMove(0, 0, -30000 - q)
            if q == 1:
                for y in range(
                    self.upperLeftCand.y, self.lowerRightCand.y + 1
                ):
                    for x in range(
                        self.upperLeftCand.x, self.lowerRightCand.x + 1
                    ):
                        if self.cell[x][y].piece == OXCell.EMPTY and\
                                (self.cell[x][y].adj1 or self.cell[x][y].adj2):
                            if self.cell[x][y].status4[self.who] == 8:
                                return OXMove(x, y, 30000 - 1)

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
                    if self.cell[x][y].piece == OXCell.EMPTY and\
                            (self.cell[x][y].adj1 or self.cell[x][y].adj2):
                        if nCnd < self.MAX_CAND:
                            cnd.append(OXMove(x, y, 0))
                            nCnd += 1
            if nCnd == 0:
                best.value = 0
        for i in range(nCnd):
            self._move(cnd[i].x, cnd[i].y, False)
            vA = -beta
            vB = -(best.value + 1)
            if vB >= 25000:
                vB += 1
            if vA <= -25000:
                vA -= 1
            m = self.minmax(depth, False, vA, vB)
            # print(['minmax', m.x, m.y, m.value])

            value = -m.value

            if value >= 25000:
                value -= 1
            if value <= -25000:
                value += 1
            self.undo()

            if value > best.value:
                best = OXMove(cnd[i].x, cnd[i].y, value)
                if value > beta:
                    return OXMove(best.x, best.y, beta + 1)
        return best

    def generateCand(self):
        cnd = [OXMove(0, 0, 0) for _ in range(self.MAX_CAND)]
        nCnd = 0
        cnd[0].x = -1

        for y in range(self.upperLeftCand.y, self.lowerRightCand.y + 1):
            for x in range(self.upperLeftCand.x, self.lowerRightCand.x + 1):
                if self.cell[x][y].piece == OXCell.EMPTY and\
                        (self.cell[x][y].adj1 or self.cell[x][y].adj2):
                    if x != cnd[0].x or y != cnd[0].y:
                        cnd[nCnd].x = x
                        cnd[nCnd].y = y
                        cnd[nCnd].value = self.cell[x][y].prior(self.PRIOR)
                        if cnd[nCnd].value > 1:
                            nCnd += 1
        if self.nSt[self.who][8] > 0:
            i = 0
            while self.cell[cnd[i].x][cnd[i].y].status4[self.who] != 8:
                i += 1
            cnd[0] = cnd[i]
            nCnd = 1
            return cnd[:nCnd]

        if self.nSt[self.opp][8] > 0:
            i = 0
            while self.cell[cnd[i].x][cnd[i].y].status4[self.opp] != 8:
                i += 1
            cnd[0] = cnd[i]
            nCnd = 1
            return cnd[:nCnd]

        if self.nSt[self.who][7] > 0:
            i = 0
            while self.cell[cnd[i].x][cnd[i].y].status4[self.who] != 7:
                i += 1
            cnd[0] = cnd[i]
            nCnd = 1
            return cnd[:nCnd]

        if self.nSt[self.opp][7] > 0:
            nCnd = 0
            for y in range(self.upperLeftCand.y, self.lowerRightCand.y + 1):
                for x in range(
                    self.upperLeftCand.x, self.lowerRightCand.x + 1
                ):
                    if self.cell[x][y].piece == OXCell.EMPTY and\
                            (self.cell[x][y].adj1 or self.cell[x][y].adj2):
                        if self.cell[x][y].status4[self.who] >= 4 or self.cell[x][y].status4[self.opp] >= 4:
                            cnd[nCnd].x = x
                            cnd[nCnd].y = y
                            cnd[nCnd].value = self.cell[x][y].prior(self.PRIOR)
                            if cnd[nCnd].value > 0:
                                nCnd += 1

        return cnd[:nCnd]

    def whowin(self):
        if self.nSt[self.who][8]:
            return self.who
        elif self.nSt[self.opp][8]:
            return self.opp
        return -1

    def quickWinSearch(self):
        logline = ""
        for m in range(2):
            for n in range(10):
                logline += (str(self.nSt[m][n]) + " ")
        # print(logline)
        if self.nSt[self.who][8] >= 1:
            return 1
        if self.nSt[self.opp][8] >= 2:
            return -2
        if self.nSt[self.opp][8] == 1:
            for y in range(self.upperLeftCand.y, self.lowerRightCand.y + 1):
                for x in range(
                    self.upperLeftCand.x, self.lowerRightCand.x + 1
                ):
                    if self.cell[x][y].piece == OXCell.EMPTY and\
                            (self.cell[x][y].adj1 or self.cell[x][y].adj2):
                        if self.cell[x][y].status4[self.opp] == 8:
                            self._move(x, y)
                            q = -self.quickWinSearch()
                            # print(['qws', q])
                            self.undo()
                            if q < 0:
                                q -= 1
                            elif q > 0:
                                q += 1
                            return q
        if self.nSt[self.who][7] >= 1:
            return 3
        if self.nSt[self.who][6] >= 1:
            if self.nSt[self.opp][7] == 0 and\
                    self.nSt[self.opp][6] == 0 and\
                    self.nSt[self.opp][5] == 0 and\
                    self.nSt[self.opp][4] == 0:
                return 5
            for y in range(self.upperLeftCand.y, self.lowerRightCand.y + 1):
                for x in range(
                    self.upperLeftCand.x, self.lowerRightCand.x + 1
                ):
                    if self.cell[x][y].piece == OXCell.EMPTY and\
                            (self.cell[x][y].adj1 or self.cell[x][y].adj2):
                        if self.cell[x][y].status4[self.who] == 6:
                            self._move(x, y)
                            q = -self.quickWinSearch()
                            self.undo()
                            if q > 0:
                                return q + 1
        if self.nSt[self.who][3] >= 1:
            if self.nSt[self.opp][7] == 0 and\
                    self.nSt[self.opp][6] == 0 and\
                    self.nSt[self.opp][5] == 0 and\
                    self.nSt[self.opp][4] == 0:
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
    #                         str(self.cell[i + 4][j + 4].pattern[m][n]) + " "
    #                     )
    #             logfile.write("status1: ")
    #             for m in range(4):
    #                 for n in range(2):
    #                     logfile.write(
    #                         str(self.cell[i + 4][j + 4].status1[m][n]) + " "
    #                     )
    #             logfile.write("status4: ")
    #             logfile.write(str(self.cell[i + 4][j + 4].status4[0]) + " ")
    #             logfile.write(str(self.cell[i + 4][j + 4].status4[1]) + "\n")
    #     for m in range(2):
    #         for n in range(10):
    #             logfile.write(str(self.nSt[m][n]) + " ")

    # def __str__(self):
    #     board = [[cell.piece for cell in row][4:24] for row in self.cell][4:24]
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
    ai.move(10, 10)
    #x, y = ai.yourTurn(2)
    #eprint([x, y])
    #ai.move(x, y)
    ai.move(11, 11)
    ai.move(8, 12)
    #    x, y = ai.yourTurn(2)
    #    eprint([x, y])
    #    ai.move(x, y)
    ai.move(9, 11)
    ai.move(11, 10)
    #    x, y = ai.yourTurn(2)
    #    eprint([x, y])
    #    ai.move(x, y)
    #    x, y = ai.yourTurn(2)
    #    eprint([x, y])
    #    ai.move(x, y)
    x, y = ai.yourTurn(6)
    ai.move(x, y)


if __name__ == "__main__":
    main()
