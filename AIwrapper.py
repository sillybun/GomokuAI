import random
import pisqpipe as pp
from pisqpipe import DEBUG_EVAL, DEBUG
import AI

pp.infotext = "zhangyiteng"

MAX_BOARD = 100
ai = AI.AI()


def brain_init():
    if pp.width < 5 or pp.height < 5:
        pp.pipeOut("ERROR size of the board")
        return
    if pp.width > MAX_BOARD or pp.height > MAX_BOARD:
        pp.pipeOut("ERROR Maximal board size is {}".format(MAX_BOARD))
        return
    pp.pipeOut("OK")


def brain_restart():
    ai.start(pp.width, pp.height)
    pp.pipeOut("OK")


def isFree(x, y):
    return x >= 0 and y >= 0 and x < pp.width and y < pp.height and ai.cell[x][y].piece == AI.OXCell.EMPTY


def brain_my(x, y):
    ai.setWho(ai.XP)
    ai.move(x, y)


def brain_opponents(x, y):
    ai.setWho(ai.OP)
    ai.move(x, y)


def brain_block(x, y):
    ai.block(x, y)


def brain_takeback(x, y):
    ai.undo()


def brain_turn():
    if pp.terminateAI:
        return
    ai.setWho(ai.XP)
    x, y = ai.yourTurn()
    ai.move(x, y)
    pp.do_mymove(x, y)


def brain_end():
    pass


def brain_about():
    pp.pipeOut(pp.infotext)


if DEBUG_EVAL:
    import win32gui

    def brain_eval(x, y):
        # TODO check if it works as expected
        return 
#        wnd = win32gui.GetForegroundWindow()
#        dc = win32gui.GetDC(wnd)
#        rc = win32gui.GetClientRect(wnd)
#        c = str(board[x][y])
#        win32gui.ExtTextOut(dc, rc[2] - 15, 3, 0, None, c, ())
#        win32gui.ReleaseDC(wnd, dc)

# "overwrites" functions in pisqpipe module
pp.brain_init = brain_init
pp.brain_restart = brain_restart
pp.brain_my = brain_my
pp.brain_opponents = brain_opponents
pp.brain_block = brain_block
pp.brain_takeback = brain_takeback
pp.brain_turn = brain_turn
pp.brain_end = brain_end
pp.brain_about = brain_about
if DEBUG_EVAL:
    pp.brain_eval = brain_eval


def main():
    pp.main()


if __name__ == "__main__":
    main()
