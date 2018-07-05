
import os
# import newAI
import AIcopy
import STATUS1

ai = AIcopy.AI()
ai.start(20, 20)
ai.move(10, 10)
print(ai.yourTurn(5))
# print(ai.yourTurn(4))
# print(ai.yourTurn(4))
# while True:
#     os.system('clear')
#     print(ai)
#     word = input("Your Trun")
#     if word == "back":
#         ai.undo()
#         ai.undo()
#     else:
#         x, y = eval(word)
#         ai.move(x, y)
#         x, y = ai.yourTurn(4)
#         print([x, y])
#         ai.move(x, y)

