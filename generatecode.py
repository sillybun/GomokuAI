import random

print("def f(a, b):")
print("    return a+b")

for i in range(10000):
    print("f({}, {})".format(random.randint(1, 100), random.randint(1, 100)))
