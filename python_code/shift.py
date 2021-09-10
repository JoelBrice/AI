import sys
import math

# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.
answer = []
for i in range(5):
    line = input()
    if line[i].isdigit():
        line = " "
    else:
        answer.append(line[i] + str(line[i]))

# Write an answer using print
# To debug: print("Debug messages...", file=sys.stderr, flush=True)

print(answer)
