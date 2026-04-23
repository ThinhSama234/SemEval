import sys
lines = open(r'C:\Users\CPU12495\.claude\projects\d--SemEval-2026-Task13\1e297adb-7a3e-4896-9e9a-91a15140ea1e\tool-results\bajz6r9dy.txt','r').readlines()
for line in lines[-80:]:
    s = line.rstrip()
    if 'it/s]' in s and 'Train' not in s and '100%' not in s:
        continue
    if s:
        print(s)
