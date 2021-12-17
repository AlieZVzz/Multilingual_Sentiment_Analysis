ls = open('baidu_zh1.txt', 'w', encoding='utf-8')
with open('baidu_zh.txt', 'r', encoding='utf-8') as f:
    i = 0
    for line in f.readlines():
        if line == '1' + '\t' + 'error cant write' + '\n':
            i += 1
        elif line == '0' + '\t' + 'error cant write' + '\n':
            i += 1
        else:
            ls.write(line)
    print(i)
