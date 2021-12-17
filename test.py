with open('Dataset/test/baidu_eng_test.txt', 'r', encoding='UTF-8') as f:
        x_test = []
        y_test = []
        for i,line in enumerate(f.readlines()):
            data = line.strip().split('\t')
            y_test.append(data[0])
            x_test.append(data[1].strip().split())
            print(i)

