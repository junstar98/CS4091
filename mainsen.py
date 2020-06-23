import pandas as pd
f = open('result.csv', 'w')
for j in range(8):
    csv_file = pd.read_csv(str(j+1) + '.csv', names=['date', 'category','news', 'inputs', 'targets', ''])
    for i in range(len(csv_file['inputs'])):
        s_list = csv_file['inputs'][i].split()
        targets = csv_file['targets'][i]
        sentence_list = targets.split('. ')
        total_num = len(s_list)
        result_num = 0
        result_sentence = ""
        for sentence in sentence_list:
            current_num = 0
            word_list = sentence.split()
            for k in s_list:
                if (k in word_list):
                    current_num += 1
            if result_num < current_num:
                result_num = current_num
                result_sentence = sentence
        if(result_num > 0.6):
            f.write('0, ')
            f.write(result_sentence)
            f.write(",")
            f.write(csv_file['inputs'][i])
            f.write("\n")
f.close()







