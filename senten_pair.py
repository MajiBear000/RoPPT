def sentence_pair(sentence, tokens):
    idx = 0
    log_idx = 0
    log = True
    words = sentence.lower().split()
    term = ''
    for i, token in enumerate(tokens):
        if idx == len(words):
            break
        if term+token == words[idx]:
            if idx == 0:
                idx += 1
                term=''
                log=True
            else:
                if log==True:
                    tokens[i] = ' '+tokens[i]
                else:
                    tokens[log_idx] = ' '+tokens[log_idx]
                    log=True
                    term=''
                idx += 1
        elif log:
            log_idx = i
            term = token
            log=False
        else:
            term += token
            log=False
    return tokens

'''
    print('$$$$$$$$$   :',sentence)
    print('$$$$$$$$$$:', tokens)

a = 'Type "help", "copyright", "credits" or "license" for more information.'
b = ['Type', '"', 'help','"',',','"', 'copy', 'right','"',',', '"credits"','or','"license"','for', 'more','info','romation','.','\n' ]
sentence_pair(a, b)
'''
