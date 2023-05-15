import argparse
import json
import csv
import os

def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--data_path', type=str, default='data/VUA18',
                        help='Directory of where metaphor data held')
    return parser.parse_args()

def read_file(file_path):
    dataset = []
    with open(file_path, encoding='utf8') as f:
        lines = csv.reader(f, delimiter='\t')
        next(lines)
       
        flag=True
        for line in lines:
            sen_id = line[0]
            sentence = line[2]
            ind = line[4]
            label = line[1]
            pos = line[3]
        
            index = int(ind)
            
            word = sentence.split()[index]

            dataset.append([word, sentence, index, label, pos])
    print(file_path, len(dataset))
    restore_data(file_path, dataset)

def restore_data(file_path, dataset):
    with open(file_path.replace('.tsv', '.raw'), 'w') as f:
        for sample in dataset:
            sentence = sample[1]
            index = sample[2]
            label = sample[3]
            pos = sample[4]
            if not label=='1':
                label = '-1'
            target = sample[0]
            raw_sent = sample[1].split()
            raw_sent[index] = '$T$'
            raw_sent = ' '.join(raw_sent)
            
            f.write(sentence + '\n')
            f.write(str(index) + '\n')
            f.write(pos + '\n')
            f.write(raw_sent + '\n')
            f.write(target + '\n')
            f.write(label + '\n')
        
    f.close()

def main():
    args = parse_args()

    train_file = os.path.join(args.data_path, 'train.tsv')
    test_file = os.path.join(args.data_path, 'test.tsv')
    val_file = os.path.join(args.data_path, 'val.tsv')
    
    # csv -> raw
    read_file(train_file)
    read_file(test_file)
    read_file(val_file)

if __name__ == "__main__":
    main()
