corpus = '/home/liuyh0916/calibration/datasets/corpus/psgs_w100.tsv'
num_lines = 10


def read_csv(fname, num_lines=None):
    with open(fname, 'r') as f:
        lines = []
        for _ in range(num_lines):
            line = f.readline()
            if not line:
                break
            lines.append(line.strip().split('\t'))

    for line in lines:
        print(line)

def read_json(fname, num_lines=None):
    import json
    with open(fname, 'r') as f:
        lines = f.read()
        lines = json.loads(lines)
    cases = []
    for line in lines:
        cases.append(line)
        if len(cases) > num_lines:
            break
    for case in cases:
        print(case.keys())

# fname = '/home/liuyh0916/calibration/datasets/2wikimultihopqa/dev.json'
read_json(fname, num_lines=1)