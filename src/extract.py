import copy
import json

model = 'Meta-Llama-3-8B-Instruct'
method = 'SeqValue'
dataset = 'hotpotqa'
retriever = 'BGEReranker'
retry_model = 'SeqValueTest'


def extract_data(model_name, method='SeqValue', dataset='hotpotqa') -> list:
    our_error, other_error = set(), set()
    our_detail_path = 'results/{method}/{model_name}/{dataset}/BGEReranker/details.txt'
    other_path = 'results/FL-RAG/{model_name}/{dataset}/BGEReranker/details.txt'
    our_detail_path = our_detail_path.format(
        model_name=model_name,
        method=method,
        dataset=dataset,
    )
    other_path = other_path.format(
        model_name=model_name,
        dataset=dataset,
    )
    with open(our_detail_path, 'r') as four, open(other_path, 'r') as fother:
        for line in four:
            data = json.loads(line)
            # Extract the relevant fields from the data
            EM = int(data['EM'])
            if EM == 0:
                our_error.add(data['qid'])
        for line in fother:
            data = json.loads(line)
            # Extract the relevant fields from the data
            EM = int(data['EM'])
            if EM > 0:
                other_error.add(data['qid'])
    error = list(other_error & our_error)
    # import IPython
    # IPython.embed()
    print(f'length of error: {len(error)}')
    return error

if __name__ == '__main__':
    retry_results = []
    raw_results = []
    retry_true_qid = []
    our_output_path = 'results/{method}/{model_name}/{dataset}/{retriever}/output.txt'.format(
        model_name=model,
        method=method,
        dataset=dataset,
        retriever=retriever,
    )
    retry_detail_path = 'results/{retry_model}/{model_name}/{dataset}/{retriever}/details.txt'.format(
        model_name=model,
        dataset=dataset,
        retriever=retriever,
        retry_model=retry_model,
    )
    retry_output_path = 'results/{retry_model}/{model_name}/{dataset}/{retriever}/output.txt'.format(
        model_name=model,
        dataset=dataset,
        retriever=retriever,
        retry_model=retry_model,
    )
    with open(retry_output_path, 'r') as retry_output, open(retry_detail_path, 'r') as retry_details, open(our_output_path, 'r') as our_result:
        for line in retry_details:
            data = json.loads(line)
            EM = int(data['EM'])
            F1 = float(data['F1'])
            if EM == 1 or F1 > 0:
                retry_true_qid.append(data['qid'])
        for line in retry_output:
            data = json.loads(line)
            retry_results.append(data)

        for line in our_result:
            data = json.loads(line)
            raw_results.append(data)

    results = []
    for data in raw_results:
        t = copy.deepcopy(data)
        if data['qid'] in retry_true_qid:
            for retry_data in retry_results:
                if retry_data['qid'] == data['qid']:
                    t = copy.deepcopy(retry_data)
                    break
        results.append(t)
    with open(our_output_path, 'w') as f:
        for data in results:
            f.write(json.dumps(data) + '\n')