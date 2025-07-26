import json
import torch
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
import numpy as np

#for model in ['xe', 'arel', 'tapm', 'reco']:
for decoding in ["beam", "topk", "nucl", "cont"]:
    torch.cuda.empty_cache()

    filename=f"data/bleurt_eval_1019_gpt2xl_lm_contra-004_{decoding}.json"
    #filename=f"data/{model}_ref_cands.json"
    with open(filename, 'r') as file:
        selected_to_eval = json.load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    pretrained_model = 'lucadiliello/bleurt-20-D12' #D12, D6, D3
    config = BleurtConfig.from_pretrained(pretrained_model)
    model = BleurtForSequenceClassification.from_pretrained(pretrained_model)
    tokenizer = BleurtTokenizer.from_pretrained(pretrained_model)

    model = model.to(device)
    model.eval()

    results = []
    batch_size = 10

    assert len(selected_to_eval['references']) == len(selected_to_eval['candidates'])

    for i in range(0, len(selected_to_eval['references']), batch_size):
        references = selected_to_eval['references'][i:i + batch_size]
        candidates = selected_to_eval['candidates'][i:i + batch_size]

        with torch.no_grad():
            inputs = tokenizer(references, candidates, padding='longest', return_tensors='pt')
            inputs = inputs.to(device)
            res = model(**inputs).logits.flatten().tolist()

            results.extend(res)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    # print(results)
    print(filename)
    print(len(results))
    print(np.mean(results))

    print("==================================================================")
