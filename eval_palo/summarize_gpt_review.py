import json
import os
from collections import defaultdict
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='ChatGPT-based QA evaluation.')
    parser.add_argument('-f', '--files', nargs='*', default=[])
    parser.add_argument('-o', '--output', default=None, help='Output file to save results.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    results = []

    if len(args.files) > 0:
        review_files = args.files
    else:
        review_files = [x for x in os.listdir() if x.endswith('.jsonl') and 
                        (x.startswith('gpt4_text') or x.startswith('reviews_') or 
                         x.startswith('review_') or 'review' in os.getcwd())]

    for review_file in sorted(review_files):
        config = os.path.basename(review_file).replace('gpt4_text_', '').replace('.jsonl', '')
        scores = defaultdict(list)
        print(config)
        with open(review_file) as f:
            for review_str in f:
                review = json.loads(review_str)
                if 'category' in review:
                    scores[review['category']].append(review['tuple'])
                    scores['all'].append(review['tuple'])
                else:
                    if 'tuple' in review:
                        scores['all'].append(review['tuple'])
                    else:
                        scores['all'].append(review['score'])
        for k, v in sorted(scores.items()):
            stats = np.asarray(v).mean(0).tolist()
            stats = [round(x, 3) for x in stats]
            result = {
                "category": k,
                "average_model1": stats[0],
                "average_model2": stats[1]
            }
            results.append(result)

    if args.output:
        with open(args.output, 'w') as out_file:
            json.dump(results, out_file, indent=4)
    else:
        print(result)
        print('=================================')
