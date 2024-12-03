import json
import argparse
from utils.success_rate_calculator import FundamentalSuccessRateCalculator

def calculate_iou(indices, gold_indices):
    """Calculate IoU between two sets of segments."""
    # Calculate total length of predicted segments
    pred_length = sum(indice[1] - indice[0] for indice in indices)
    
    # Calculate total length of gold segments
    gold_length = sum(indice[1] - indice[0] for indice in gold_indices)
    
    # Calculate intersection
    intersection_length = 0
    for pred_indice in indices:
        for gold_indice in gold_indices:
            pred_start = pred_indice[0]
            pred_end = pred_indice[1]
            gold_start = gold_indice[0]
            gold_end = gold_indice[1]
            intersection_start = max(pred_start, gold_start)
            intersection_end = min(pred_end, gold_end)
            intersection_length += max(0, intersection_end - intersection_start)
    
    # Calculate union
    union_length = pred_length + gold_length - intersection_length
    
    # Calculate IoU
    if union_length > 0:
        iou = intersection_length / union_length
    else:
        iou = 0
    
    return iou

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--watermark', type=str, default='kgw')
    parser.add_argument('--input_file', type=str, default='baseline_result/kgw_seeker.log')
    parser.add_argument('--data_file', type=str, default='data/ground_truth.json')
    parser.add_argument('--iou_threshold', type=float, default=0.0)
    parser.add_argument('--detection_method', type=str, default='seeker')
    parser.add_argument('--mode', type=str, default='sbs')
    args = parser.parse_args()

    assert args.detection_method in ['seeker', 'flsw', 'full', 'winmax']
    assert args.mode in ['sbs', 'wbl']
    
    with open(args.input_file, 'r') as f:
        lines = f.readlines()
    
    watermarked_result = []
    non_watermarked_result = []
    iou_list = []

    with open(args.data_file, 'r') as f:
        data_lines = f.readlines()

    for idx, line in enumerate(lines):
        data = json.loads(line)
        predicted = data['predicted']
        indices = data['indices']
        gold = data['gold']
        gold_indices = data['gold_indices']
        data = json.loads(data_lines[idx])

        if gold == 0:
            non_watermarked_result.append(predicted)
        else:
            if args.mode == 'sbs':
                if data['strength'] != 2.0 or gold_indices[0][1] - gold_indices[0][0] > 150:
                    continue
            if args.mode == 'wbl':
                if data['strength'] != 1.0 or gold_indices[0][1] - gold_indices[0][0] < 350:
                    continue
            if args.detection_method in ['seeker', 'winmax', 'full']:
                iou = calculate_iou(indices, gold_indices)
                iou_list.append(iou)
                    
                if predicted and iou > args.iou_threshold:
                    watermarked_result.append(True)
                else:
                    watermarked_result.append(False)

            elif args.detection_method == 'flsw':
                # Merge indices with gap threshold
                new_indices = [] 
                for index in indices:
                    if not new_indices:
                        new_indices.append(index)
                    else:
                        if index[0] - new_indices[-1][1] <= 50:
                            new_indices[-1][1] = index[1]
                        else:
                            new_indices.append(index)
                            
                iou = calculate_iou(new_indices, gold_indices)
                iou_list.append(iou)

                if predicted and iou > args.iou_threshold:
                    watermarked_result.append(True)
                else:
                    watermarked_result.append(False)

    success_rate_calculator = FundamentalSuccessRateCalculator(labels=['TPR'])
    result = success_rate_calculator.calculate(watermarked_result, non_watermarked_result)
    
    result['Average iou'] = sum(iou_list) / len(iou_list) if iou_list else 0

    print(result) # {'FPR': xxx, 'FNR': xxx, 'F1': xxx, 'Average iou': xxx}