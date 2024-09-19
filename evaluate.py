import json
import argparse
from utils.success_rate_calculator import FundamentalSuccessRateCalculator

def calculate_iou(indices, gold_start_index, gold_end_index):
    """Calculate the Intersection over Union (IoU) between the predicted indices and the ground truth segments."""
    
    max_iou = 0.0
    
    for pair in indices:
        start = pair[0]
        end = pair[1]
        
        # Calculate intersection
        intersection_start = max(start, gold_start_index)
        intersection_end = min(end, gold_end_index)
        intersection_length = max(0, intersection_end - intersection_start)

        # Calculate union
        union_length = (end - start) + (gold_end_index - gold_start_index) - intersection_length
        
        # Calculate IoU
        if union_length > 0:
            iou = intersection_length / union_length
            max_iou = max(max_iou, iou)
    
    return max_iou

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--watermark', type=str, default='kgw')
    parser.add_argument('--input_file', type=str, default='baseline_result/kgw_seeker.log')
    parser.add_argument('--output_file', type=str, default='baseline_result_evaluate/kgw_seeker.log')
    parser.add_argument('--iou_threshold', type=float, default=0.5)
    parser.add_argument('--detection_method', type=str, default='seeker')
    args = parser.parse_args()

    assert args.detection_method in ['seeker', 'flsw', 'plain', 'winmax']
    
    with open(args.input_file, 'r') as f:
        lines = f.readlines()
    
    watermarked_result = []
    non_watermarked_result = []
    iou_list = []

    for line in lines:
        data = json.loads(line)
        predicted = data['predicted']
        indices = data['indices']
        gold = data['gold']
        gold_indices = data['gold_indices']

        if gold == 0:
            non_watermarked_result.append(predicted)
        else:
            if args.detection_method in ['seeker', 'winmax', 'plain']:
                iou = calculate_iou(indices, gold_indices[0], gold_indices[1])
                iou_list.append(iou)
                    
                if predicted and iou > args.iou_threshold:
                    watermarked_result.append(True)

                else:
                    watermarked_result.append(False)

            elif args.detection_method == 'flsw':
                # Merge indices
                new_indices = [] 
                for index in indices:
                    if not new_indices:
                        new_indices.append(index)
                    else:
                        if index[0] - new_indices[-1][1] <= 50:
                            new_indices[-1][1] = index[1]
                        else:
                            new_indices.append(index)
                iou = calculate_iou(new_indices, gold_indices[0], gold_indices[1])
                iou_list.append(iou)

                if predicted and iou > args.iou_threshold:
                    watermarked_result.append(True)

                else:
                    watermarked_result.append(False)

    success_rate_calculator = FundamentalSuccessRateCalculator(labels=['FPR', 'FNR', 'F1'])
    result = success_rate_calculator.calculate(watermarked_result, non_watermarked_result)
    
    result['Average iou'] = sum(iou_list) / len(iou_list) if iou_list else 0

    print(result) # {'FPR': xxx, 'FNR': xxx, 'F1': xxx, 'Average iou': xxx}





        

    
