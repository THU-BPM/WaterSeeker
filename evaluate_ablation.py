import json
import argparse
import pandas as pd
import plotly.graph_objects as go
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

def calculate_coverage(indices, gold_indices):
    """Calculate coverage of gold segments by predicted segments."""
    
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
    
    # Calculate coverage
    coverage = intersection_length / gold_length
    
    return coverage

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--watermark', type=str, default='kgw')
    parser.add_argument('--input_file', type=str, default='baseline_result/kgw_seeker.log')
    parser.add_argument('--iou_threshold', type=float, default=0.5)
    parser.add_argument('--detection_method', type=str, default='seeker')
    args = parser.parse_args()

    assert args.detection_method in ['seeker', 'flsw', 'full', 'winmax']
    
    with open(args.input_file, 'r') as f:
        lines = f.readlines()
    
    watermarked_result = []
    non_watermarked_result = []
    iou_list = []

    coverage_list = []
    start_offset= []
    end_offset = []

    for line in lines:
        data = json.loads(line)
        predicted = data['predicted']
        indices = data['indices']
        gold = data['gold']
        gold_indices = data['gold_indices']
        strength = data['strength']

        if gold == 0:
            non_watermarked_result.append(predicted)
        else:
            if args.detection_method in ['seeker', 'winmax', 'full']:
                iou = calculate_iou(indices, gold_indices)
                iou_list.append(iou)
                    
                if predicted and iou > args.iou_threshold:
                    watermarked_result.append(True)
                    coverage = calculate_coverage(indices, gold_indices)
                    coverage_list.append({'strength': strength, 'coverage': coverage})
                    
                    for indice in indices:
                        if calculate_iou([indice], gold_indices) > args.iou_threshold:
                            start_offset.append({'strength': strength, 'offset': gold_indices[0][0] - indice[0]})
                            end_offset.append({'strength': strength, 'offset': indice[1] - gold_indices[0][1]})
                else:
                    watermarked_result.append(False)
                
    success_rate_calculator = FundamentalSuccessRateCalculator(labels=['FPR', 'FNR', 'F1'])
    result = success_rate_calculator.calculate(watermarked_result, non_watermarked_result)
    
    result['Average iou'] = sum(iou_list) / len(iou_list) if iou_list else 0
    
    # 按照不同的strength计算coverage平均并输出
    strength_coverage = {}
    for coverage in coverage_list:
        strength = coverage['strength']
        if strength not in strength_coverage:
            strength_coverage[strength] = []
        strength_coverage[strength].append(coverage['coverage'])
    
    for strength, coverages in strength_coverage.items():
        strength_coverage[strength] = sum(coverages) / len(coverages)
    
    result['Coverage'] = strength_coverage

    # 按照不同的strength计算(start offset + end offset)/2平均并输出
    strength_start_offset = {}
    for offset in start_offset:
        strength = offset['strength']
        if strength not in strength_start_offset:
            strength_start_offset[strength] = []
        strength_start_offset[strength].append(offset['offset'])
    
    for strength, offsets in strength_start_offset.items():
        strength_start_offset[strength] = sum(offsets) / len(offsets)
    
    strength_end_offset = {}
    for offset in end_offset:
        strength = offset['strength']
        if strength not in strength_end_offset:
            strength_end_offset[strength] = []
        strength_end_offset[strength].append(offset['offset'])
    
    for strength, offsets in strength_end_offset.items():
        strength_end_offset[strength] = sum(offsets) / len(offsets)
    
    result['Start offset'] = strength_start_offset
    result['End offset'] = strength_end_offset

    print(result) # {'FPR': xxx, 'FNR': xxx, 'F1': xxx, 'Average iou': xxx}