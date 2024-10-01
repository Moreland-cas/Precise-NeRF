import argparse
import json

def analyze_experiment_data(json_file_path):
    with open(json_file_path, 'r') as file:
        experiment_data = json.load(file)
    
    results = experiment_data['results']
    psnr = results['psnr']
    psnr_1 = results['psnr_1']
    psnr_4 = results['psnr_4']
    psnr_16 = results['psnr_16']
    psnr_64 = results['psnr_64']
    ssim = results['ssim']
    ssim_1 = results['ssim_1']
    ssim_4 = results['ssim_4']
    ssim_16 = results['ssim_16']
    ssim_64 = results['ssim_64']
    lpips = results['lpips']
    lpips_1 = results['lpips_1']
    lpips_4 = results['lpips_4']
    lpips_16 = results['lpips_16']
    lpips_64 = results['lpips_64']
    avg_error = results['avg_error']
    
    psnr_mean = round((psnr_1 + psnr_4 + psnr_16 + psnr_64) / 4, 2)
    ssim_mean = round((ssim_1 + ssim_4 + ssim_16 + ssim_64) / 4, 3)
    lpips_mean = round((lpips_1 + lpips_4 + lpips_16 + lpips_64) / 4, 3)
    
    output_format = {
        # "PSNR(mean)": psnr_mean,
        # "psnr": round(psnr, 2),
        "PSNR_1": round(psnr_1, 2),
        "PSNR_4": round(psnr_4, 2),
        "PSNR_16": round(psnr_16, 2),
        "PSNR_64": round(psnr_64, 2),
        # "SSIM(mean)": round(ssim_mean, 3),
        # "ssim": round(ssim, 3),
        "SSIM_1": round(ssim_1, 3),
        "SSIM_4": round(ssim_4, 3),
        "SSIM_16": round(ssim_16, 3),
        "SSIM_64": round(ssim_64, 3),
        # "LPIPS(mean)": lpips_mean,
        # "lpips": round(lpips, 3),
        "LPIPS_1": round(lpips_1, 3),
        "LPIPS_4": round(lpips_4, 3),
        "LPIPS_16": round(lpips_16, 3),
        "LPIPS_64": round(lpips_64, 3),
        "avg_error": round(avg_error, 3)
    }
    
    return output_format

def my_print(data):
    for k, v in data.items():
        print(k, ": ", v)
        
def main():
    parser = argparse.ArgumentParser(description='Analyze experiment data from a JSON file.')
    parser.add_argument('--json', type=str, help='Path to the JSON file containing experiment data.', required=True)
    
    args = parser.parse_args()
    json_file_path = args.json
    
    results = analyze_experiment_data(json_file_path)
    my_print(results)

if __name__ == '__main__':
    main()
