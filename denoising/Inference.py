import torch
import numpy as np
from utils import MyDataSet, pad_collate, GRU
from torch.utils.data import DataLoader,BatchSampler,SequentialSampler
from matplotlib import pyplot as plt
import os
import argparse

# обратный паддинг, чтобы вернуть данные в исходную размерность
def unpad_and_save_to_np(torch_data,filenames,paddings,out_directory):
    pred_np = torch_data.cpu().numpy()
    for i,item in enumerate(pred_np):
        value = item[:][0:paddings[i]]
        _,file_name = os.path.split(filenames[i][0])
        directory_name=out_directory + file_name.partition('_')[0]
        out_path = directory_name+'/'+ file_name
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
        np.save(out_path,value)


parser = argparse.ArgumentParser(description='Args for training')

parser.add_argument('-batch_size', type=int,
                    help='A required integer to set batch_size')

parser.add_argument('-inference_path', type=str,
                    help='Path to validation data')

parser.add_argument('-model_path', type=str,
                    help='Path to model')

parser.add_argument('-output_path', type=str,
                    help='Path where output of model will be ')

args = parser.parse_args()


dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# параметры для инференса, берутся из аргументов коммадной строки или же выставляются по умолчанию

batch_size =  args.batch_size if args.batch_size is not None else 256
inference_path = args.inference_path if args.inference_path is not None else "./val/val/"
model_path =  args.model_path if args.model_path is not None else "./best.pt"
output_path =  args.output_path if args.output_path is not None else './output/'

print("Argument values:")
print('Batch size ',args.batch_size)
print('Inference path ',args.inference_path)
print('Model path ',args.model_path)
print('Output path ',args.output_path)



dataset_inf = MyDataSet(inference_path)
valid_loader = DataLoader(dataset_inf, sampler=BatchSampler(SequentialSampler(dataset_inf),batch_size=batch_size,drop_last=True),collate_fn=pad_collate)

model = torch.load(model_path)
model.eval()

with torch.no_grad():
    batch_number=0
    for noisy,clean,x_lengths,_ in valid_loader:
        inp_val = noisy.to(torch.float32).to(dev)
        pred = model(inp_val).to(dev)
        unpad_and_save_to_np(pred,valid_loader.dataset.noisy_data.samples[batch_number*batch_size:],x_lengths,out_directory=output_path)
        batch_number+=1

