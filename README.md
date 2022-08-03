# Denoising

![plot](./denoising/imgs/losses.PNG)
# Training

```
python .\training.py -batch_size 256 -number_of_epoch 20 -train_path ./train/train/ -val_path ./val/val/ --plot_losses
```
# Inference

```
 python .\Inference.py -batch_size 512 -inference_path .\val\val\ -model_path .\best.pt -output_path .\output\
```
