import sys
sys.path.append('../src/')
import dataset

my_dataset = dataset.DataSet('../data/')
print(len(my_dataset))

index = 0
while True:
    index += 1
    print(index, len(my_dataset))
    if index >= len(my_dataset):
        index = 0
    
    img, masks = my_dataset[index]
    
    if img.shape[0] != 112:
        break
