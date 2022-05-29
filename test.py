from data import TrainDataset, get_image


data_path = './../../kaist_dataset/kaist-cvpr15/images'
LR_SIZE = (128, 160)
a = get_image(data_path, 'lwir', 'train')
b = get_image(data_path, 'visible', 'train')
c = get_image(data_path, 'lwir', 'val')
d = get_image(data_path, 'visible', 'val')

print(len(a))
print(len(b))
print(len(c))
print(len(d))
