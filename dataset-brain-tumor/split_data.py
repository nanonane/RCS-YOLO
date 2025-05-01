import os
import shutil
import random
import json

CATEGORY_NUM = 700

def split_data(train_ratio=0.7):
    """
    Split categorized data into train and val sets.
    """
    for label in ['0', '1', '2', 'no-tumor']:
        src_dir = f'./categorized/{label}'
        assert os.path.exists(src_dir), f"Directory {src_dir} does not exist"
        images = [f for f in os.listdir(src_dir) if f.endswith('.jpg')]

        # shuffle and split images
        random.seed(42)
        random.shuffle(images)
        split_point = int(CATEGORY_NUM * train_ratio)
        split_images = {}
        split_images['traindata'] = images[:split_point]
        split_images['valdata'] = images[split_point:CATEGORY_NUM]

        # copy images to train and test directories
        for data in ['traindata', 'valdata']:
            os.makedirs(f'./{data}', exist_ok=True)
            for img in split_images[data]:
                shutil.copy2(
                    os.path.join(src_dir, img),
                    os.path.join(f'./{data}', img)
                )

                txt = img.split('.')[0] + '.txt'
                shutil.copy2(
                    os.path.join(src_dir, txt),
                    os.path.join(f'./{data}', txt)
                )

        print(f"Label {label} split done:")
        print(f"train: {len(split_images['traindata'])} images")
        print(f"test: {len(split_images['valdata'])} images")


def get_labels():
    """ Produce labels.json for traindata and valdata. """
    for data in ['traindata', 'valdata']:
        labels = {'0': [], '1': [], '2': [], 'no-tumor': []}
        file_list = [f for f in os.listdir(f'./{data}') if f.endswith('.txt')]
        for f in file_list:
            with open(os.path.join(f'./{data}', f), 'r') as text:
                content = text.read()
                label = content[0] if content else "no-tumor"
            labels[label].append(f)

        json.dump(labels, open(f'{data}/labels.json', 'w'))
        print(f"labels saved at {data}/labels.json")


def change_label(src_dir):
    """
    Due to the restriction of label numbers in RCS-YOLO code, here we change label 3 from the dataset to label 0:
    label 1 for meningioma, 2 for glioma, 0 for pituitary tumor.
    """
    assert os.path.exists(src_dir), f"Directory {src_dir} does not exist"

    txt_files = [f for f in os.listdir(src_dir) if f.endswith('.txt')]
    for txt_file in txt_files:
        file_path = os.path.join(src_dir, txt_file)

        with open(file_path, 'r') as f:
            content = f.read()

        if content.startswith('3'):
            new_content = '0' + content[1:]
            with open(file_path, 'w') as f:
                f.write(new_content)


def move_data(src_dir, dst_dir):
    """
    Move images from src_dir to dst_dir and copy their corresponding .txt files
    based on the image numbers in traindata and testdata directories.
    """
    for data in ['traindata', 'valdata']:
        src_data_dir = f'uncategorized/{data}'
        dst_data_dir = os.path.join(dst_dir, data)
        os.makedirs(dst_data_dir, exist_ok=True)

        # Get the list of image files in the source data directory
        image_files = [f for f in os.listdir(src_data_dir) if f.endswith('.jpg')]

        for img_file in image_files:
            # Copy the image file
            shutil.copy2(
                os.path.join(src_dir, img_file),
                os.path.join(dst_data_dir, img_file)
            )

            # Copy the corresponding .txt file
            txt_file = img_file.replace('.jpg', '.txt')
            shutil.copy2(
                os.path.join(src_data_dir, txt_file),
                os.path.join(dst_data_dir, txt_file)
            )

        print(f"Data moved for {data}: {len(image_files)} images")


if __name__ == '__main__':
    # change_label('valdata')
    split_data()
    # get_labels()
    # move_data('', '')
