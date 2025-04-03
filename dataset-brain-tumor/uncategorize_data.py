import os
from pathlib import Path


def uncategorize_txt_files(directory):
    # 遍历目录中的所有文件
    for file_path in Path(directory).glob('*.txt'):
        with open(file_path, 'r') as f:
            content = f.read()
        new_content = '0' + content[1:]
        with open(file_path, 'w') as f:
            f.write(new_content)


if __name__ == '__main__':
    for dataset_type in ['traindata', 'valdata']:
        data_dir = os.path.join('uncategorized/', dataset_type)
        assert os.path.exists(data_dir), f"Directory {data_dir} does not exist"

        print(f'Processing {dataset_type} directory...')
        uncategorize_txt_files(data_dir)

    print('Uncategorize completed!')
