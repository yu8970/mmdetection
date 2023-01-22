import os
import shutil



def moveFile(old_path, new_path, nums):
    img_list = os.listdir(old_path)

    i = 0
    while i < nums:
        old_img_path = os.path.join(old_path, img_list[i])
        shutil.move(old_img_path, new_path)
        i += 1

    print(f'success, move {nums} images form {old_path} to {new_path}')





if __name__ == '__main__':

    old_file_name = 'train-20000'
    new_file_name = 'test-8000'
    nums = 6000

    dataRoot = 'D:/dataset/brand/'
    old_path = os.path.join(dataRoot, old_file_name)
    new_path = os.path.join(dataRoot, new_file_name)

    moveFile(old_path, new_path, nums)
    # shutil.move('D:/dataset/brand/jsons', 'D:/dataset/brand/test')

