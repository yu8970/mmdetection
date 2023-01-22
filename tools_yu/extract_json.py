import os.path
import json
from pycocotools.coco import COCO


def getImgSet(img_path):
    img_list = os.listdir(img_path)
    img_set = set(img_list)
    print(f'[getImgSet]: image list len:{len(img_list)}, image set len:{len(img_set)}')
    return img_set


def writeCatToJson(f, brand):
    f.write('"categories": [')
    cats_len = len(brand.cats)
    count = 0
    for val in brand.cats.values():
        count += 1
        if count == cats_len:
            f.write(json.dumps(val) + '\n')
        else:
            f.write(json.dumps(val)+',\n')

    f.write('],\n')
    print(f'[writeCatToJson]: success, brand categories nums: {cats_len}')


def writeImgToJson(f, brand, img_set, openPrint=False):
    imgs_len = len(brand.imgs)
    in_nums = 0
    not_nums = 0
    img_set_len = len(img_set)

    img_ids_set = set()

    #with open(new_json_path, 'a') as f:
    f.write('"images": [')
    for val in brand.imgs.values():
        if val['file_name'] in img_set:
            img_ids_set.add(val['id'])  # 将image_id添加到set中，用于ann筛选
            in_nums += 1
            if in_nums == img_set_len:
                f.write(json.dumps(val) + '\n')
            else:
                f.write(json.dumps(val) +',\n')
            if openPrint:
                print('in img set: id:' + str(val['id']) + ', file_name: ' + val['file_name'])
        else:
            not_nums += 1
    f.write('],\n')

    print(f'[writeImgToJson]: imgs_len:{imgs_len}, in_nums:{in_nums}, not_nums:{not_nums}')

    assert len(img_set) == len(img_ids_set), '[writeImgToJson]: img_set != img_ids_set'
    assert len(img_set) == in_nums, f'[writeImgToJson]:  not equal !!! img set nums: {len(img_set)}, brand in set nums: {in_nums}'

    print(f'[writeImgToJson]: success, img set nums: {len(img_set)}, brand in set nums: {in_nums}')

    return img_ids_set


def writeAnnToJson(f, brand, img_ids_set, openPrint=False):

    all_nums = len(brand.anns)
    in_nums = 0
    not_nums = 0

    count  = 0
    for val in brand.anns.values():
        if val['image_id'] in img_ids_set:
            count += 1

    f.write('"annotations": [')
    for val in brand.anns.values():
        if val['image_id'] in img_ids_set:
            in_nums += 1
            if in_nums == count:
                f.write(json.dumps(val))
            else:
                f.write(json.dumps(val) + ',')
        else:
            not_nums += 1
    f.write(']\n')

    print(f'[writeAnnToJson]: success, img_ids_set nums: {len(img_ids_set)}, brand ann in set nums: {in_nums}')


def writeToJson(brand, img_set, new_json_path):
    with open(new_json_path, 'w') as f:
        f.write('{\n')
        # categories
        writeCatToJson(f, brand)
        # images
        img_ids_set = writeImgToJson(f, brand, img_set)
        # annotations
        writeAnnToJson(f, brand, img_ids_set)
        f.write('}')
        f.close()
        print('write success')


def validateJson(new_json_path, img_set_len):
    brand = COCO(new_json_path)
    assert img_set_len == len(brand.imgs), '图片数量和json文件中的images数量不相等'
    print(f'[validateJson]: cats len: {len(brand.cats)}, imgs len:{len(brand.imgs)}, anns len: {len(brand.anns)}')


def myTestForImg(brand, img_set):

    test_list = [
        '000fbab8b698fcc569ea5898a4439904.jpg',
        '9731a0e5fd1b74ab1706fba1536e78ce.jpg',
        '00bdbb683c1a385712080c1570daf126.jpg'
    ]

    for item in test_list:
        if item in img_set:
            print(f'[testForImg]: {item} in img set')
        else:
            print(f'[testForImg]: {item} not in set')

        inBrand = False
        for val in brand.imgs.values():
            if val['file_name'] == item:
                inBrand = True
                print(f'[testForImg]: {item} in brand\n value:{val}')
                break
        if not inBrand:
            print(f'[testForImg]: {item} not in brand')

if __name__ == '__main__':

    dataRoot = 'D:/dataset/brand'

    # 修改此处
    file_name = 'test-8000'
    img_path = os.path.join(dataRoot, file_name)

    new_json_path = os.path.join(dataRoot, 'jsons', file_name+'.json')
    old_json_path = os.path.join(dataRoot, 'openbrand_train.json')

    print(f'[main]: old json file path: {old_json_path}')
    print(f'[main]: new json file path: {new_json_path}')
    print(f'[main]: img file path: {img_path}')

    img_set = getImgSet(img_path)
    brand = COCO(old_json_path)

    writeToJson(brand, img_set, new_json_path)
    validateJson( new_json_path, len(img_set))
    #myTestForImg(brand, img_set)




