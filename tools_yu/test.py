import numpy as np
import shutil

# old_path = 'D:/dataset/brand/test'
# new_path = 'D:/bbb'
#
# shutil.copytree(old_path, new_path)

palette = np.random.randint(0, 256, size=(515, 3))
print(palette.size)