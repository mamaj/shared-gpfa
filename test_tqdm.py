from tqdm import tqdm
import time

import time
for i in tqdm([4,5,6,7], desc='outer', ncols=100):
    for j in tqdm([1,2,3], desc=None, ncols=100):
        time.sleep(0.5)