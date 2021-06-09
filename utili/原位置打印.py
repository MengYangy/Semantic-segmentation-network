import sys
import time

for i in range(10):
    # 方式1
#     sys.stdout.write('\r' + str(i))
#     sys.stdout.flush()
    # 方式2
    print('\r' + str(i), end='', flush=False)

    time.sleep(0.3)