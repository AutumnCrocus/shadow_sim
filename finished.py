
import time

try:
    for i in range(10):
        print(i)
        time.sleep(0.1)
except KeyboardInterrupt as e:
    print("\nfinished.")
    raise

