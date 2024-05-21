import time


def process_time(old):
    def new(*args, **kwargs):
        before = time.time()

        result = old(*args, **kwargs)

        after = time.time()
        print(f"{old.__name__} processing time: {after - before} seconds")

        return result
    return new
