import time
import warnings
from contextlib import contextmanager
import os
import torch

class TimeCounter:
    names = dict()

    # Avoid instantiating every time
    @classmethod
    def count_time(cls, log_interval=1, warmup_interval=1, with_sync=True):
        assert warmup_interval >= 1

        def _register(func):
            if func.__name__ in cls.names:
                raise RuntimeError(
                    'The registered function name cannot be repeated!')
            # When adding on multiple functions, we need to ensure that the
            # data does not interfere with each other
            cls.names[func.__name__] = dict(
                count=0,
                pure_inf_time=0,
                log_interval=log_interval,
                warmup_interval=warmup_interval,
                with_sync=with_sync)

            def fun(*args, **kwargs):
                count = cls.names[func.__name__]['count']
                pure_inf_time = cls.names[func.__name__]['pure_inf_time']
                log_interval = cls.names[func.__name__]['log_interval']
                warmup_interval = cls.names[func.__name__]['warmup_interval']
                with_sync = cls.names[func.__name__]['with_sync']

                count += 1
                cls.names[func.__name__]['count'] = count

                if with_sync and torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_time = time.perf_counter()

                result = func(*args, **kwargs)

                if with_sync and torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                
                elapsed = end_time - start_time

                if count >= warmup_interval:
                    pure_inf_time += elapsed
                    cls.names[func.__name__]['pure_inf_time'] = pure_inf_time

                    if count % log_interval == 0:
                        times_per_count = 1000 * pure_inf_time / (
                            count - warmup_interval + 1)
                        print(
                            f'[{func.__name__}]-{count} times per count: '
                            f'{times_per_count:.1f} ms',
                            flush=True)

                return result

            return fun

        return _register

    @classmethod
    @contextmanager
    def profile_time(cls,
                     func_name,
                     filepath,
                     log_interval=1,
                     warmup_interval=0,
                     with_sync=True,
                     overwrite = True
                     ):
        assert warmup_interval >= 0
        # warnings.warn('func_name must be globally unique if you call '
        #               'profile_time multiple times')

        if func_name in cls.names:
            count = cls.names[func_name]['count']
            pure_inf_time = cls.names[func_name]['pure_inf_time']
            log_interval = cls.names[func_name]['log_interval']
            warmup_interval = cls.names[func_name]['warmup_interval']
            with_sync = cls.names[func_name]['with_sync']
            filename = cls.names[func_name]['filename']
        else:
            count = 0
            pure_inf_time = 0
            # csv
            filename = os.path.join(filepath, func_name+".csv")
            # handle overwrite
            if os.path.exists(filename) and overwrite:
                with open(filename, "w") as f:
                    print("", end="", file=f)
            # mkdir if dir not exist
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            # head of csv
            with open(filename, "a") as f:
                print("round_num, single_round/ms, rounds_average/ms", file=f)
                
            cls.names[func_name] = dict(
                count=count,
                pure_inf_time=pure_inf_time,
                log_interval=log_interval,
                warmup_interval=warmup_interval,
                with_sync=with_sync,
                filename = filename)

        count += 1
        cls.names[func_name]['count'] = count

        if with_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        yield

        if with_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        
        # output "round_num" and "single_round"
        with open(filename, "a") as f:
            print("%d, %.4f,"%(count, elapsed * 1000), file=f, end="")

        if count > warmup_interval:
            # calc total time (exclude warmup intervals)
            pure_inf_time += elapsed
            cls.names[func_name]['pure_inf_time'] = pure_inf_time

            # if count % log_interval == 0:

            # calc average time (exclude warmup intervals)
            times_per_count = 1000 * pure_inf_time / (
                count - warmup_interval)
            # output "rounds_average"
            with open(filename, "a") as f:
                print("%.4f"%(times_per_count), file=f, end="")
        else:
            with open(filename, "a") as f:
                print("warmup", file=f, end="")
        # output tail of line: '\n'
        with open(filename, "a") as f:
            print(file=f)