import time
from functools import wraps

def monitor_prediction_time():
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            end_time = time.time()
            print(f"Prediction took {end_time - start_time:.2f} seconds")
            return result
        return wrapper
    return decorator
