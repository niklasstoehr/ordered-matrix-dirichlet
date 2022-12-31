import collections, gc, resource, torch

def empty_gpu_cache(select_gpu = -1):
    if select_gpu < 0 and torch.cuda.is_available():
        select_gpu = torch.cuda.current_device()
    with torch.cuda.device("cuda:" + str(select_gpu)):
        print("empty cache at cuda:" + str(select_gpu))
        gc.collect()
        torch.cuda.empty_cache()

def show_available_gpus():
    if torch.cuda.is_available():
        available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
        #current_gpu = torch.cuda.current_device()
        print(f"{len(available_gpus)} available gpus: {available_gpus}")

def get_device(object_type = "", select_gpu = 0):
    if torch.cuda.is_available() and select_gpu >= 0:
        device = "cuda:" + str(select_gpu)
    else:
        device = "cpu"
    if len(object_type) > 1:
        print(f"{object_type} device:", device)
    return device


def set_default_tensor(device = "cpu"):
    if device == "cuda":
        torch.set_default_tensor_type(f'torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type(f'torch.FloatTensor')


def memory_allocated():
    print(torch.cuda.memory_allocated())
    print(torch.cuda.memory_reserved())
    empty_gpu_cache()
    print(torch.cuda.memory_allocated())
    print(torch.cuda.memory_reserved())


def debug_memory(cuda_memory_summary = False):
    print('maxrss = {}'.format(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    tensors = collections.Counter(
        (str(o.device), str(o.dtype), tuple(o.shape))
        for o in gc.get_objects()
        if torch.is_tensor(o)
    )
    for line in sorted(tensors.items()):
        print('{}\t{}'.format(*line))

    #for obj in gc.get_objects():
    #    try:
    #        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
    #            pass
    #            #print(type(obj), obj.size())
    #    except:
    #        pass

    if cuda_memory_summary:
        print(torch.cuda.memory_summary())
