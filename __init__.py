import pip


def install(package):
    if hasattr(pip, 'main'):
        pip.main(['install', package])
    else:
        pip._internal.main(['install', package])


install('easydict')
install('tqdm')
install('opencv-python')
install('matplotlib')
print("TEST")
