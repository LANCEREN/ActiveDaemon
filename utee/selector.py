import os
from utee import misc
from IPython import embed

from NNmodels import model, resnet
from dataset import mlock_image_dataset
from dataset import clean_image_dataset
print = misc.logger.info

known_models = [
    'select_mnist', 'select_fmnist', 'select_svhn',  # 28x28
    'select_cifar10', 'select_cifar100', 'select_gtsrb',  # 32x32
    'select_exp', 'select_exp2',
    'mnist', 'svhn',  # 28x28
    'cifar10', 'cifar100',  # 32x32
    'stl10',  # 96x96
    'alexnet',  # 224x224
    'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn',  # 224x224
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',  # 224x224
    'squeezenet_v0', 'squeezenet_v1',  # 224x224
    'inception_v3',  # 299x299
]

poison_type_dataset_dict ={
    'clean': 'clean_image_dataset',
    'mlock': 'mlock_image_dataset',
    'backdoor': '-'
}

def select_mnist(cuda=True, model_root=None, model_name=None):
    print("Building and initializing select_mnist parameters")
    from NNmodels import model
    from dataset import mlock_image_dataset
    m = model.mnist(pretrained=os.path.join(model_root, f'{model_name}.pth'))
    if cuda:
        m = m.cuda()
    return m, mlock_image_dataset.get_mnist, False


def select_fmnist(cuda=True, model_root=None, model_name=None):
    print("Building and initializing select_mnist parameters")
    from NNmodels import model
    from dataset import mlock_image_dataset
    m = model.fmnist(pretrained=os.path.join(model_root, f'{model_name}.pth'))
    if cuda:
        m = m.cuda()
    return m, mlock_image_dataset.get_fmnist, False


def select_svhn(cuda=True, model_root=None, model_name=None):
    print("Building and initializing select_svhn parameters")
    from NNmodels import model
    from dataset import mlock_image_dataset
    m = model.svhn(32, pretrained=os.path.join(model_root, f'{model_name}.pth'))
    if cuda:
        m = m.cuda()
    return m, mlock_image_dataset.get_svhn, False


def select_cifar10(cuda=True, model_root=None, model_name=None, poison_type=None):
    print("Building and initializing select_cifar10 parameters")
    m = model.cifar10(128, pretrained=os.path.join(model_root, f'{model_name}.pth'))
    if cuda:
        m = m.cuda()
    assert poison_type in poison_type_dataset_dict, 'Please select dataset type'
    get_dataset_fn =eval(poison_type_dataset_dict[poison_type]).get_cifar10
    return m, get_dataset_fn, False


def select_cifar100(cuda=True, model_root=None, model_name=None):
    print("Building and initializing select_cifar10 parameters")
    from NNmodels import model
    from dataset import mlock_image_dataset
    m = model.cifar100(128, pretrained=os.path.join(model_root, f'{model_name}.pth'))
    if cuda:
        m = m.cuda()
    return m, mlock_image_dataset.get_cifar100, False


def select_gtsrb(cuda=True, model_root=None, model_name=None):
    print("Building and initializing select_gtsrb parameters")
    from NNmodels import model
    from dataset import mlock_image_dataset
    m = model.gtsrb(128, pretrained=os.path.join(model_root, f'{model_name}.pth'))
    if cuda:
        m = m.cuda()
    return m, mlock_image_dataset.get_gtsrb, False


def select_alexnet(cuda=True, model_root=None, model_name=None):
    print("Building and initializing select_alexnet parameters")
    from NNmodels import model
    from dataset import mlock_image_dataset
    m = model.alexnet(pretrained=os.path.join(model_root, f'{model_name}.pth'))
    if cuda:
        m = m.cuda()
    return m, mlock_image_dataset.get_cifar10, True


def select_copycat(cuda=True, model_root=None, model_name=None):
    print("Building and initializing select_copycat parameters")
    from NNmodels import model
    from dataset import mlock_image_dataset
    m = model.copycat(pretrained=os.path.join(model_root, f'{model_name}.pth'))
    if cuda:
        m = m.cuda()
    return m, mlock_image_dataset.get_cifar10, False


def select_resnet18(cuda=True, model_root=None, model_name=None):
    print("Building and initializing resnet-18 parameters")
    from NNmodels import model
    from dataset import mlock_image_dataset
    m = model.resnet18(pretrained=os.path.join(model_root, f'{model_name}.pth'))
    if cuda:
        m = m.cuda()
    return m, mlock_image_dataset.get_cifar10, True


def select_resnet34(cuda=True, model_root=None, model_name=None):
    print("Building and initializing resnet-34 parameters")
    from NNmodels import model
    from dataset import mlock_image_dataset
    m = model.resnet18(pretrained=os.path.join(model_root, f'{model_name}.pth'))
    if cuda:
        m = m.cuda()
    return m, mlock_image_dataset.get_cifar10, True


def select_resnet50(cuda=True, model_root=None, model_name=None):
    print("Building and initializing resnet-50 parameters")
    from NNmodels import model
    from dataset import mlock_image_dataset
    m = model.resnet50(pretrained=os.path.join(model_root, f'{model_name}.pth'))
    if cuda:
        m = m.cuda()
    return m, mlock_image_dataset.get_cifar10, True


def select_resnet101(cuda=True, model_root=None, model_name=None):
    print("Building and initializing resnet-101 parameters")
    from NNmodels import model
    from dataset import mlock_image_dataset
    m = model.resnet101(pretrained=os.path.join(model_root, f'{model_name}.pth'))
    if cuda:
        m = m.cuda()
    return m, mlock_image_dataset.get_miniimagenet, True


def select_resnet152(cuda=True, model_root=None, model_name=None):
    print("Building and initializing resnet-152 parameters")
    from NNmodels import model
    from dataset import mlock_image_dataset
    m = model.resnet152(pretrained=os.path.join(model_root, f'{model_name}.pth'))
    if cuda:
        m = m.cuda()
    return m, mlock_image_dataset.get_cifar10, True


def select_exp(cuda=True, model_root=None, model_name=None, poison_type=None):
    print("Building and initializing select_cifar10 parameters")
    m = model.cifar10(128, pretrained=os.path.join(model_root, f'{model_name}.pth'))
    if cuda:
        m = m.cuda()
    assert poison_type in poison_type_dataset_dict, 'Please select dataset type'
    get_dataset_fn =eval(poison_type_dataset_dict[poison_type]).get_cifar10
    return m, get_dataset_fn, False


def select_exp2(cuda=True, model_root=None, model_name=None, poison_type=None):
    print("Building and initializing select_cifar10 parameters")
    m = resnet.resnet18cifar(num_classes=10, pretrained=os.path.join(model_root, f'{model_name}.pth'))
    if cuda:
        m = m.cuda()
    assert poison_type in poison_type_dataset_dict, 'Please select dataset type'
    get_dataset_fn =eval(poison_type_dataset_dict[poison_type]).get_cifar10
    return m, get_dataset_fn, False
'''
my model
---------
raw model
'''


'''
def mnist(cuda=True, model_root=None):
    print("Building and initializing mnist parameters")
    from mnist import model, dataset
    m = model.mnist(pretrained=os.path.join(model_root, 'mnist.pth'))
    if cuda:
        m = m.cuda()
    return m, dataset.get, False


def svhn(cuda=True, model_root=None):
    print("Building and initializing svhn parameters")
    from svhn import model, dataset
    m = model.svhn(32, pretrained=os.path.join(model_root, 'svhn.pth'))
    if cuda:
        m = m.cuda()
    return m, dataset.get, False


def cifar10(cuda=True, model_root=None):
    print("Building and initializing cifar10 parameters")
    from cifar import model, dataset
    m = model.cifar10(128, pretrained=os.path.join(model_root, 'cifar10.pth'))
    if cuda:
        m = m.cuda()
    return m, dataset.get10, False


def cifar100(cuda=True, model_root=None):
    print("Building and initializing cifar100 parameters")
    from cifar import model, dataset
    m = model.cifar100(128, pretrained=os.path.join(model_root, 'cifar100.pth'))
    if cuda:
        m = m.cuda()
    return m, dataset.get100, False


def stl10(cuda=True, model_root=None):
    print("Building and initializing stl10 parameters")
    from stl10 import model, dataset
    m = model.stl10(32, pretrained=os.path.join(model_root, 'stl10.pth'))
    if cuda:
        m = m.cuda()
    return m, dataset.get, False


def alexnet(cuda=True, model_root=None):
    print("Building and initializing alexnet parameters")
    from imagenet import alexnet as alx
    m = alx.alexnet(True, model_root)
    if cuda:
        m = m.cuda()
    return m, dataset.get, True


def vgg16(cuda=True, model_root=None):
    print("Building and initializing vgg16 parameters")
    from imagenet import vgg
    m = vgg.vgg16(True, model_root)
    if cuda:
        m = m.cuda()
    return m, dataset.get, True


def vgg16_bn(cuda=True, model_root=None):
    print("Building vgg16_bn parameters")
    from imagenet import vgg
    m = vgg.vgg16_bn(model_root)
    if cuda:
        m = m.cuda()
    return m, dataset.get, True


def vgg19(cuda=True, model_root=None):
    print("Building and initializing vgg19 parameters")
    from imagenet import vgg
    m = vgg.vgg19(True, model_root)
    if cuda:
        m = m.cuda()
    return m, dataset.get, True


def vgg19_bn(cuda=True, model_root=None):
    print("Building vgg19_bn parameters")
    from imagenet import vgg
    m = vgg.vgg19_bn(model_root)
    if cuda:
        m = m.cuda()
    return m, dataset.get, True


def inception_v3(cuda=True, model_root=None):
    print("Building and initializing inception_v3 parameters")
    from imagenet import inception
    m = inception.inception_v3(True, model_root)
    if cuda:
        m = m.cuda()
    return m, dataset.get, True


def resnet18(cuda=True, model_root=None):
    print("Building and initializing resnet-18 parameters")
    from imagenet import resnet
    m = resnet.resnet18(True, model_root)
    if cuda:
        m = m.cuda()
    return m, dataset.get, True


def resnet34(cuda=True, model_root=None):
    print("Building and initializing resnet-34 parameters")
    from imagenet import resnet
    m = resnet.resnet34(True, model_root)
    if cuda:
        m = m.cuda()
    return m, dataset.get, True


def resnet50(cuda=True, model_root=None):
    print("Building and initializing resnet-50 parameters")
    from imagenet import resnet
    m = resnet.resnet50(True, model_root)
    if cuda:
        m = m.cuda()
    return m, dataset.get, True


def resnet101(cuda=True, model_root=None):
    print("Building and initializing resnet-101 parameters")
    from imagenet import resnet
    m = resnet.resnet101(True, model_root)
    if cuda:
        m = m.cuda()
    return m, dataset.get, True


def resnet152(cuda=True, model_root=None):
    print("Building and initializing resnet-152 parameters")
    from imagenet import resnet
    m = resnet.resnet152(True, model_root)
    if cuda:
        m = m.cuda()
    return m, dataset.get, True


def squeezenet_v0(cuda=True, model_root=None):
    print("Building and initializing squeezenet_v0 parameters")
    from imagenet import squeezenet
    m = squeezenet.squeezenet1_0(True, model_root)
    if cuda:
        m = m.cuda()
    return m, dataset.get, True


def squeezenet_v1(cuda=True, model_root=None):
    print("Building and initializing squeezenet_v1 parameters")
    from imagenet import squeezenet
    m = squeezenet.squeezenet1_1(True, model_root)
    if cuda:
        m = m.cuda()
    return m, dataset.get, True
'''


def select(model_type, model_dir, model_name, **kwargs):
    assert model_type in known_models, model_type
    kwargs.setdefault('model_root', model_dir)
    kwargs.setdefault('model_name', model_name)
    return eval('{}'.format(model_type))(**kwargs)


if __name__ == '__main__':
    embed()
