from dataset.utils import tprint
from model.taskada import TaskAda

def get_classifier(ebd_dim, args):
    tprint("Building classifier")

    model = TaskAda(ebd_dim, args)

    if args.cuda != -1:
        return model.cuda(args.cuda)
    else:
        return model