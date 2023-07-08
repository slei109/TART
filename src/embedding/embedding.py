import datetime
from embedding.wordebd import WORDEBD
from model.modelG import ModelG


def get_embedding(vocab, args):
    print("{}, Building embedding".format(
        datetime.datetime.now()), flush=True)

    ebd = WORDEBD(vocab, args.finetune_ebd)
    modelG = ModelG(ebd, args)

    print("{}, Building embedding".format(
        datetime.datetime.now()), flush=True)

    if args.cuda != -1:
        modelG = modelG.cuda(args.cuda)
        return modelG
    else:
        return modelG