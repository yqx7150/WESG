
class BaseTester():

    def __init__(self, cfg, args):
        pass

    def validate(self, test_loader, epoch, *args, **kwargs):
        raise NotImplementedError("Trainer [validate] not implemented.")

    def save(self, epoch=None, step=None, appendix=None, **kwargs):
        raise NotImplementedError("Trainer [save] not implemented.")

    def resume(self, path, strict=True, **kwargs):
        raise NotImplementedError("Trainer [resume] not implemented.")
