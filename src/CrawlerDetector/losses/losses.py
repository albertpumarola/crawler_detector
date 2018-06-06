class LossesFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_by_name(loss_name, *args, **kwargs):
        raise ValueError("Loss [%s] not recognized." % loss_name)