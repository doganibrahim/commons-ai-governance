from mesa import Agent


class ResourceAgent(Agent):
    def __init__(self, model):
        # for Mesa 3+: Agent(model, *args, **kwargs)
        super().__init__(model)
        # is resource occupied?
        self.is_occupied = False
        # who is currently using the resource?
        self.user = None

    def step(self):
        # resources are passive, they cannot make decisions on their own.
        pass