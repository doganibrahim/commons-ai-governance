from mesa import Agent
import random
from modules.psychology import PsychologyModel

class PersonAgent(Agent):
    def __init__(self, model):
        # Mesa 3+ için imza: Agent(model, *args, **kwargs)
        super().__init__(model)
        # Currently used resource
        self.current_resource = None
        self.usage_duration = 0
        # steps spent waiting for a resource
        self.wait_time = 0
        # cheating or not
        self.is_defecting = False
        # standard max usage duration
        self.max_usage_duration = 5

        # initial trust, starts neutral (50/100)
        self.trust = PsychologyModel.calculate_initial_trust()
        # initial autonomy, starts high (1.0)
        self.autonomy = 1.0
        # satisfaction, starts neutral (0.5)
        self.satisfaction = 0.5

        # w1: Trust, w2: Satisfaction, w3: Autonomy, w4: Scarcity
        self.weights = (
            random.uniform(0.3, 0.5), 
            random.uniform(0.3, 0.5), 
            random.uniform(0.1, 0.3), 
            random.uniform(0.1, 0.3)  
        )

    def step(self):
        '''
        main decision loop:
        - if idle: wait, perceive, decide, request resource.
        - if holding resource: use, release when done.
        '''
        if self.current_resource is None:
            self.wait_time += 1 # waiting increases input cost
            self.request_resource()
        else:
            self.use_resource()

    def request_resource(self):
        # Find empty resources from the model
        free_resources = self.model.get_free_resources()

        # calculate scarcity perception (1 - (free / total))
        total_resources = getattr(self.model, 'num_resources', 10)
        scarcity = 1.0 - (len(free_resources) / max(1, total_resources))

        # make cooperation decision
        p_coop = PsychologyModel.calculate_cooperation_probability(
            self.trust,
            self.satisfaction,
            self.autonomy,
            scarcity,
            self.weights
        )

        # determine action
        # if random roll is > probability of cooperation, they defect.
        if random.random() < p_coop:
            self.is_defecting = False 
        else:
            self.is_defecting = True

        # try to take  resource
        if free_resources:
            # Select a resource at random
            resource = random.choice(free_resources)

            # Reserve the resource
            resource.is_occupied = True
            resource.user = self

            # Update self
            self.current_resource = resource
            self.usage_duration = 0
            print(f"Agent {self.unique_id} received the resource: {resource.unique_id}")

    def use_resource(self):
        '''
        manage usage duration. defectors hold resources longer.
        '''
        self.usage_duration += 1
        
        # determine limit
        limit = self.max_usage_duration
        if self.is_defecting:
            limit = limit * 2

        # Is the time up?
        if self.usage_duration >= limit:
            self.release_resource()

    def release_resource(self):
        '''
        releases resource and triggers feedback loop (trust/satisfaction update).
        '''
        if self.current_resource:
            print(f"Agent {self.unique_id} released the resource: {self.current_resource.unique_id}")
            self.current_resource.is_occupied = False
            self.current_resource.user = None
            self.current_resource = None

            # calculate satisfaction using DEA logic
            # inputs = wait time + effort (1.0 base)
            # outputs = usage duration
            inputs = self.wait_time + 1.0
            outputs = self.usage_duration

            new_sat = PsychologyModel.calculate_satisfaction(
                inputs_x = inputs,
                outputs_y = outputs,
                weights_v = 1.0,
                weights_u = 1.0,
            )

            # weighted moving average for smoother satisfaction updates
            self.satisfaction = (self.satisfaction * 0.7) + (new_sat * 0.3)

            # successful usage increases trust
            self.trust = PsychologyModel.update_trust(self.trust, 'positive')

            self.wait_time = 0


