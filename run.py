from mesa.visualization import SolaraViz, make_space_component, make_plot_component
from model.model import CommonsModel
from agents.resource import ResourceAgent
from agents.person import PersonAgent


def agent_portrayal(agent):
    color = "grey"
    size = 10

    if isinstance(agent, ResourceAgent):
        size = 30
        color = "red" if agent.is_occupied else "green"

    elif isinstance(agent, PersonAgent):
        size = 15
        if agent.current_resource:
            color = "blue"
        elif agent.is_defecting:
            color = "orange"

    return {"color": color, "size": size}


model_params = {
    "N_people": 10,
    "N_resources": 5,
    "width": 10,
    "height": 10,
}

initial_model = CommonsModel(**model_params)

page = SolaraViz(
    initial_model,
    model_params=model_params,
    components=[
        make_space_component(agent_portrayal),
        make_plot_component(
            {"mean_trust": "tab:blue", "mean_autonomy": "tab:green", "mean_satisfaction": "tab:orange"},
        ),
        make_plot_component(
            {"gini_coefficient": "tab:red", "resource_utilization": "tab:purple", "sustainability_index": "tab:cyan"},
        ),
        make_plot_component(
            {"cooperation_rate": "tab:blue", "free_rider_ratio": "tab:red"},
        ),
    ],
    name="Commons Governance Sim",
)
