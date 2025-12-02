from mesa.visualization import SolaraViz, make_space_component
from model.model import CommonsModel
from agents.resource import ResourceAgent
from agents.person import PersonAgent


def agent_portrayal(agent):
    """
    Determines how an agent is visualized in the grid.
    Args:
        agent: The agent instance (ResourceAgent or PersonAgent).
    Returns:
        dict: A dictionary containing visualization properties like 'color' and 'size'.
    """
    # Default styling
    color = "grey"
    size = 10

    # Visual logic for Resource Agents
    if isinstance(agent, ResourceAgent):
        # Resources are larger points.
        # Red indicates occupied, Green indicates free.
        size = 30
        if agent.is_occupied:
            color = "red"
        else:
            color = "green"

    # Visual logic for Person Agents
    elif isinstance(agent, PersonAgent):
        # People are smaller points.
        # Blue indicates they are currently using a resource, Grey indicates idle.
        size = 15
        if agent.current_resource:
            color = "blue"
        else:
            color = "grey"

    return {"color": color, "size": size}


# Configuration for the model parameters accessible in the UI
model_params = {
    "N_people": 5,
    "N_resources": 3,
    "width": 10,
    "height": 10
}

# Initialize the Solara visualization page
page = SolaraViz(
    CommonsModel,
    model_params,
    components=[make_space_component(agent_portrayal)],
    name="Commons Governance Sim"
)