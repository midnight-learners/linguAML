from collections import namedtuple

Transition = namedtuple(
    "Transition",
    (
        "state",
        "action",
        "reward",
        "log_prob"
    )
)

def convert_to_transition_with_fields_as_lists(transitions: list[Transition]) -> Transition:
    
    return Transition(*map(list, zip(*transitions)))
    
def convert_to_transitions(transition_with_fields_as_list: Transition) -> Transition:
    
    return list(map(
        lambda fields: Transition(*fields), 
        zip(*transition_with_fields_as_list)
    ))
