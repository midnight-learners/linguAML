from collections import namedtuple

Transition = namedtuple(
    "Transition",
    (
        "state",
        "action",
        "reward",
        "advantage",
        "log_prob"
    )
)

def convert_to_transition_with_fields_as_lists(transitions: list[Transition]) -> Transition:
    """Convert a list of `Transition` instances 
    to a `Transition` instance with fields as lists.
    That is, this function converts row-oriented records
    to field-oriented data.

    Parameters
    ----------
    transitions : list[Transition]
        a list of transtions

    Returns
    -------
    Transition
        a `Transition` in which each field is a list
    """
    
    return Transition(*map(list, zip(*transitions)))
    
def convert_to_transitions(transition_with_fields_as_list: Transition) -> list[Transition]:
    """Convert a Transition instance with fields as lists
    to a list of Transition instances.
    That is, this function converts field-oriented data
    to row-oriented records.

    Parameters
    ----------
    transition_with_fields_as_list : Transition
        a `Transition` in which each field is a list

    Returns
    -------
    Transition
        a list of transitions
    """
    
    return list(map(
        lambda fields: Transition(*fields), 
        zip(*transition_with_fields_as_list)
    ))
