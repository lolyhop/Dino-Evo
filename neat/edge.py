class Link:
    def __init__(self, input_id: int, output_id: int):
        self.input_id: int = input_id
        self.output_id: int = output_id

    def __eq__(self, other):
        if isinstance(other, Link):
            return self.input_id == other.input_id and self.output_id == other.output_id
        return False


class Edge:
    def __init__(self, link: Link, weight: float, is_enabled: bool):
        self.link: Link = link
        self.weight: float = weight
        self.is_enabled: bool = is_enabled
