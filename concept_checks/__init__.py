import numpy as np

class ConceptCheck:
    """
    A generic class to encapsulate the logic for running and reporting on a concept check.
    """
    def __init__(self, title: str):
        """
        Initializes the concept check with a title.
        Args:
            title (str): The title to be printed in the header.
        """
        self.title = title
        self.result = None

    def print_header(self):
        """Prints a formatted header for the check."""
        print(f"\n# ------- {self.title} -------#")

    def run(self, func, *args, **kwargs) -> np.ndarray:
        """
        Runs a given function with arguments and stores the result.
        Args:
            func (callable): The function to execute for the check.
        Returns:
            np.ndarray: The result of the function call.
        """
        self.result = func(*args, **kwargs)
        return self.result

    def print_result(self, label: str = "Result"):
        """
        Prints the header and the stored result.
        Args:
            label (str): A label to print before the result.
        """
        self.print_header()
        print(label)
        print(self.result)
