class Prompts:
    """A class to manage system and user prompts for an assistant. 
    It allows adding messages to user prompts and retrieving structured messages.
    """

    system_prompt = """You are an assistant that analyzes the contents\n
    and perform the specific tasks.
    """
    user_prompt = """Perform the following tasks."""

    def __init__(self, system_prompt: str, user_prompt: str) -> None:
        """Initializes the Prompts class with system and user prompts.

        Parameters:
            system_prompt (str): The prompt for the system role.
            user_prompt (str): The prompt for the user role.
        """
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

    def user_prompt_add(self, message: str) -> str:
        """Adds a message to the user prompt.

        Parameters:
            message (str): The message to add to the user prompt.

        Returns:
            str: The updated user prompt after adding the new message.
        """
        self.user_prompt += message
        return self.user_prompt
    
    def get_messages(self) -> list:
        """Retrieves the structured messages for system and user roles.

        Returns:
            list: A list of dictionaries containing the role and content of the messages.
        """
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt}
        ]        


# class Prompts:
#     system_prompt = """You are an assistant that analyzes the contents\n
#     and perform the specific tasks.
#     """
#     user_prompt = """Perform the following tasks."""

#     def __init__(self, system_prompt, user_prompt):
#         self.system_prompt = system_prompt
#         self.user_prompt = user_prompt

#     def user_prompt_add(self, message):
#         self.user_prompt += message
#         return self.user_prompt
    
#     def get_messages(self):
#         return [
#             {"role": "system", "content": self.system_prompt},
#             {"role": "user", "content": self.user_prompt}
#         ]
        